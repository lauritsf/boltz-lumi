// The Welford update/combine and affine expression follow PyTorch's
// layer_norm_kernel.cu at 70d99e998b4955e0049d13a98d77ae1b14db1f45.
// BSD license and upstream copyright notices: third_party/pytorch-LICENSE.
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cuda_runtime.h>
#include <type_traits>

// This specialization matches PyTorch's wave64 reduction for 128 channels.
constexpr int channels = 128;
constexpr int wave_size = 64;
constexpr int values_per_lane = 4;
constexpr int active_lanes = channels / values_per_lane;
constexpr int shared_padding = 4;
constexpr int shared_row_stride = channels + shared_padding;

struct Moments {
  float mean, sigma2, count;
};

__device__ __forceinline__ Moments update(float value, Moments a) {
  float delta = value - a.mean;
  float count = a.count + 1.f;
  float mean = a.mean + delta * (1.f / count);
  return {mean, a.sigma2 + delta * (value - mean), count};
}

// PyTorch's operand order is needed for bitwise agreement.
__device__ __forceinline__ Moments combine(Moments b, Moments a) {
  float delta = b.mean - a.mean, count = a.count + b.count;
  float mean = 0.f, variance = 0.f;
  if (count > 0.f) {
    float coef = 1.f / count, na = a.count * coef, nb = b.count * coef;
    mean = na * a.mean + nb * b.mean;
    variance = a.sigma2 + b.sigma2 + delta * delta * a.count * nb;
  }
  return {mean, variance, count};
}

template <typename T, typename Output, bool CHANNEL_FIRST, int ROWS, int WAVES>
__global__ void norm128_kernel(const T *x, const float *weight, const float *bias, Output *out,
                               int64_t m, float eps) {
  // Transpose channel-first tiles with padding between shared-memory rows.
  __shared__ float tile[CHANNEL_FIRST ? ROWS * shared_row_stride : 1];
  int lane = threadIdx.x % wave_size, wave = threadIdx.x / wave_size;
  int64_t start = int64_t(blockIdx.x) * ROWS;
  if constexpr (CHANNEL_FIRST) {
    for (int i = threadIdx.x; i < ROWS * channels; i += WAVES * wave_size) {
      int c = i / ROWS, r = i % ROWS;
      tile[r * shared_row_stride + c] = start + r < m ? float(x[int64_t(c) * m + start + r]) : 0.f;
    }
    __syncthreads();
  }
  for (int r = wave; r < ROWS; r += WAVES) {
    int64_t row = start + r;
    float values[values_per_lane] = {0.f, 0.f, 0.f, 0.f};
    Moments a{0.f, 0.f, 0.f};
    if (lane < active_lanes && row < m) {
#pragma unroll
      for (int j = 0; j < values_per_lane; ++j) {
        int c = lane * values_per_lane + j;
        if constexpr (CHANNEL_FIRST)
          values[j] = tile[r * shared_row_stride + c];
        else
          values[j] = float(x[row * channels + c]);
        a = update(values[j], a);
      }
    }
#pragma unroll
    for (int offset = wave_size / 2; offset > 0; offset /= 2) {
      Moments b{__shfl_down(a.mean, offset, wave_size), __shfl_down(a.sigma2, offset, wave_size),
                __shfl_down(a.count, offset, wave_size)};
      a = combine(a, b);
    }
    float mean = __shfl(a.mean, 0, wave_size);
    float variance = __shfl(a.sigma2, 0, wave_size) / float(channels);
    float inv = rsqrtf(variance + eps);
    if (lane < active_lanes && row < m) {
#pragma unroll
      for (int j = 0; j < values_per_lane; ++j) {
        int c = lane * values_per_lane + j;
        out[row * channels + c] = weight[c] * (inv * (values[j] - mean)) + bias[c];
      }
    }
  }
}

template <typename T, typename Output, bool CF, int W>
void launch_norm(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &bias,
                 at::Tensor &out, int64_t m, float eps, int rows, cudaStream_t stream) {
#define CASE(R)                                                                                    \
  case R:                                                                                          \
    norm128_kernel<T, Output, CF, R, W><<<(m + R - 1) / R, W * wave_size, 0, stream>>>(                   \
        x.data_ptr<T>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<Output>(), \
        m, eps);                                                                                   \
    break;
  switch (rows) { CASE(4) CASE(8) CASE(16) default : TORCH_CHECK(false, "Unsupported row tile"); }
#undef CASE
}

template <typename Output>
at::Tensor normalize(const at::Tensor &x, const at::Tensor &weight, const at::Tensor &bias,
                     double eps, bool channel_first, int waves, int rows) {
  TORCH_CHECK(x.is_cuda() && weight.device() == x.device() && bias.device() == x.device());
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous());
  TORCH_CHECK(x.dim() == 2 && (channel_first ? x.size(0) == channels : x.size(1) == channels));
  TORCH_CHECK(weight.numel() == channels && bias.numel() == channels && weight.scalar_type() == at::kFloat &&
              bias.scalar_type() == at::kFloat);
  TORCH_CHECK(!at::GradMode::is_enabled(), "Inference only");
  TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kBFloat16);
  TORCH_CHECK(eps > 0 && std::isfinite(eps) && (waves == 4 || waves == 8));
  c10::cuda::CUDAGuard guard(x.device());
  int64_t m = channel_first ? x.size(1) : x.size(0);
  auto out = at::empty(
      {m, channels}, x.options().dtype(std::is_same<Output, float>::value ? at::kFloat : at::kBFloat16));
  if (!m) return out;
  auto stream = c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
#define WAVES(T, CF)                                                                               \
  switch (waves) {                                                                                 \
  case 4:                                                                                          \
    launch_norm<T, Output, CF, 4>(x, weight, bias, out, m, float(eps), rows, stream);              \
    break;                                                                                         \
  case 8:                                                                                          \
    launch_norm<T, Output, CF, 8>(x, weight, bias, out, m, float(eps), rows, stream);              \
    break;                                                                                         \
  }
  if (x.scalar_type() == at::kFloat) {
    if (channel_first) {
      WAVES(float, true)
    } else {
      WAVES(float, false)
    }
  } else {
    if (channel_first) {
      WAVES(c10::BFloat16, true)
    } else {
      WAVES(c10::BFloat16, false)
    }
  }
#undef WAVES
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor multiplication_norm(const at::Tensor &x, const at::Tensor &weight,
                               const at::Tensor &bias, double eps, bool channel_first, int waves,
                               int rows, bool bf16_output) {
  if (bf16_output) {
    return normalize<c10::BFloat16>(x, weight, bias, eps, channel_first, waves, rows);
  }
  return normalize<float>(x, weight, bias, eps, channel_first, waves, rows);
}
