#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cuda_runtime.h>

using bf16 = c10::BFloat16;

constexpr int channels_per_operand = 128;
constexpr int projection_channels = 2 * channels_per_operand;
constexpr int joint_projection_stride = 2 * projection_channels;
constexpr int spatial_tile = 8;

__device__ __forceinline__ bf16 gated(bf16 x, bf16 gate) {
  // Round sigmoid to BF16 before multiplying, as PyTorch does.
  bf16 sigmoid(1.f / (1.f + expf(-float(gate))));
  return bf16(float(x) * float(sigmoid));
}

__global__ void gate_kernel(const bf16 *x, const bf16 *gate, bf16 *out, int64_t elements) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) out[index] = gated(x[index], gate[index]);
}

template <int TILE, int CHANNELS, bool OUTGOING, int STRIDE>
__global__ void gate_pack_kernel(const bf16 *p, const bf16 *g, const float *mask, bf16 *a, bf16 *b,
                                 int n) {
  // Pad the transposed shared-memory rows by one element.
  __shared__ bf16 values[2][CHANNELS][TILE][TILE + 1];
  int i0 = blockIdx.x * TILE, j0 = blockIdx.y * TILE, c0 = blockIdx.z * CHANNELS;
  for (int index = threadIdx.x; index < TILE * TILE * CHANNELS; index += blockDim.x) {
    int c = index % CHANNELS, j = (index / CHANNELS) % TILE, i = index / (CHANNELS * TILE);
    int64_t pair = int64_t(i0 + i) * n + j0 + j;
    if (i0 + i < n && j0 + j < n) {
      float m = mask[pair];
      bf16 av = gated(p[pair * STRIDE + c0 + c], g[pair * STRIDE + c0 + c]);
      bf16 bv = gated(p[pair * STRIDE + channels_per_operand + c0 + c],
                      g[pair * STRIDE + channels_per_operand + c0 + c]);
      values[0][c][i][j] = bf16(float(av) * m);
      values[1][c][i][j] = bf16(float(bv) * m);
    } else {
      values[0][c][i][j] = bf16(0.f);
      values[1][c][i][j] = bf16(0.f);
    }
  }
  __syncthreads();
  for (int index = threadIdx.x; index < TILE * TILE * CHANNELS; index += blockDim.x) {
    int j = index % TILE, i = (index / TILE) % TILE, c = index / (TILE * TILE);
    int64_t plane = int64_t(c0 + c) * n * n;
    if constexpr (OUTGOING) {
      if (i0 + i < n && j0 + j < n) a[plane + int64_t(i0 + i) * n + j0 + j] = values[0][c][i][j];
      if (j0 + i < n && i0 + j < n) b[plane + int64_t(j0 + i) * n + i0 + j] = values[1][c][j][i];
    } else {
      if (j0 + i < n && i0 + j < n) a[plane + int64_t(j0 + i) * n + i0 + j] = values[0][c][j][i];
      if (i0 + i < n && j0 + j < n) b[plane + int64_t(i0 + i) * n + j0 + j] = values[1][c][i][j];
    }
  }
}

template <int STRIDE>
__global__ void gate_split_kernel(const bf16 *p, const bf16 *g, const float *mask, bf16 *a, bf16 *b,
                                  int64_t count) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  int64_t pair = index / channels_per_operand;
  int64_t c = index % channels_per_operand;
  int64_t source = pair * STRIDE + c;
  float m = mask[pair];
  a[index] = bf16(float(gated(p[source], g[source])) * m);
  b[index] = bf16(float(gated(p[source + channels_per_operand], g[source + channels_per_operand])) * m);
}

void check_gate(const at::Tensor &p, const at::Tensor &g) {
  TORCH_CHECK(p.is_cuda() && p.device() == g.device() && p.sizes() == g.sizes());
  TORCH_CHECK(p.scalar_type() == at::kBFloat16 && g.scalar_type() == at::kBFloat16);
  TORCH_CHECK(!at::GradMode::is_enabled(), "Inference only");
}

at::Tensor multiplication_gate(const at::Tensor &p, const at::Tensor &g) {
  check_gate(p, g);
  TORCH_CHECK(p.is_contiguous() && g.is_contiguous());
  constexpr int threads = 256;
  c10::cuda::CUDAGuard guard(p.device());
  auto out = at::empty_like(p);
  int64_t count = p.numel();
  if (!count) return out;
  auto stream = c10::cuda::getCurrentCUDAStream(p.get_device()).stream();
  gate_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
      p.data_ptr<bf16>(), g.data_ptr<bf16>(), out.data_ptr<bf16>(), count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<at::Tensor> multiplication_gate_split(const at::Tensor &p, const at::Tensor &g,
                                                  const at::Tensor &mask) {
  check_gate(p, g);
  TORCH_CHECK(p.dim() == 4 && p.size(0) == 1 && p.size(1) == p.size(2) && p.size(3) == projection_channels);
  TORCH_CHECK((p.stride(2) == projection_channels || p.stride(2) == joint_projection_stride) &&
                  p.stride(3) == 1 &&
                  p.stride(1) == p.size(2) * p.stride(2) && g.strides() == p.strides(),
              "Expected dense or joint-projection rows");
  TORCH_CHECK(mask.device() == p.device() && mask.scalar_type() == at::kFloat &&
              mask.is_contiguous() && mask.numel() * projection_channels == p.numel());
  constexpr int threads = 256;
  c10::cuda::CUDAGuard guard(p.device());
  int64_t n = p.size(1), count = n * n * channels_per_operand;
  auto a = at::empty({1, n, n, channels_per_operand}, p.options()), b = at::empty_like(a);
  if (!count) return {a, b};
  auto stream = c10::cuda::getCurrentCUDAStream(p.get_device()).stream();
  if (p.stride(2) == joint_projection_stride) {
    gate_split_kernel<joint_projection_stride><<<(count + threads - 1) / threads, threads, 0, stream>>>(
        p.data_ptr<bf16>(), g.data_ptr<bf16>(), mask.data_ptr<float>(), a.data_ptr<bf16>(),
        b.data_ptr<bf16>(), count);
  } else {
    gate_split_kernel<projection_channels><<<(count + threads - 1) / threads, threads, 0, stream>>>(
        p.data_ptr<bf16>(), g.data_ptr<bf16>(), mask.data_ptr<float>(), a.data_ptr<bf16>(),
        b.data_ptr<bf16>(), count);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a, b};
}

template <int TILE, int C, int STRIDE>
void launch_pack(const at::Tensor &p, const at::Tensor &g, const at::Tensor &mask, at::Tensor &a,
                 at::Tensor &b, int n, bool outgoing, int threads, cudaStream_t stream) {
  dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE, channels_per_operand / C);
  if (outgoing) {
    gate_pack_kernel<TILE, C, true, STRIDE><<<grid, threads, 0, stream>>>(
        p.data_ptr<bf16>(), g.data_ptr<bf16>(), mask.data_ptr<float>(), a.data_ptr<bf16>(),
        b.data_ptr<bf16>(), n);
  } else {
    gate_pack_kernel<TILE, C, false, STRIDE><<<grid, threads, 0, stream>>>(
        p.data_ptr<bf16>(), g.data_ptr<bf16>(), mask.data_ptr<float>(), a.data_ptr<bf16>(),
        b.data_ptr<bf16>(), n);
  }
}

std::vector<at::Tensor> multiplication_gate_pack(const at::Tensor &p, const at::Tensor &g,
                                                 const at::Tensor &mask, bool outgoing,
                                                 int channels_per_tile) {
  check_gate(p, g);
  TORCH_CHECK(p.dim() == 4 && p.size(0) == 1 && p.size(1) == p.size(2) && p.size(3) == projection_channels);
  TORCH_CHECK((p.stride(2) == projection_channels || p.stride(2) == joint_projection_stride) &&
                  p.stride(3) == 1 &&
                  p.stride(1) == p.size(2) * p.stride(2) && g.strides() == p.strides(),
              "Expected dense or joint-projection rows");
  TORCH_CHECK(mask.device() == p.device() && mask.scalar_type() == at::kFloat &&
              mask.is_contiguous() && mask.numel() * projection_channels == p.numel());
  constexpr int threads = 256;
  TORCH_CHECK(channels_per_tile == 16 || channels_per_tile == 32);
  c10::cuda::CUDAGuard guard(p.device());
  int n = p.size(1);
  auto a = at::empty({channels_per_operand, n, n}, p.options()), b = at::empty_like(a);
  if (!n) return {a, b};
  auto stream = c10::cuda::getCurrentCUDAStream(p.get_device()).stream();
#define PACK(T, C)                                                                                  \
  if (p.stride(2) == joint_projection_stride) {                                                     \
    launch_pack<T, C, joint_projection_stride>(p, g, mask, a, b, n, outgoing, threads, stream);     \
  } else {                                                                                          \
    launch_pack<T, C, projection_channels>(p, g, mask, a, b, n, outgoing, threads, stream);         \
  }
  if (channels_per_tile == 16) {
    PACK(spatial_tile, 16);
  } else {
    PACK(spatial_tile, 32);
  }
#undef PACK
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {a, b};
}
