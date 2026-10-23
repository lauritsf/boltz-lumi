// Based on PyTorch's cunn_SoftMaxForwardFast in SoftMax.cu,
// commit 70d99e998b4955e0049d13a98d77ae1b14db1f45.
// License and copyright: third_party/pytorch-LICENSE.
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <ATen/native/cuda/block_reduce.cuh>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

constexpr int sequence_length = 2511;
constexpr int threads_per_block = 512;
constexpr int wave_size = 64;
constexpr int alignment_bytes = 16;
constexpr int bf16_bits = 16;
constexpr int elements_per_word = sizeof(uint32_t) / sizeof(c10::BFloat16);

struct BiasInput {
  const c10::BFloat16 *scores;
  const c10::BFloat16 *bias;
  const float *mask;
  int64_t n, bias_k, mask_k;
  __device__ __forceinline__ float at(int k) const {
    // PyTorch rounds to BF16 after each bias addition.
    c10::BFloat16 first(float(scores[k]) + mask[int64_t(k) * mask_k]);
    c10::BFloat16 second(float(first) + float(bias[int64_t(k) * bias_k]));
    return float(second);
  }
};

struct SumOp {
  __device__ float combine(float a, float b) const { return a + b; }
  __device__ float warp_shfl_down(float value, int offset) const {
    return __shfl_down(value, offset, wave_size);
  }
};
struct MaxOp {
  __device__ float combine(float a, float b) const { return fmaxf(a, b); }
  __device__ float warp_shfl_down(float value, int offset) const {
    return __shfl_down(value, offset, wave_size);
  }
};

// Packed BF16 scores avoid reloading bias and mask for the sum/output passes.
template <int PACKED>
__device__ __forceinline__ float cached_at(const uint32_t (&values)[PACKED], int slot) {
  c10::BFloat16 value;
  value.x = uint16_t(values[slot / elements_per_word] >> (bf16_bits * (slot % elements_per_word)));
  return float(value);
}

__global__ __launch_bounds__(threads_per_block) void triangle_cached_softmax_kernel(
    const c10::BFloat16 *scores, const c10::BFloat16 *bias, const float *mask, c10::BFloat16 *out,
    int outer, int heads, int64_t bb, int64_t bh, int64_t bq, int64_t bk, int64_t mb, int64_t mi,
    int64_t mk) {
  // Slot 0 is the alignment prefix; remaining slots are one block width apart.
  constexpr int SLOTS = (sequence_length + threads_per_block - 1) / threads_per_block + 1;
  constexpr int PACKED = (SLOTS + elements_per_word - 1) / elements_per_word;
  int64_t row = blockIdx.x;
  int q = row % sequence_length;
  int h = (row / sequence_length) % heads;
  int i = (row / (int64_t(sequence_length) * heads)) % outer;
  int64_t batch = row / (int64_t(sequence_length) * heads * outer);
  BiasInput input{scores + row * sequence_length,
                  bias + batch * bb + int64_t(h) * bh + int64_t(q) * bq,
                  mask + batch * mb + int64_t(i) * mi,
                  sequence_length,
                  bk,
                  mk};
  // Row alignment determines which terms each PyTorch thread sums.
  int shift = (reinterpret_cast<uintptr_t>(input.scores) % alignment_bytes) / sizeof(c10::BFloat16);
  int start = shift > 0 ? threads_per_block - shift : 0;
  int tid = threadIdx.x;
  uint32_t values[PACKED];
  float partial_max = std::numeric_limits<float>::lowest();
#pragma unroll
  for (int pair = 0; pair < PACKED; ++pair) {
    uint32_t packed = 0;
#pragma unroll
    for (int half = 0; half < elements_per_word; ++half) {
      int slot = pair * elements_per_word + half;
      int k = slot == 0 ? tid - shift : start + tid + (slot - 1) * threads_per_block;
      bool valid = slot < SLOTS && (slot == 0 ? shift > 0 && tid >= shift : k < sequence_length);
      c10::BFloat16 value(valid ? input.at(k) : 0.f);
      packed |= uint32_t(value.x) << (half * bf16_bits);
      if (valid) partial_max = fmaxf(partial_max, float(value));
    }
    values[pair] = packed;
  }
  __shared__ float scratch[threads_per_block / wave_size];
  float reduced = at::native::cuda_utils::BlockReduce<float>(
      partial_max, MaxOp(), std::numeric_limits<float>::lowest(), scratch);
  if (tid == 0) scratch[0] = reduced;
  __syncthreads();
  float maximum = scratch[0];
  float partial_sum = 0.f;
#pragma unroll
  for (int slot = 0; slot < SLOTS; ++slot) {
    int k = slot == 0 ? tid - shift : start + tid + (slot - 1) * threads_per_block;
    bool valid = slot == 0 ? shift > 0 && tid >= shift : k < sequence_length;
    if (valid) partial_sum = partial_sum + __expf(cached_at(values, slot) - maximum);
  }
  reduced = at::native::cuda_utils::BlockReduce<float>(partial_sum, SumOp(), 0.f, scratch);
  if (tid == 0) scratch[0] = 1.f / reduced;
  __syncthreads();
  float reciprocal_sum = scratch[0];
#pragma unroll
  for (int slot = 0; slot < SLOTS; ++slot) {
    int k = slot == 0 ? tid - shift : start + tid + (slot - 1) * threads_per_block;
    bool valid = slot == 0 ? shift > 0 && tid >= shift : k < sequence_length;
    if (valid)
      out[row * sequence_length + k] =
          c10::BFloat16(__expf(cached_at(values, slot) - maximum) * reciprocal_sum);
  }
}

at::Tensor triangle_block_softmax(const at::Tensor &scores, const at::Tensor &bias,
                                  const at::Tensor &mask) {
  TORCH_CHECK(!at::GradMode::is_enabled(), "Inference only");
  TORCH_CHECK(scores.is_cuda() && scores.device() == bias.device() &&
              scores.device() == mask.device());
  TORCH_CHECK(scores.scalar_type() == at::kBFloat16 && bias.scalar_type() == at::kBFloat16 &&
              mask.scalar_type() == at::kFloat);
  TORCH_CHECK(scores.is_contiguous() && scores.dim() == 5 && scores.size(3) == sequence_length &&
              scores.size(4) == sequence_length);
  int outer = scores.size(1), heads = scores.size(2);
  int64_t batch = scores.size(0), rows = batch * outer * heads * sequence_length;
  TORCH_CHECK(batch > 0 && outer > 0 && heads > 0 && rows <= std::numeric_limits<int32_t>::max());
  TORCH_CHECK(bias.sizes() == at::IntArrayRef({batch, 1, heads, sequence_length, sequence_length}));
  TORCH_CHECK(mask.sizes() == at::IntArrayRef({batch, outer, 1, 1, sequence_length}));
  c10::cuda::CUDAGuard guard(scores.device());
  auto output = at::empty_like(scores);
  auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device()).stream();
  triangle_cached_softmax_kernel<<<rows, threads_per_block, 0, stream>>>(
      scores.data_ptr<c10::BFloat16>(), bias.data_ptr<c10::BFloat16>(), mask.data_ptr<float>(),
      output.data_ptr<c10::BFloat16>(), outer, heads, bias.stride(0), bias.stride(2),
      bias.stride(3), bias.stride(4), mask.stride(0), mask.stride(1), mask.stride(4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
