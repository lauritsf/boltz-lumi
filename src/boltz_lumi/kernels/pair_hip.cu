#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

constexpr int vector_elements = sizeof(uint64_t) / sizeof(c10::BFloat16);
constexpr int bf16_bits = 16;

__device__ __forceinline__ c10::BFloat16 activated_product(c10::BFloat16 a, c10::BFloat16 b) {
  float x = float(a);
  // Round SiLU to BF16 before multiplying, as PyTorch does.
  c10::BFloat16 activation(x / (1.f + expf(-x)));
  return c10::BFloat16(float(activation) * float(b));
}

template <int VEC>
__global__ void pair_silu_kernel(const c10::BFloat16 *a, const c10::BFloat16 *b, c10::BFloat16 *out,
                                 int64_t n) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if constexpr (VEC == vector_elements) {
    if (index >= n / vector_elements) return;
    uint64_t av, bv, result = 0;
    __builtin_memcpy(&av, a + index * VEC, sizeof(uint64_t));
    __builtin_memcpy(&bv, b + index * VEC, sizeof(uint64_t));
#pragma unroll
    for (int j = 0; j < VEC; ++j) {
      c10::BFloat16 aa, bb;
      aa.x = uint16_t(av >> (j * bf16_bits));
      bb.x = uint16_t(bv >> (j * bf16_bits));
      auto value = activated_product(aa, bb);
      result |= uint64_t(value.x) << (j * bf16_bits);
    }
    __builtin_memcpy(out + index * VEC, &result, sizeof(uint64_t));
  } else {
    if (index < n) out[index] = activated_product(a[index], b[index]);
  }
}

at::Tensor pair_silu_mul(const at::Tensor &a, const at::Tensor &b) {
  TORCH_CHECK(a.is_cuda() && a.device() == b.device() && a.sizes() == b.sizes());
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
              a.is_contiguous() && b.is_contiguous());
  TORCH_CHECK(!a.requires_grad() && !b.requires_grad());
  constexpr int threads = 128;
  c10::cuda::CUDAGuard guard(a.device());
  auto out = at::empty_like(a);
  int64_t n = a.numel();
  if (n == 0) return out;
  auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  bool aligned = reinterpret_cast<uintptr_t>(a.data_ptr()) % sizeof(uint64_t) == 0 &&
                 reinterpret_cast<uintptr_t>(b.data_ptr()) % sizeof(uint64_t) == 0;
  if (aligned && n % vector_elements == 0) {
    pair_silu_kernel<vector_elements><<<(n / vector_elements + threads - 1) / threads, threads, 0, stream>>>(
        a.data_ptr<c10::BFloat16>(), b.data_ptr<c10::BFloat16>(), out.data_ptr<c10::BFloat16>(), n);
  } else {
    pair_silu_kernel<1><<<(n + threads - 1) / threads, threads, 0, stream>>>(
        a.data_ptr<c10::BFloat16>(), b.data_ptr<c10::BFloat16>(), out.data_ptr<c10::BFloat16>(), n);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
