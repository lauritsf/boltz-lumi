#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <cuda_runtime.h>

// Values match the operation identifiers passed by the Python adapter.
enum Operation { Modulation = 0, Gate = 1, GatedSwiGLU = 3 };

constexpr int vector_elements = sizeof(float4) / sizeof(float);

// The multiply and add round separately in PyTorch.
template <int KIND> __device__ __forceinline__ float value(float a, float b, float c) {
  if constexpr (KIND == Modulation) {
    float g = 1.f / (1.f + expf(-b));
    float product = g * a;
    return product + c;
  } else if constexpr (KIND == Gate) {
    float g = 1.f / (1.f + expf(-a));
    return g * b;
  } else {
    float activation = b / (1.f + expf(-b));
    float product = activation * a;
    return product * c;
  }
}

template <int KIND, int VEC>
__global__ void epilogue_kernel(const float *a, const float *b, const float *c, float *out,
                                int64_t n, int64_t cols) {
  int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if constexpr (VEC == vector_elements) {
    if (index >= n / vector_elements) return;
    int64_t input_index = index;
    if constexpr (KIND == GatedSwiGLU)
      input_index = (index / (cols / vector_elements)) * (2 * cols / vector_elements) +
                    index % (cols / vector_elements);
    float4 av = reinterpret_cast<const float4 *>(a)[input_index];
    float4 bv;
    if constexpr (KIND == GatedSwiGLU)
      bv = reinterpret_cast<const float4 *>(a)[input_index + cols / vector_elements];
    else
      bv = reinterpret_cast<const float4 *>(b)[index];
    float4 cv = make_float4(0, 0, 0, 0);
    if constexpr (KIND == Modulation) cv = reinterpret_cast<const float4 *>(c)[index];
    if constexpr (KIND == GatedSwiGLU) cv = reinterpret_cast<const float4 *>(b)[index];
    float4 result = make_float4(value<KIND>(av.x, bv.x, cv.x), value<KIND>(av.y, bv.y, cv.y),
                                value<KIND>(av.z, bv.z, cv.z), value<KIND>(av.w, bv.w, cv.w));
    reinterpret_cast<float4 *>(out)[index] = result;
  } else {
    if (index >= n) return;
    int64_t input_index = index;
    if constexpr (KIND == GatedSwiGLU) input_index = (index / cols) * (2 * cols) + index % cols;
    float av = a[input_index];
    float bv;
    if constexpr (KIND == GatedSwiGLU)
      bv = a[input_index + cols];
    else
      bv = b[index];
    float cv = 0;
    if constexpr (KIND == Modulation) cv = c[index];
    if constexpr (KIND == GatedSwiGLU) cv = b[index];
    out[index] = value<KIND>(av, bv, cv);
  }
}

template <int KIND>
void run(const float *a, const float *b, const float *c, float *out, int64_t n, int64_t cols,
         int threads, int vec, cudaStream_t stream) {
  if (vec == vector_elements) {
    epilogue_kernel<KIND, vector_elements>
        <<<(n / vector_elements + threads - 1) / threads, threads, 0, stream>>>(a, b, c, out, n, cols);
  } else {
    epilogue_kernel<KIND, 1>
        <<<(n + threads - 1) / threads, threads, 0, stream>>>(a, b, c, out, n, cols);
  }
}

at::Tensor diffusion_epilogue(const at::Tensor &a, const at::Tensor &b,
                              const c10::optional<at::Tensor> &c, int kind) {
  TORCH_CHECK(kind == Modulation || kind == Gate || kind == GatedSwiGLU);
  constexpr int threads = 128;
  TORCH_CHECK(a.is_cuda() && b.device() == a.device() && a.scalar_type() == at::kFloat &&
              b.scalar_type() == at::kFloat);
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous());
  c10::cuda::CUDAGuard guard(a.device());
  auto shape = a.sizes().vec();
  int64_t cols = shape.back();
  // SwiGLU receives two concatenated projections.
  if (kind == GatedSwiGLU) {
    TORCH_CHECK(cols % 2 == 0);
    shape.back() /= 2;
    cols /= 2;
  }
  auto out = at::empty(shape, a.options());
  if (kind == Modulation || kind == Gate || kind == GatedSwiGLU) TORCH_CHECK(b.sizes() == out.sizes());
  if (kind == Modulation)
    TORCH_CHECK(c.has_value() && c->sizes() == out.sizes() && c->device() == a.device() &&
                c->scalar_type() == at::kFloat && c->is_contiguous());
  int64_t n = out.numel();
  if (n == 0) return out;
  bool aligned = reinterpret_cast<uintptr_t>(a.data_ptr()) % alignof(float4) == 0 &&
                 reinterpret_cast<uintptr_t>(b.data_ptr()) % alignof(float4) == 0 &&
                 (!c || reinterpret_cast<uintptr_t>(c->data_ptr()) % alignof(float4) == 0);
  // A row boundary must not split a float4 load.
  int vec = n % vector_elements == 0 && cols % vector_elements == 0 && aligned ? vector_elements : 1;
  auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const float *cp = c ? c->data_ptr<float>() : nullptr;
  switch (kind) {
  case Modulation:
    run<Modulation>(a.data_ptr<float>(), b.data_ptr<float>(), cp, out.data_ptr<float>(), n, cols, threads,
           vec, stream);
    break;
  case Gate:
    run<Gate>(a.data_ptr<float>(), b.data_ptr<float>(), cp, out.data_ptr<float>(), n, cols, threads,
           vec, stream);
    break;
  case GatedSwiGLU:
    run<GatedSwiGLU>(a.data_ptr<float>(), b.data_ptr<float>(), cp, out.data_ptr<float>(), n, cols, threads,
           vec, stream);
    break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
