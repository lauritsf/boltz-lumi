#include <torch/extension.h>
at::Tensor pair_silu_mul(const at::Tensor &a, const at::Tensor &b);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("silu_mul", &pair_silu_mul);
}
