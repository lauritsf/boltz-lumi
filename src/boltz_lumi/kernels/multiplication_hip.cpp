#include <torch/extension.h>
at::Tensor multiplication_norm(const at::Tensor &, const at::Tensor &, const at::Tensor &, double,
                               bool, int, int, bool);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm", &multiplication_norm);
}
