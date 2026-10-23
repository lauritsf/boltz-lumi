#include <torch/extension.h>
at::Tensor triangle_block_softmax(const at::Tensor &, const at::Tensor &, const at::Tensor &);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &triangle_block_softmax);
}
