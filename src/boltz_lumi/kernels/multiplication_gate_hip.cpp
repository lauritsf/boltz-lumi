#include <torch/extension.h>
at::Tensor multiplication_gate(const at::Tensor &, const at::Tensor &);
std::vector<at::Tensor> multiplication_gate_pack(const at::Tensor &, const at::Tensor &,
                                                 const at::Tensor &, bool, int);
std::vector<at::Tensor> multiplication_gate_split(const at::Tensor &, const at::Tensor &,
                                                  const at::Tensor &);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gate", &multiplication_gate);
  m.def("pack", &multiplication_gate_pack);
  m.def("split", &multiplication_gate_split);
}
