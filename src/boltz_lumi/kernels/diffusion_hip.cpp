#include <torch/extension.h>

at::Tensor diffusion_epilogue(const at::Tensor &, const at::Tensor &,
                              const c10::optional<at::Tensor> &, int);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("epilogue", &diffusion_epilogue,
        "FP32 Boltz diffusion epilogues on the current PyTorch stream");
}
