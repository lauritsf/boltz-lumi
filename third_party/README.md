# Source notices

- `kernels/layout_copy.py` adapts AMD AITER's transpose:
  AITER `9c971f5742f218a4d07a5ab12d5689c85b7279fc`, Triton 3.6.0. MIT notices
  are in `aiter-LICENSE` and `triton-LICENSE`.
- Ordered softmax and normalization kernels preserve the algorithms in PyTorch
  `70d99e998b4955e0049d13a98d77ae1b14db1f45`. BSD notices: `pytorch-LICENSE`.
- Adapters preserve Boltz 2.2.1 arithmetic, commit
  `cb04aeccdd480fd4db707f0bbafde538397fa2ac`. MIT notice: `boltz-LICENSE`.
- The protein–ligand input uses data from Schrödinger's public benchmark.
  License: `schrodinger-benchmark-LICENSE`.
