"""Install gfx90a inference operators on Boltz-2 models."""

from types import MethodType
import torch


def install(model):
    from boltz.model.models.boltz2 import Boltz2

    if (
        not isinstance(model, Boltz2)
        or getattr(model, "_lumi_optimized", False)
        or not torch.version.hip
        or not torch.cuda.is_available()
        or not torch.cuda.get_device_properties(torch.cuda.current_device()).gcnArchName.startswith("gfx90a")
    ):
        return
    from .triangle import contiguous_mask_chunk, ordered_triangle_forward, normalization_forward
    from .multiplication import normalized_forward
    from .diffusion import adaln_forward, transition_forward, optimized_sample
    from .pair_transition import transition_forward as pair_forward
    from .pair_bias import prepared_forward

    for name, module in model.named_modules():
        cls = type(module).__name__
        if cls == "TriangleAttentionEndingNode":
            module._chunk = MethodType(contiguous_mask_chunk, module)
        if cls in {"TriangleAttention", "TriangleAttentionEndingNode"}:
            module.mha.forward = MethodType(ordered_triangle_forward, module.mha)
            module.layer_norm.forward = MethodType(normalization_forward, module.layer_norm)
        if cls in {"TriangleMultiplicationIncoming", "TriangleMultiplicationOutgoing"}:
            module.forward = MethodType(normalized_forward, module)
        if name.startswith("structure_module.score_model.") and cls in {
            "AdaLN",
            "ConditionedTransitionBlock",
        }:
            forward = adaln_forward if cls == "AdaLN" else transition_forward
            module.forward = MethodType(forward, module)
        if (
            name.startswith("structure_module.score_model.token_transformer.")
            and cls == "AttentionPairBias"
        ):
            module.forward = MethodType(prepared_forward, module)
        if name.endswith(".transition_z") and cls == "Transition":
            module.forward = MethodType(pair_forward, module)
    model.structure_module.sample = MethodType(optimized_sample, model.structure_module)
    model._lumi_optimized = True
