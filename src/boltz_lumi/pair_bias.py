"""Dense storage for fixed diffusion attention biases."""

import torch


def prepare_token_bias(value):
    """Pack immutable per-layer token biases once, preserving every value.

    Return the same logical [B,Q,K,layer*head] shape with dense head/Q/K
    storage. Splitting the final dimension into layer/head remains a view.
    """
    return value.permute(0, 3, 1, 2).contiguous().float().permute(0, 2, 3, 1)


def prepared_forward(module, s, z, mask, k_in, multiplicity=1):
    """Pinned math, with the repeat-one copy omitted for an already dense bias."""
    from boltz.model.layers.attentionv2 import AttentionPairBias

    if torch.is_grad_enabled() or module.training or module.compute_pair_bias:
        return AttentionPairBias.forward(module, s, z, mask, k_in, multiplicity)
    batch = s.shape[0]
    q = module.proj_q(s).view(batch, -1, module.num_heads, module.head_dim)
    k = module.proj_k(k_in).view(batch, -1, module.num_heads, module.head_dim)
    v = module.proj_v(k_in).view(batch, -1, module.num_heads, module.head_dim)
    bias = module.proj_z(z)
    if multiplicity != 1 or not bias.is_contiguous():
        bias = bias.repeat_interleave(multiplicity, 0)
    gate = module.proj_g(s).sigmoid()
    with torch.autocast("cuda", enabled=False):
        attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn = attn / (module.head_dim**0.5) + bias.float()
        attn = attn + (1 - mask[:, None, None].float()) * -module.inf
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    out = out.reshape(batch, -1, module.c_s)
    return module.proj_o(gate * out)
