"""Check conditioning-cache correctness with changing activations and restore original forward methods."""

from types import SimpleNamespace
from unittest.mock import patch
import torch
from boltz.model.modules.transformersv2 import DiffusionTransformerLayer
from boltz_lumi.atom_conditioning import atom_conditioning_cache
from boltz.model.modules.diffusionv2 import AtomDiffusion


def equal(actual, expected):
    delta = actual.double() - expected.double()
    norm = expected.double().norm().item()
    assert torch.equal(actual, expected), dict(
        max_absolute_error=delta.abs().max().item(),
        relative_l2_error=delta.norm().item() / norm if norm else None,
        failing_values=int((actual != expected).sum()),
        atol=0,
        rtol=0,
    )


class SamplingFixture(AtomDiffusion):
    """Use a minimal score model and deterministic coordinate scaling for sampling."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.score_model = torch.nn.Linear(1, 1).cuda()
        self.sigma_min = 0.0004
        self.sigma_max = 160.0
        self.sigma_data = 16.0
        self.rho = 7.0
        self.num_sampling_steps = 200
        self.gamma_0 = 0.8
        self.gamma_min = 1.0
        self.noise_scale = 1.003
        self.step_scale = 1.5
        self.step_scale_random = None
        self.alignment_reverse_diff = False
        self.eval()

    def preconditioned_network_forward(self, coords, sigma, network_condition_kwargs):
        return coords * 0.75


def main():
    assert torch.version.hip
    torch.manual_seed(317)
    batch_size = 2
    num_atoms = 17
    atom_dim = 32
    conditioning_dim = 32
    num_heads = 4
    layers_per_stack = 2
    num_masked_atoms = 3
    layers = [
        DiffusionTransformerLayer(num_heads, atom_dim, conditioning_dim).cuda().eval()
        for _ in range(2 * layers_per_stack)
    ]
    # Randomize weights so the conditioning projections contribute to the outputs.
    with torch.no_grad():
        for layer in layers:
            for parameter in layer.parameters():
                parameter.normal_(mean=0, std=0.07)
    score = SimpleNamespace(
        atom_attention_encoder=SimpleNamespace(
            atom_encoder=SimpleNamespace(
                diffusion_transformer=SimpleNamespace(layers=layers[:layers_per_stack])
            )
        ),
        atom_attention_decoder=SimpleNamespace(
            atom_decoder=SimpleNamespace(
                diffusion_transformer=SimpleNamespace(layers=layers[layers_per_stack:])
            )
        ),
    )
    conditioning = torch.randn(batch_size, num_atoms, conditioning_dim, device="cuda")
    bias = torch.randn(batch_size, num_atoms, num_atoms, num_heads, device="cuda")
    mask = torch.ones(batch_size, num_atoms, device="cuda")
    mask[:, -num_masked_atoms:] = 0
    activations = [torch.randn_like(conditioning) * scale for scale in (1, 2, 0.3)]
    originals = [(m, m.forward) for layer in layers for m in layer.modules()]
    with torch.inference_mode():
        reference = [
            layer(activation, conditioning, bias, mask)
            for activation in activations
            for layer in layers
        ]
        target = layers[0].adaln.s_norm
        with patch.object(target, "forward", wraps=target.forward) as original:
            with atom_conditioning_cache(score):
                actual = [
                    layer(activation, conditioning, bias, mask)
                    for activation in activations
                    for layer in layers
                ]
            assert original.call_count == 1
        for actual_output, expected_output in zip(actual, reference):
            equal(actual_output, expected_output)
        # A fresh context must recompute projections for a different input.
        changed_conditioning = conditioning + 1
        with atom_conditioning_cache(score):
            changed = [layer(activations[0], changed_conditioning, bias, mask) for layer in layers]
        for out, layer in zip(changed, layers):
            equal(out, layer(activations[0], changed_conditioning, bias, mask))
        try:
            with atom_conditioning_cache(score):
                raise RuntimeError("intentional context failure")
        except RuntimeError:
            pass
        assert all(m.forward == old for m, old in originals)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reference = layers[0](activations[0], conditioning, bias, mask)
            with atom_conditioning_cache(score):
                actual = layers[0](activations[0], conditioning, bias, mask)
            equal(actual, reference)
    # With gradient tracking enabled, bypass the cache: changing conditioning
    # must produce correct outputs, and backward() must compute input gradients.
    activation = torch.randn(
        batch_size, num_atoms, atom_dim, device="cuda", requires_grad=True
    )
    with atom_conditioning_cache(score):
        first = layers[0](activation, conditioning, bias, mask)
        second = layers[0](activation, conditioning + 2, bias, mask)
    equal(first, layers[0](activation, conditioning, bias, mask))
    equal(second, layers[0](activation, conditioning + 2, bias, mask))
    second.sum().backward()
    assert activation.grad is not None
    assert torch.isfinite(activation.grad).all()
    assert activation.grad.abs().max() > 0
    from boltz_lumi.diffusion import optimized_sample

    fixture = SamplingFixture()
    sampling_batch_size = 1
    mask = torch.ones(sampling_batch_size, num_atoms, device="cuda")
    steering = {
        k: False for k in ["fk_steering", "physical_guidance_update", "contact_guidance_update"]
    }
    outcomes = []
    with torch.inference_mode():
        for fn in [AtomDiffusion.sample, optimized_sample]:
            torch.manual_seed(99)
            # Two samples with one in parallel exercise multiple sampling batches.
            result = fn(
                fixture,
                atom_mask=mask,
                num_sampling_steps=20,
                multiplicity=2,
                max_parallel_samples=1,
                steering_args=steering,
                diffusion_conditioning={
                    "c": torch.zeros(
                        sampling_batch_size, num_atoms, conditioning_dim, device="cuda"
                    ),
                    "token_trans_bias": torch.zeros(
                        sampling_batch_size, num_atoms, num_atoms, num_heads, device="cuda"
                    ),
                },
            )
            outcomes.append(
                (result["sample_atom_coords"], torch.cuda.get_rng_state(), torch.get_rng_state())
            )
    for original_output, optimized_output in zip(*outcomes):
        equal(original_output, optimized_output)
    print("Atom conditioning passed: absolute error=0, relative L2 error=0; atol=rtol=0, failures=0")


if __name__ == "__main__":
    main()
