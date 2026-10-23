"""Run the Boltz CLI with MI250X inference kernels."""

from importlib.metadata import version
import sys
from unittest.mock import patch

import torch
from pytorch_lightning import Callback, Trainer

from . import SUPPORTED_BOLTZ_VERSION
from .adapters import install


class InferenceAdapter(Callback):
    def __init__(self, optimize=True):
        self.optimize = optimize

    def on_predict_start(self, trainer, model):
        # Boltz's affinity loader does not forward --no_kernels.
        if torch.version.hip and hasattr(model, "use_kernels"):
            model.use_kernels = False
        if self.optimize and model.device.type == "cuda":
            install(model)

    def on_predict_batch_end(self, trainer, model, outputs, batch, batch_idx, dataloader_idx=0):
        if outputs.get("exception"):
            raise RuntimeError("Boltz prediction failed; inspect the preceding log")
        for key in ("coords", "confidence_score", "affinity_pred_value", "affinity_probability_binary"):
            if key in outputs and not torch.isfinite(outputs[key]).all():
                raise RuntimeError(f"Nonfinite prediction: {key}")


def run(arguments=None, optimize=True):
    from boltz.main import cli

    if version("boltz") != SUPPORTED_BOLTZ_VERSION:
        raise RuntimeError(f"This adapter supports Boltz {SUPPORTED_BOLTZ_VERSION}")
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if torch.version.hip and arguments[:1] == ["predict"] and "--no_kernels" not in arguments:
        arguments.append("--no_kernels")
    original_predict = Trainer.predict

    def predict(trainer, model, *args, **kwargs):
        adapter = InferenceAdapter(optimize)
        trainer.callbacks.append(adapter)
        try:
            return original_predict(trainer, model, *args, **kwargs)
        finally:
            trainer.callbacks.remove(adapter)

    with patch.object(Trainer, "predict", predict):
        return cli(arguments, standalone_mode=False)


if __name__ == "__main__":
    run()
