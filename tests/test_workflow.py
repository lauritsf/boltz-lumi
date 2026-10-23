"""Check native CLI forwarding, prediction failures and benchmark outputs."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from pytorch_lightning import Trainer
from boltz_lumi.__main__ import InferenceAdapter, run
from boltz_lumi.benchmark import CONFIDENCE_FIELDS, validate_outputs


class WorkflowTests(unittest.TestCase):
    def test_every_candidate_and_sample_is_required(self):
        num_samples = 2
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for candidate in ("first", "second"):
                (root / candidate).mkdir()
                for rank in range(num_samples):
                    stem = f"{candidate}_model_{rank}"
                    (root / candidate / (stem + ".cif")).write_text("structure")
                    (root / candidate / ("confidence_" + stem + ".json")).write_text(
                        json.dumps(dict.fromkeys(CONFIDENCE_FIELDS, 0.5))
                    )
            validate_outputs(root, ["first", "second"], num_samples)
            confidence = root / "second/confidence_second_model_1.json"
            confidence.unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing output for second_model_1"):
                validate_outputs(root, ["first", "second"], num_samples)
            confidence.write_text(json.dumps(dict.fromkeys(CONFIDENCE_FIELDS, float("nan"))))
            with self.assertRaisesRegex(RuntimeError, "Invalid confidence"):
                validate_outputs(root, ["first", "second"], num_samples)
            with self.assertRaisesRegex(RuntimeError, "No inputs"):
                validate_outputs(root, [], 1)

    def test_pdb_and_affinity_outputs(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "ligand"
            directory.mkdir()
            (directory / "ligand_model_0.pdb").write_text("structure")
            (directory / "confidence_ligand_model_0.json").write_text(
                json.dumps(dict.fromkeys(CONFIDENCE_FIELDS, 0.5))
            )
            validate_outputs(root, ["ligand"], 1, output_format="pdb")
            affinity = directory / "affinity_ligand.json"
            affinity.write_text(
                json.dumps(
                    dict(
                        affinity_pred_value=-2.0,
                        affinity_probability_binary=0.8,
                    )
                )
            )
            # Affinity validation must not require five structure samples.
            validate_outputs(root, ["ligand"], 5, affinity=True)
            affinity.write_text('{"affinity_pred_value": -2}')
            with self.assertRaisesRegex(RuntimeError, "Invalid confidence"):
                validate_outputs(root, ["ligand"], 5, affinity=True)

    def test_native_options_and_existing_output_directory(self):
        with TemporaryDirectory() as directory:
            arguments = [
                "predict", "input.yaml",
                "--out_dir", directory,
                "--model", "boltz1",
                "--devices", "2",
                "--output_format", "pdb",
                "--checkpoint", "custom.ckpt",
                "--cache", directory,
                "--override",
            ]
            with patch("boltz.main.cli") as cli:
                run(arguments)
            forwarded = cli.call_args.args[0]
            expected_extra_arguments = ["--no_kernels"] if torch.version.hip else []
            self.assertEqual(forwarded[:len(arguments)], arguments)
            self.assertEqual(forwarded[len(arguments):], expected_extra_arguments)

    def test_worker_installation_and_cpu_fallback(self):
        adapter = InferenceAdapter()
        model = SimpleNamespace(device=torch.device("cpu"), use_kernels=True)
        with patch("boltz_lumi.__main__.install") as install:
            adapter.on_predict_start(None, model)
            install.assert_not_called()
            model.device = torch.device("cuda", 1)
            adapter.on_predict_start(None, model)
            install.assert_called_once_with(model)
        if torch.version.hip:
            self.assertFalse(model.use_kernels)

    def test_affinity_and_swallowed_prediction_failures(self):
        adapter = InferenceAdapter()
        output = dict(
            affinity_pred_value=torch.tensor(-2.0),
            affinity_probability_binary=torch.tensor(0.8),
        )
        adapter.on_predict_batch_end(None, None, output, None, 0)
        for output in (
            {"exception": True},
            {"affinity_pred_value": torch.tensor(float("nan"))},
        ):
            with self.assertRaises(RuntimeError):
                adapter.on_predict_batch_end(None, None, output, None, 0)

    def test_trainer_patch_and_callback_restored_after_failure(self):
        trainer = SimpleNamespace(callbacks=[])

        def fail(trainer, model, *args, **kwargs):
            self.assertIsInstance(trainer.callbacks[-1], InferenceAdapter)
            raise RuntimeError("intentional failure")

        with patch.object(Trainer, "predict", fail):
            with patch("boltz.main.cli", side_effect=lambda *a, **kw: Trainer.predict(trainer, None)):
                with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                    run(["predict", "input.yaml"])
            self.assertIs(Trainer.predict, fail)
        self.assertEqual(trainer.callbacks, [])


if __name__ == "__main__":
    unittest.main()
