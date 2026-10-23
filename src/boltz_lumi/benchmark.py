"""Compare native and optimized Boltz inference, with optional profiling."""

from contextlib import contextmanager, ExitStack
import argparse
from importlib.metadata import version
import json
import math
import os
import statistics
import time
from unittest.mock import patch
import torch
import triton
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
from . import SUPPORTED_BOLTZ_VERSION
from .adapters import install


@contextmanager
def stage_timers(model, events):
    def wrap(original, name):
        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                with torch.profiler.record_function(name):
                    return original(*args, **kwargs)
            finally:
                end.record()
                events.append((name, start, end))

        return timed

    names = {
        "input_embedder",
        "template_module",
        "msa_module",
        "pairformer_module",
        "diffusion_conditioning",
        "confidence_module",
    }
    with ExitStack() as stack:
        for name, module in model.named_modules():
            if name in names:
                stack.enter_context(patch.object(module, "forward", wrap(module.forward, name)))
        sampler = model.structure_module
        stack.enter_context(
            patch.object(sampler, "sample", wrap(sampler.sample, "diffusion_sample"))
        )
        yield


def benchmark(args, settings, trainer, model, writer, predict, positional, keywords):
    report = dict(
        settings,
        torch=torch.__version__,
        triton=triton.__version__,
        hip=torch.version.hip,
        gpu=torch.cuda.get_device_name(0),
        slurm_job=os.environ.get("SLURM_JOB_ID"),
        model_calls=[],
        rounds=[],
        errors=[],
        status="running",
    )
    normal_output = writer.output_dir
    references = {}
    events = []
    current = {}
    step = model.predict_step
    started = time.perf_counter()
    # Reuse the same preprocessed features, including any randomly chosen ones.
    datamodule = keywords["datamodule"]
    batches = list(datamodule.predict_dataloader())
    candidate_ids = [record.id for batch in batches for record in batch["record"]]
    if candidate_ids != [r.id for r in datamodule.manifest.records]:
        raise RuntimeError("Prediction loader changed candidate identity or order")
    report["feature_load_s"] = time.perf_counter() - started

    def save():
        (args.output / "timings.json").write_text(json.dumps(report, indent=2) + "\n")

    def measured_step(batch, *a, **kw):
        seed_everything(args.seed, workers=True, verbose=False)
        if len(batch["record"]) != 1:
            raise RuntimeError("Only one input per model call is supported")
        candidate = batch["record"][0].id
        events.clear()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        output = step(batch, *a, **kw)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        if output.get("exception"):
            raise RuntimeError("Boltz prediction failed; inspect the preceding log")
        stages = {}
        for name, start, end in events:
            stages[name] = stages.get(name, 0) + start.elapsed_time(end) / 1000
        report["model_calls"].append(
            dict(
                current,
                candidate=candidate,
                model_s=elapsed,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                stages_s=stages,
            )
        )
        values = {
            key: output[key].detach().cpu()
            for key in (
                "coords",
                "plddt",
                "confidence_score",
                "ptm",
                "iptm",
                "complex_plddt",
                "ligand_iptm",
                "protein_iptm",
                "affinity_pred_value",
                "affinity_probability_binary",
                "affinity_pred_value1",
                "affinity_pred_value2",
                "affinity_probability_binary1",
                "affinity_probability_binary2",
            )
            if key in output
        }
        if any(not torch.isfinite(value).all() for value in values.values()):
            raise RuntimeError("Nonfinite prediction")
        expected = references.setdefault(candidate, values)
        for key, value in values.items():
            delta = value.double() - expected[key].double()
            denominator = expected[key].double().norm().item()
            report["errors"].append(
                dict(
                    current,
                    candidate=candidate,
                    field=key,
                    max_absolute_error=delta.abs().max().item(),
                    relative_l2_error=delta.norm().item() / denominator
                    if denominator
                    else None,
                    atol=0,
                    rtol=0,
                    failing_values=int((value != expected[key]).sum()),
                )
            )
        return output

    recipes = ["upstream", "optimized"]

    def run_round(recipe, iteration, profiled=False):
        current.update(recipe=recipe, iteration=iteration, first=iteration == 0, profiled=profiled)
        if recipe == "optimized" and iteration == args.repeats and not profiled:
            # Leave the final structures where Boltz's affinity stage expects them.
            writer.output_dir = normal_output
        else:
            writer.output_dir = args.output / recipe / ("profile" if profiled else f"round-{iteration:02}")
        writer.output_dir.mkdir(parents=True, exist_ok=True)
        seed_everything(args.seed, workers=True, verbose=False)
        before = len(report["model_calls"])
        torch.cuda.synchronize()
        begin = time.perf_counter()
        result = predict(trainer, model, *positional, **keywords)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        calls = report["model_calls"][before:]
        if [c["candidate"] for c in calls] != candidate_ids:
            raise RuntimeError("Missing or reordered prediction calls")
        validate_outputs(writer.output_dir, candidate_ids, args.diffusion_samples,
                         output_format=getattr(writer, "output_format", "mmcif"),
                         affinity=settings["phase"] == "affinity")
        report["rounds"].append(dict(current, candidates=candidate_ids, round_s=elapsed))
        save()
        return result

    try:
        with (
            patch.object(
                datamodule,
                "predict_dataloader",
                lambda: DataLoader(batches, batch_size=None, num_workers=0),
            ),
            patch.object(model, "predict_step", measured_step),
        ):
            for recipe in recipes:
                if recipe == "optimized":
                    install(model)
                with stage_timers(model, events):
                    for index in range(args.repeats + 1):
                        result = run_round(recipe, index)
            if args.profile:
                with (
                    stage_timers(model, events),
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler,
                ):
                    run_round(recipes[-1], args.repeats + 1, profiled=True)
                profiler.export_chrome_trace(str(args.output / "trace.json"))
                operators = [e for e in profiler.key_averages() if e.key.startswith("aten::")]
                operators.sort(key=lambda e: e.self_device_time_total, reverse=True)
                lines = ["Operator                         GPU self seconds    Calls"]
                lines += [
                    f"{e.key:32} {e.self_device_time_total / 1e6:15.6f} {e.count:8}"
                    for e in operators[:20]
                ]
                (args.output / "profile.txt").write_text("\n".join(lines) + "\n")
        for recipe in recipes:
            warm = [
                c["model_s"]
                for c in report["model_calls"]
                if c["recipe"] == recipe and not c["first"] and not c["profiled"]
            ]
            if warm:
                report[recipe + "_warm_mean_s"] = statistics.mean(warm)
        upstream = report["upstream_warm_mean_s"]
        saved = upstream - report["optimized_warm_mean_s"]
        report.update(saved_s=saved, saved_percent=100 * saved / upstream)
        report["status"] = (
            "numerical_difference"
            if any(e["failing_values"] for e in report["errors"])
            else "complete"
        )
    except BaseException as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        writer.output_dir = normal_output
        save()
    if report["status"] != "complete":
        raise RuntimeError("Predictions differ; see timings.json for absolute and relative errors")
    print(args.output / "timings.json")
    return result


CONFIDENCE_FIELDS = (
    "confidence_score", "ptm", "iptm", "ligand_iptm", "protein_iptm",
    "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
)


def validate_outputs(directory, candidates, samples, output_format="mmcif", affinity=False):
    if not candidates:
        raise RuntimeError("No inputs were predicted")
    for candidate in candidates:
        for rank in range(1 if affinity else samples):
            stem = f"{candidate}_model_{rank}"
            if affinity:
                summary = directory / candidate / f"affinity_{candidate}.json"
                fields = ("affinity_pred_value", "affinity_probability_binary")
            else:
                suffix = ".pdb" if output_format == "pdb" else ".cif"
                structure = directory / candidate / (stem + suffix)
                if not structure.is_file() or structure.stat().st_size == 0:
                    raise RuntimeError(f"Missing output for {stem}")
                summary = directory / candidate / ("confidence_" + stem + ".json")
                fields = CONFIDENCE_FIELDS
            if not summary.is_file():
                raise RuntimeError(f"Missing output for {stem}")
            values = json.loads(summary.read_text())
            if any(key not in values or not isinstance(values[key], (int, float))
                   or not math.isfinite(values[key]) for key in fields):
                raise RuntimeError(f"Invalid confidence summary for {stem}")


def main():
    from boltz.main import cli
    from boltz.data.write.writer import BoltzWriter, BoltzAffinityWriter
    from pytorch_lightning import Trainer
    from .__main__ import InferenceAdapter

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args, arguments = parser.parse_known_args()
    if args.repeats < 1:
        parser.error("Use at least one warm repeat")
    if version("boltz") != SUPPORTED_BOLTZ_VERSION:
        raise RuntimeError(f"This adapter supports Boltz {SUPPORTED_BOLTZ_VERSION}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Run the comparison with one visible GPU")
    if arguments[:1] == ["predict"]:
        arguments.pop(0)
    with cli.commands["predict"].make_context("predict", list(arguments)) as context:
        options = context.params
    args.seed = options["seed"] if options["seed"] is not None else 42
    arguments += ["--seed", str(args.seed), "--no_kernels"]
    original_predict = Trainer.predict
    started = time.perf_counter()

    def predict(trainer, model, *positional, **keywords):
        nonlocal started
        writers = [c for c in trainer.callbacks if isinstance(c, (BoltzWriter, BoltzAffinityWriter))]
        if len(writers) != 1:
            raise RuntimeError("Expected one Boltz prediction writer")
        writer = writers[0]
        phase = "affinity" if isinstance(writer, BoltzAffinityWriter) else "structure"
        args.output = writer.output_dir.parent / "benchmark" / phase
        args.output.mkdir(parents=True)
        args.diffusion_samples = model.predict_args["diffusion_samples"]
        settings = dict(boltz=version("boltz"), input=options["data"], seed=args.seed,
                        phase=phase, setup_s=time.perf_counter()-started,
                        prediction_settings=model.predict_args)
        adapter = InferenceAdapter(optimize=False)
        trainer.callbacks.append(adapter)
        try:
            return benchmark(args, settings, trainer, model, writer, original_predict,
                             positional, keywords)
        finally:
            trainer.callbacks.remove(adapter)
            started = time.perf_counter()

    with patch.object(Trainer, "predict", predict):
        cli(["predict", *arguments], standalone_mode=False)


if __name__ == "__main__":
    main()
