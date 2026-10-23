"""Build native kernels lazily, in the writable workspace."""

from functools import lru_cache
import os
from pathlib import Path
import shutil


@lru_cache(maxsize=None)
def extension(name):
    from torch.utils.cpp_extension import load

    source = Path(__file__).parent
    workspace = Path(os.environ.get("BOLTZ_LUMI_WORK", Path.home() / ".cache/boltz-lumi"))
    build = workspace / "build" / name
    staged = build / "sources"
    staged.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in (".cpp", ".cu"):
        original = source / (name + "_hip" + suffix)
        destination = staged / original.name
        # HIPify writes .hip files next to its inputs.
        if not destination.exists() or destination.read_bytes() != original.read_bytes():
            shutil.copyfile(original, destination)
        paths.append(str(destination))
    # Layer norm matches PyTorch with fused multiply-add enabled.
    contraction = "fast" if name == "multiplication" else "off"
    return load(
        name="boltz_lumi_" + name,
        sources=paths,
        build_directory=str(build),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-fno-fast-math", "-ffp-contract=" + contraction],
    )
