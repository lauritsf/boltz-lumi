#!/bin/bash -l
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${BOLTZ_LUMI_ENV:-}" ]]; then
    source "$BOLTZ_LUMI_ENV"
fi
: "${BOLTZ_LUMI_WORK:?Set BOLTZ_LUMI_WORK in your environment or site file}"
: "${BOLTZ_LUMI_IMAGE:?Set BOLTZ_LUMI_IMAGE to a compatible ROCm/PyTorch image}"
[[ $# -gt 0 ]] || { echo 'Usage: scripts/lumi.sh setup | python ...' >&2; exit 2; }
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings
export CXX="${BOLTZ_LUMI_CXX:-g++}"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
# PyTorch's HIP loader orders sources through a set; keep cache links stable.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-$BOLTZ_LUMI_WORK/cache/boltz}"
export XDG_CACHE_HOME="$BOLTZ_LUMI_WORK/cache"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/inductor"
export SINGULARITYENV_PYTHONPATH="$ROOT/src:$BOLTZ_LUMI_WORK/venv/lib/python3.12/site-packages:/opt/venv/lib/python3.12/site-packages"
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
export SINGULARITYENV_PREPEND_PATH="$BOLTZ_LUMI_WORK/venv/bin"
mkdir -p "$BOLTZ_LUMI_WORK" "$BOLTZ_CACHE"
if [[ "${1:-}" == setup ]]; then
    singularity exec "$BOLTZ_LUMI_IMAGE" python -m venv --system-site-packages "$BOLTZ_LUMI_WORK/venv"
    singularity exec "$BOLTZ_LUMI_IMAGE" "$BOLTZ_LUMI_WORK/venv/bin/python" - <<'PYTHON'
import torch
if not torch.version.hip or torch.__version__.split("+")[0] != "2.11.0":
    raise SystemExit("Use the tested ROCm/PyTorch image")
PYTHON
    singularity exec "$BOLTZ_LUMI_IMAGE" "$BOLTZ_LUMI_WORK/venv/bin/python" -m pip install \
        -r "$ROOT/requirements.txt" -c "$ROOT/container-constraints.txt"
    exit
fi
COMMAND="$1"
shift
if [[ -x "$BOLTZ_LUMI_WORK/venv/bin/$COMMAND" ]]; then
    COMMAND="$BOLTZ_LUMI_WORK/venv/bin/$COMMAND"
fi
exec singularity exec "$BOLTZ_LUMI_IMAGE" "$COMMAND" "$@"
