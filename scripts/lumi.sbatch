#!/bin/bash -l
#SBATCH --job-name=boltz
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=mi250:1
#SBATCH --cpus-per-task=7
#SBATCH --threads-per-core=1
#SBATCH --mem=60G
#SBATCH --time=02:00:00
set -euo pipefail
export SLURM_EXPORT_ENV=ALL
# Let Boltz fork its prediction workers instead of using Slurm task ranks.
srun --job-name=interactive --export=ALL --ntasks=1 \
    --gpus-per-task="$SLURM_GPUS_PER_NODE" --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
    bash scripts/lumi.sh "$@"
