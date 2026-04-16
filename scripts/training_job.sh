#!/bin/bash
#PBS -N npfl140_training
#PBS -l select=1:ncpus=1:mem=8gb:scratch_local=1gb
#PBS -l walltime=2:00:00

BRANCH=${1:-main}
REPOSITORY="https://github.com/bugulin/SemEval-2026-11"

outdir="out"

trap 'clean_scratch' EXIT

cd "${SCRATCHDIR}" || exit

# Prepare the environment
git clone --branch="${BRANCH}" --depth=1 "${REPOSITORY}" repo
curl -LsSf https://astral.sh/uv/install.sh | env UV_PRINT_QUIET=1 UV_UNMANAGED_INSTALL="bin" sh
mkdir -p "${outdir}"

# Print out the basic info about the job
echo "Hello ${PBS_JOBNAME} at $(date) from user ${USER}!"
echo "${PBS_JOBID} is running on node $(hostname -f) in a scratch directory ${SCRATCHDIR}"

# Run the main task
bin/uv run src/cli.py fine-tune -o "${outdir}"

# Clean up
mv "${outdir}" "${PBS_O_WORKDIR}/${PBS_JOBNAME}.${PBS_JOBID}"
## Optionally, clean after uv
# uv cache clean
# rm -r "$(uv python dir)"
# rm -r "$(uv tool dir)"
