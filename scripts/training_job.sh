#!/bin/bash
#PBS -N npfl140_training
#PBS -l select=1:ncpus=2:ngpus=1:mem=16gb:gpu_mem=24gb:scratch_local=64gb
#PBS -l walltime=15:00:00

# Print out the basic info about the job
echo "Hello ${PBS_JOBNAME} at $(date) from user ${USER}!"
echo "${PBS_JOBID} is running on node $(hostname -f) in a scratch directory ${SCRATCHDIR}"
echo "Command: $0 ${BRANCH} ${FINE_TUNE_ARGS[*]}"

set -x
set -e

usage() {
    echo "Usage: $0 [--branch BRANCH] [FINE_TUNE_ARGS...]"
    echo
    echo "  --branch BRANCH   Branch or commit to checkout (default: main)"
    echo "  FINE_TUNE_ARGS    Additional arguments passed to 'src/cli.py fine-tune'"
    echo
    echo "HuggingFace authentication:"
    echo "  Place your token in \$PBS_O_WORKDIR/.hf_token (chmod 600)."
    exit 1
}

BRANCH="main"
FINE_TUNE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "${1}" in
        -h|--help)
            usage
            ;;
        --branch)
            BRANCH="${2}"
            shift 2
            ;;
        *)
            FINE_TUNE_ARGS+=("${1}")
            shift
            ;;
    esac
done

REPOSITORY="https://github.com/bugulin/SemEval-2026-11"
outdir="out"

trap 'clean_scratch' EXIT

cd "${SCRATCHDIR}" || exit

# Prepare the environment
git clone --branch="${BRANCH}" --depth=1 "${REPOSITORY}" repo
cd repo || exit
curl -LsSf https://astral.sh/uv/install.sh | env UV_PRINT_QUIET=1 UV_UNMANAGED_INSTALL="bin" sh
mkdir -p "${outdir}"

# Authenticate with HuggingFace
hf_token_file="${PBS_O_WORKDIR}/.hf_token"
if [[ -f "${hf_token_file}" ]]; then
    HF_TOKEN=$(cat "${hf_token_file}")
    export HF_TOKEN
fi

# Run the main task
bin/uv run src/cli.py fine-tune -o "${outdir}" "${FINE_TUNE_ARGS[@]}"

# Clean up
mv "${outdir}" "${PBS_O_WORKDIR}/${PBS_JOBNAME}.${PBS_JOBID}"
