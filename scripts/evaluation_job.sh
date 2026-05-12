#!/bin/bash
#PBS -N npfl140_training
#PBS -l select=1:ncpus=1:ngpus=1:mem=32gb:gpu_mem=24gb:scratch_local=64gb
#PBS -l walltime=3:00:00

## JOB SETTINGS
REPOSITORY="https://github.com/bugulin/SemEval-2026-11"
BRANCH="jachym-fine-tuning"
# arguments passed to src/cli.py
SCRIPT_ARGS="download-and-evaluate --thinking --dataset datasets/validation.json --adapter Jajasek/llama-3.1-8B-syllogism-grpo-lora"
# Optional: path to the file containing the huggingface token, relative to the directory from which the job is submitted
HF_TOKEN_RELATIVE_PATH=".hf_token"

# Print out the basic info about the job
echo "Hello ${PBS_JOBNAME} at $(date) from user ${USER}!"
echo "${PBS_JOBID} is running on node $(hostname -f) in a scratch directory ${SCRATCHDIR}"
echo "Repository: $REPOSITORY"
echo "Branch: $BRANCH"
echo "Command: src/cli.py $SCRIPT_ARGS"

# Exit on any error
set -e
# Debug output - print all executed lines to stdout
set -x

trap 'clean_scratch' EXIT

cd "${SCRATCHDIR}" || exit

# Prepare the environment
git clone --branch="${BRANCH}" --depth=1 "${REPOSITORY}" repo
cd repo || exit
curl -LsSf https://astral.sh/uv/install.sh | env UV_PRINT_QUIET=1 UV_UNMANAGED_INSTALL="bin" sh

# Authenticate with HuggingFace
HF_TOKEN_PATH="${PBS_O_WORKDIR}/${HF_TOKEN_RELATIVE_PATH}"
if [[ -f "${HF_TOKEN_PATH}" ]]; then
    export HF_TOKEN_PATH
fi

# Run the main task
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
bin/uv run src/cli.py ${SCRIPT_ARGS}
