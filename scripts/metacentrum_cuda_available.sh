#!/usr/bin/env bash

# reserve 1 CPU, 1 GPU, 1 GB RAM, 1 GB disc space
#PBS -l select=1:ncpus=1:ngpus=1:mem=1gb:scratch_local=1gb  

# run the job max 10 minutes
#PBS -l walltime=0:10:00

# name of output file
outfile=output.${PBS_JOBID}

# copy the project with .venv into the scratch directory
dirname=${PBS_O_WORKDIR##*/}
cp -r ${PBS_O_WORKDIR} ${SCRATCHDIR}/

# cd into the copied project and create the output file
cd ${SCRATCHDIR}/${dirname}
touch ${outfile}

# get python version and CUDA availability
source .venv/bin/activate
python -c "
import sys
print(sys.version)
import torch
print(torch.cuda.is_available())
" 2&> ${outfile}

# print out the basic info about the job 
echo -e "Hello world at `date` from user ${USER}!\n" >> ${outfile}
echo -e "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR\n" >> ${outfile}

# copy the output file to the directory from where the
# job was submitted
cp ${outfile} ${PBS_O_WORKDIR}/

# apply a scratch automatic cleanup utility
clean_scratch

