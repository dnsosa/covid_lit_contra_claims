#!/bin/bash

# Expects to be in the same folder as generic.sh

# Edit this if you want more or fewer jobs in parallel
jobs_in_parallel=8

if [ ! -f "$1" ]
then
    echo "Error: file passed does not exist"
    exit 1
fi

# This convoluted way of counting also works if a final EOL character is missing
n_lines=$(grep -c '^' "$1")

# Use file name for job name
job_name=$(basename "$1" .txt)

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/share/software/user/open/cudnn/7.6.5/lib64"

## pip install -e .
## sbatch -p rbaltman,normal,owners -t 2- --array=1-${n_lines}%${jobs_in_parallel} --nice=1000 --job-name ${job_name} $(dirname "$0")/generic.sh "$1"
sbatch --time=2-00:00:00 \
       --array=1-${n_lines}%${jobs_in_parallel} \
       --mem=16G \
       --partition=rbaltman \
       --gres gpu:1 \
       --mail-type=END,FAIL \
       --mail-user=dnsosa@stanford.edu \
       --job-name ${job_name} $(dirname "$0")/generic.sh "$1"



# --output=./runs/${job_name}-%j.out \
# --error=./runs/${job_name}-%j.err \
