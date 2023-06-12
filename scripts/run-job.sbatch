#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 
#SBATCH --time=1:00:00

source $HOME/.bashrc

# variables
name=$1
env=$2
script=$3
log=$4

echo "Job ID: $SLURM_JOB_ID"
echo "Name: $name"
echo "Env: $env"
echo "Script: $script"
echo "Log: $log"

# check if conda is installed
if ! command -v conda &> /dev/null; then
   echo "Conda could not be found."
   exit
fi

# check if environment exists
if ! conda env list | grep -q "$name"; then
   echo "Creating conda environment $name"
   conda env create -f "$env"
else
   echo "Conda environment $name already exists."
fi

# activate environment
conda activate $name
echo "Activated $CONDA_DEFAULT_ENV"

# if tensorflow, set environment variable
if [ $name = "tf-n2v" ]; then
   echo "$name: export CUDA path" 
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
fi

# run python script
echo "Run script"
python $script

# write efficiency
path_to_log="$log""efficiency/"
sh scripts/write-efficiency.sh $SLURM_JOB_ID $path_to_log