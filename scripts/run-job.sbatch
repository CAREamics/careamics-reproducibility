#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1 
#SBATCH --time=1:00:00

source $HOME/.bashrc

# variables
name=$3
script=$4
stats=$5

echo "Name: $name"
echo "Script: $script"
echo "Stats: $stats"

# check if conda is installed
if ! command -v conda &> /dev/null; then
   echo "Conda could not be found."
   exit
fi

# check if environment exists
if ! conda env list | grep -q "$name"; then
   echo "Creating conda environment $name"
   #conda env create -y -f "$name"
else
   echo "Conda environment $name already exists."
fi

# activate environment
#conda activate "$name"

# if tensorflow, set environment variable
if [ "$name" = "tf-n2v" ]; then
   echo "export TF"
   #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
fi

# run python script
python "$script"

# statistics
sstat -j $SLURM_JOB_ID