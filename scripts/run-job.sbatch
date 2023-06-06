#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4 
#SBATCH --time=8:00:00

source $HOME/.bashrc

# variables
name=$1
script=$2
stats=$3

# check if conda is installed
if ! command -v conda &> /dev/null; then
   echo "Conda could not be found."
   exit
fi

# check if environment exists
if ! conda env list | grep -q "$name"; then
   echo "Creating conda environment $name"
   conda env create -y -f "$name"
else
   echo "Conda environment $name already exists."
fi

# activate environment
conda activate "$name"

# if tensorflow, set environment variable
if [ "$name" = "tf-n2v" ]; then
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
fi

# run python script
python "$script"

# write statistics
sstat -j $SLURM_JOB_ID > "$stats"