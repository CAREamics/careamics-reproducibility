#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=1GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

source $HOME/.bashrc

# variables
name=$1
env=$2
script=$3
log_stem=$4

echo "Job ID: $SLURM_JOB_ID"
echo "Name: $name"
echo "Env: $env"
echo "Script: $script"
echo "Log stem: $log_stem"


#####################################
############## Conda ################
#####################################

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

#####################################
############### GPU #################
#####################################
# log nvidia smi once
nvidia-smi -i ${CUDA_VISIBLE_DEVICES} > "$log_stem""-nvidia-smi.txt"

# start process to log GPU info
# -i: GPU index
# -s: select metrics, (m) Frame Buffer and Bar1 memory usage and (u) utilization
# -d: monitoring interval
# -o: date format
nvidia-smi dmon -i ${CUDA_VISIBLE_DEVICES} -s mu -d 5 -o TD > "$log_stem""-nvidia-dmon.txt" &
nvidia_job_id=$(pgrep nvidia-smi)


#####################################
############# Training ##############
#####################################

# run python script
echo "Run script"
python $script

#####################################
############### End #################
#####################################

# kill nvidia-smi process
kill $nvidia_job_id
echo "Done!"