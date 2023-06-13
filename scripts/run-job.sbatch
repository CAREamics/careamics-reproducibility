#!/bin/bash
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4 
#SBATCH --time=08:00:00

source $HOME/.bashrc

# variables
name=$1
env=$2
script=$3
logs=$4

echo "Job ID: $SLURM_JOB_ID"
echo "Name: $name"
echo "Env: $env"
echo "Script: $script"
echo "Logs: $logs"

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

# run sacct and seff
seff $job_id > "$logs-seff.txt"

# output sacct to file
#export MODULEPATH=/share/apps/spack/testing/modules-files/linux-centos8-skylake_avx512/:$MODULEPATH
#module load python-3.9.10/py-reportseff/2.7.2 
sacct --format JobID,JobName,Partition,AveRSS,MaxRSS,ReqMem,State,AllocCPUS -j $job_id > "$logs-sacct.txt"
