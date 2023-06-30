#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=64MB

# variables
job_id=$1
log_stem=$2

echo "Job ID: $job_id"
echo "Log stem: $log_stem"

# create output path
output_seff="$log_stem""-seff.txt"
output_sacct="$log_stem""-sacct.txt"

# output seff to file
seff $job_id > $output_seff

# output sacct to file
sacct -l -j $job_id > $output_sacct
#export MODULEPATH=/share/apps/spack/testing/modules-files/linux-centos8-skylake_avx512/:$MODULEPATH
#module load python-3.9.10/py-reportseff/2.7.2 
