#!/bin/bash
#SBATCH --partition=cpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=09:00:00

# variables
job_id=$1
output_folder=$2

# wait while job is running
job_id_running=$(squeue | grep $job_id | awk '{print $1}')
while [[ -n $job_id_running ]]
do
   sleep 120
   job_id_running=$(squeue | grep $job_id | awk '{print $1}')
done

# create output path
output_path_seff="$output_folder""$job_id""-seff.txt"
output_path_sacct="$output_folder""$job_id""-sacct.txt"

# output seff to file
seff $job_id > $output_path_seff

# output sacct to file
sacct --format JobID,JobName,Partition,AveRSS,MaxRSS,ReqMem,State,AllocCPUS -j $job_id > $output_path_sacct
