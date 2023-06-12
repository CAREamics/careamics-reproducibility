#!/bin/bash

# variables
job_id=$1
output_folder=$2

# create output path
output_path_seff="$output_folder""$job_id""-seff.txt"
output_path_sacct="$output_folder""$job_id""-sacct.txt"

# output seff to file
seff $job_id > $output_path_seff

# output sacct to file
sacct --format JobID,JobName,Partition,AveRSS,MaxRSS,ReqMem,State,AllocCPUS -j $job_id > $output_path_sacct
