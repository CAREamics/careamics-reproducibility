#!/bin/bash

file=$1

job_id=$(grep "Job ID:" $file | awk '{print $3}')
echo $job_id