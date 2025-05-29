#!/bin/bash
#SBATCH --job-name=bsd68_n2v
#SBATCH --output=bsd68_n2v_%j.out
#SBATCH --error=bsd68_n2v_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load necessary modules
module load python/3.11
module load cuda/12.8

# Activate environment and run script
source /path/to/your/venv/bin/activate
python careamics-bsd68.py