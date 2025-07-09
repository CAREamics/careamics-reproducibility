#!/bin/bash
#SBATCH --job-name=convallaria_n2v
#SBATCH --output=convallaria_n2v_%j.out
#SBATCH --error=convallaria_n2v_%j.err
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1

# Load necessary modules
module load python/3.11
module load cuda/12.8

# Activate environment and run script
source /path/to/your/venv/bin/activate
conda activate careamics

# Run the script
python careamics-convallaria.py