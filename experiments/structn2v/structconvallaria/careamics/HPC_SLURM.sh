#!/bin/bash
#SBATCH --job-name=structconvallaria_structn2v
#SBATCH --output=structconvallaria_structn2v_%j.out
#SBATCH --error=structconvallaria_structn2v_%j.err
#SBATCH --time=0:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1

# Load necessary modules
module load python/3.11
module load cuda/12.8

# Activate environment
source /path/to/your/venv/bin/activate
conda activate careamics

# Run the script
python careamics-structconvallaria.py