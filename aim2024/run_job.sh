#!/bin/bash

# SLURM job options (name, compute nodes, job time)
#SBATCH --time=03:00:00
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --job-name=test-job2
#SBATCH --output=test-job2.out
#SBATCH --gres=gpu:1           # Request 1 A100 GPU
#SBATCH --cpus-per-task=12     # Request 8 CPU cores
#SBATCH --mem=16000MB          # Request 12 GB of memory

nvidia-smi


# Activate the virtual environment
source /home/sprice/satellite_v2/samEnv/bin/activate

# Run your Python script
python /home/sprice/satellite_v2/aim2024/performanceCalculationsJoined.py

# Deactivate the virtual environment
deactivate
