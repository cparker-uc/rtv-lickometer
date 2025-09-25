#!/bin/bash
#SBATCH --job-name=d3d_training
#SBATCH -p public
#SBATCH -N 16
#SBATCH --time 24:00:00
#SBATCH --mem=16GB
#SBATCH -o "slurm_logs/network_training_%j.out"
#SBATCH -e "slurm_logs/network_training_%j.err"

source ~/.bash_profile_conda
mamba activate tf
python network_training.py
