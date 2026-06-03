#!/bin/bash
#SBATCH --account=project_465003017
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

module load Local-quantum/default
module load fiqci-vtt-qiskit

cd /scratch/project_465003017/jferreir/Junction-Hackaton
python3 pre/src/pipeline.py
