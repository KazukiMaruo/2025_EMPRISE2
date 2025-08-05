#!/bin/bash
#SBATCH --job-name=N004
#SBATCH --output=N004.out
#SBATCH --error=N004.err
#SBATCH --time=4-10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --partition=long


# Run your Python script
python /data/u_kazuki_software/EMPRISE_2/code/main.py