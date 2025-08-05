#!/bin/bash
#SBATCH --job-name=N003
#SBATCH --output=N003.out
#SBATCH --error=N003.err
#SBATCH --time=7-10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --partition=long


# Run your Python script
python /data/u_kazuki_software/EMPRISE_2/code/main.py