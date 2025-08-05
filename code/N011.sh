#!/bin/bash
#SBATCH --job-name=N011
#SBATCH --output=N011.out
#SBATCH --error=N011.err
#SBATCH --time=5-10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --partition=long


# Run your Python script
python /data/u_kazuki_software/EMPRISE_2/code/main.py