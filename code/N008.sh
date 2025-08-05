#!/bin/bash
#SBATCH --job-name=N008
#SBATCH --output=N008.out
#SBATCH --error=N008.err
#SBATCH --time=5-10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --partition=long


# Run your Python script
python /data/u_kazuki_software/EMPRISE_2/code/main.py