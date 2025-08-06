# Protocol

## Overview

This repository includes scripts for data analysis and outlines the steps. The goal is to ensure transparency, reproducibility, and clarity of the data analysis for all.

## Table of Contents

1. [Objective](#objective)
2. [Analysis](#analysis)
3. [Directory Structure](#directory-structure)
6. [Contact](#contact)


## Objective

The objective is to explore numerosity response in dyscalculia using:
- Modeling:   Numerosity population receptive field models (numerosity pRF model).
- ROIs:       Spatial clustering (AFNI tools).
- Statistics: A linear mixed-effects model and Mann-Whitney U-Test.


## Analysis

### 1. numerosity pRF model. 
1. Open 'main.py'
2. Change the parameters
```bash
# Parameters
#-------------------------------------------------------------------------#
"""
Modify here only for your analysis
condition: digit, spoken, visual, audio1, audio2
code_dir: specify the directory containing "config.json" file.
"""
subject  = "D005"
session  = "visual"
model    = "NumAna" # model name # if you want a spatial smoothing, put 'FWHM' and '3' for the value
space    = "fsaverage" # "fsnative" or "T1w"
target   = "all" # "all" or "split"
code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"
 ```

3. Run 'main.py' via 'sub.sh' as a Slurm job
 ```bash
sbatch sub.sh
 ```
4. Computation takes 2 days with HPC clusters in the MPI-CBS


### 2. ROIs
1. Open '6_ROIs'
2. Change the parameters
 ```bash
# Parameters
#-------------------------------------------------------------------------#
count_thr     = 3 # number of participants overlapping in the surface-based for the next spatial clustering
hemis         = ['L', 'R']
subject_lists = ["N001", "N004", "N005","N006", "N007","N008","N011", "N012"]
surf_imgs     = {}
 ```
3. Run '6_ROIs'
 ```bash
python 6_ROIs.py
 ```

### 3. Statistics
1. Open and run '7_count_vertex.py' for the linear-mixed effects model
2. '7_count_vertex.py' also generates the figures [find here](https://github.com/KazukiMaruo/2025_EMPRISE2/tree/kazu/figures/Sub-all)
3. Open and run '8_nonpara_test.py' for the Mann-Whitney U-Test



## Directory Structure
The code folder should be configured like this:
```bash
experiment/
    ├── Shell/   # AFNI spatial clustering functions
    │
    ├── 1_fd-dvars_dist.py   # scripts 
    ├── ... 
    ├── 8_nonpara_test.py   
    │
    ├── EMPRISE.py   # functions for the projects
    ├── Figures.py   # functions for the figures
    ├── NumpRF.py   # functions for numerosity pRF model
    ├── PySPMs.py   # functions from SPMs
    ├── README.md
    ├── config.json  # specify the parameters
    ├── main.py  # main script for running
    ├── main_figure.py  # main script for generating figures
    └── sub.sh  # bash script for slurm job in HPC cluster
```


## Contact
If you have any questions, feel free to ask me!
 ```bash
kazuki@cbs.mpg.de
 ```



