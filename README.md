# MINT

## Overview

This repository includes all used for MINT project.
- **NOTE:** This project has been already terminated after six adults data collection

## Table of Contents

1. [Objective](#objective)
2. [Steps for Reproducibility](#steps-for-reproducibility)
3. [Directory Structure](#directory-structure)
4. [Contact](#contact)


## Objective

The objective of this project was to:
- Explore electrophysiological representation (= EEG) of numerosity in infants.

Methods:
- 6 adults
- EEG
- Psychopy
- Multivariate decoding

Figures:
![Stimuli](images/exp_figure.jpg)

## Steps for Reproducibility

This section outlines the general pipeline.

### 1. Conduct the experiment. 
Detailed steps for implementing the experiment are described in the following folder
```bash
cd MINT/experiment
```

### 2. Analyze the EEG data
Detailed steps for the data analysis are described in the following folder
```bash
cd MINT/code
```

## Directory Structure
```bash
MINT/
    ├── code/   # contains scripts for data analysis
    │
    ├── env/   # includes all the packages used in the conda enviironment 
    │
    ├── experiment/ # psychopy scripts and stimuli
    │
    ├── images/ # includes figures
    ├── .gitignore # ignore some folders not to upload to GitHub.
    ├── README.md
    ├── debugging_tips.txt # My debugging notes.
```

## Contact
If you have any questions, feel free to ask me!
 ```bash
kazuki@cbs.mpg.de
 ```