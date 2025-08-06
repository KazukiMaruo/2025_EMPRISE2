# Protocol

## Overview

This repository includes scripts for data analysis and outlines the steps. The goal is to ensure transparency, reproducibility, and clarity of the data analysis for all.

## Table of Contents

1. [Objective](#objective)
2. [EEG preprocessing](#EEG-preprocessing)
3. [Single level analysis](#Single-level-analysis)
4. [Group level analysis](#Group-level-analysis)
5. [Directory Structure](#directory-structure)
6. [Contact](#contact)


## Objective

The objective is to explore numerosity response in dyscalculia using:
- Modeling:   Numerosity population receptive field models (numerosity pRF model).
- ROIs:       Spatial clustering (AFNI tools).
- Statistics: A linear mixed-effects model.


## EEG preprocessing

### 1. Download the code folder. 
1. Open https://download-directory.github.io/
2. Paste the URL of 'code' folder
3. Place it in the 'MINT' folder

### 2. Run preprocessing

1. go to 'pre-process' folder
2. Run 'preprocess.py'
 ```bash
python preprocess.py
 ```


## Single level analysis
### 1. Decoding: Time-resolved logistic regression.
1. go to '1st-level' folder
2. run '1_logistic_multisessions.py'
 ```bash
python 1_logistic_multisessions.py
 ```

### 2. Decoding: EEGNetV4 (=Convolutional Neural Network).
1. go to '1st-level' folder
2. run '2_eegnet_global.py'
 ```bash
python 2_eegnet_global.py
 ```

### 3. Visualize the decoding results
1. go to '1st-level' folder
2. run '3_logistic_visualization.ipynb' or '3_eegnet_visualization.ipynb'


### 3. Generate representational dissimilarity matrix.
1. go to '1st-level' folder
2. run '4_rdm_generator.py'
 ```bash
python 4_rdm_generator.py
 ```
- **NOTE:** 'param.json' in [stimuli folder](../experiment/stimuli/visual) is necessary.
3. output 4 different dissimilarity matrixes by .png and .npy, and the color bar in [RDM folder](./1st-level/RDM), larger value indicates more dissimilar
-  'rdm_area_of_a_single_dot' represents dissimilarity based on the area of a single dot in an image.
-  'rdm_area_of_total_dots' represents dissimilarity based on the area of total dots in an image.
-  'rdm_circumference_of_total_dots' represents dissimilarity based on the circumference of total dots in an image.
-  'rdm_numerosity' represents dissimilarity based on the number of dot in an image.

4. run '4_rdm_generator_spatialfrequency.py' 
 ```bash
python 4_rdm_generator_spatialfrequency.py
 ```
- **NOTE:** more than 10 hours will be needed until completed
5. output 1 dissimilarity matrix by .png and .npy in [RDM folder](./1st-level/RDM)
-  'rdm_spatial_frequency_raw' represents matrix without cleaning up
-  'rdm_spatial_frequency' represents dissimilarity matrix based on the spatial frequency in an image.


## Group level analysis
1. go to '2nd-level' folder
2. run '1_logistic.ipynb' or '2_eegnet.ipynb'


## Directory Structure
The code folder should be configured like this:
```bash
experiment/
    ├── 1st-level/   # single level analysis
    │
    ├── 2nd-level/   # group level analysis 
    │
    ├── Preprocess/   # EEG preprocessing
    │
    ├── config.json   # general info for analysis
    ├── MINT.code-workspace   # setting up VScode interface
    ├── README.md   # overview of the directory
    └── utils.py  # function lists
```


## Contact
If you have any questions, feel free to ask me!
 ```bash
kazuki@cbs.mpg.de
 ```



