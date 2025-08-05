# install library
#-------------------------------------------------------------------------#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import nibabel as nib
import nibabel.freesurfer.io as fsio
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
import EMPRISE
from nilearn import surface
from surfplot import Plot
from nibabel.gifti import GiftiImage, GiftiDataArray

# Parameters
#-------------------------------------------------------------------------#
"""
Modify here only for your analysis
condition: digit, spoken, visual, audio1, audio2
code_dir: specify the directory containing "config.json" file.
"""
hemis = ['L', 'R']
ROIs  = ['NTO','NPO','NPC','NF']
Groups =['control', 'dyscal'] # 'control' or 'dyscal'

group_data = {}

for Group in Groups:
    group_data[Group] = {}

    if Group == 'control':
        subjects = ["N001", "N004", "N005","N006", "N007","N008","N011", "N012"]
    else:
        subjects = ["D001", "D002", "D003","D004", "D005"]


    for ROI in ROIs:
        group_data[Group][ROI] = {}

        # data storage
        counts_by_subject = {}
        for subject in subjects: # hemisphere loop
            counts_by_subject[subject] = {}
            for hemi in hemis: # subject loop

                session  = "visual"
                model    = "NumAna" # model name # if you want a spatial smoothing, put 'FWHM' and '3' for the value
                space    = "fsaverage" # "fsnative" or "T1w"
                target   = "all" # "all" or "split"
                code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"

                # ROIs mask: participant count based
                path = f'/data/pt_02495/emprise7t_2_analysis/derivatives/numprf/sub-all/ses-{session}/model-{model}/'
                if ROI == 'NTO' and hemi == 'R':
                    file_1      = f'ROIs_cluster_{hemi}_{ROI}-1.label'
                    file_2      = f'ROIs_cluster_{hemi}_{ROI}-2.label'
                    filepath_1  = os.path.join(path, file_1)
                    filepath_2  = os.path.join(path, file_2)
                    ROI_label_1 = fsio.read_label(filepath_1)
                    ROI_label_2 = fsio.read_label(filepath_2)
                    ROI_label   = np.concatenate([ROI_label_1, ROI_label_2]) 
                else:
                    file        = f'ROIs_cluster_{hemi}_{ROI}.label'
                    filepath    = os.path.join(path, file)
                    ROI_label   = fsio.read_label(filepath) # Load label file

                # start Session class
                #-------------------------------------------------------------------------#
                sess = EMPRISE.Session(subject, session)
                # start Model class
                #-------------------------------------------------------------------------#
                mod = EMPRISE.Model(subject, session, model, space_id=space)


                # Get thresholded mu values
                res_file = mod.get_results_file(hemi)
                filepath = res_file[:res_file.find('numprf.mat')]
                mu_map   = filepath + 'mu_thr-Rsqmb,p=0.05B.surf.gii'
                if os.path.exists(res_file) and os.path.exists(mu_map):
                    NpRF    = sp.io.loadmat(res_file)
                    image   = nib.load(mu_map)
                    mu      = image.darrays[0].data
                else:
                    print(f'file {filepath} does not exsist.')


                # extract the mu values in the ROI
                mu_ROI   = mu[ROI_label]

                # Get data for histogram
                max_num  = 5
                bin_size = 0.5
                bins = np.arange(1, max_num + bin_size, bin_size)

                counts, bin_edges = np.histogram(mu_ROI, bins=bins) # mu count
                counts_by_subject[subject][hemi] = counts
        
        group_data[Group][ROI] = counts_by_subject



# nonparametric test
#-------------------------------------------------------------------------#
from scipy.stats import mannwhitneyu

con_subj = ["N001", "N004", "N005","N006", "N007","N008","N011", "N012"]
dys_subj = ["D001", "D002", "D003","D004", "D005"]

ROI  = 'NPC'
hemi = 'R'

# Example: get your data from storage (fill in your actual source)
control_data = []  # list of values from control subjects for ROI='NPC', hemi='R'
dyscal_data  = []  # list of values from dyscal subjects for ROI='NPC', hemi='R'

# Example — if you're pulling from nested dict:
for subj in con_subj:
    control_data.append(group_data['control'][ROI][subj][hemi])
for subj in dys_subj:
    dyscal_data.append(group_data['dyscal'][ROI][subj][hemi])

# Apply Mann-Whitney U test
statistic, p_value = mannwhitneyu(control_data, dyscal_data, alternative='two-sided')

# Print result
print(f"ROI = {ROI}, hemi = {hemi}")
for i, val in enumerate(p_value):
    i += 1
    print(f"{i}: p = {val:.5f}")



