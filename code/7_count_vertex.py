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

# Chose a Group
Group = 'dyscal' # 'control' or 'dyscal'

if Group == 'control':
    subjects = ["N001", "N004", "N005","N006", "N007","N008","N011", "N012"]
else:
    subjects = ["D001", "D002", "D003","D004", "D005"]


for ROI in ROIs:

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
            mu_ROI = mu[ROI_label]

            # Get data for histogram
            max_num  = 5
            bin_size = 0.5
            bins = np.arange(1, max_num + bin_size, bin_size)

            counts, bin_edges = np.histogram(mu_ROI, bins=bins) # mu count
            counts_by_subject[subject][hemi] = counts



    # plot the values
    #-------------------------------------------------------------------------#
    # Extract values
    # Convert each hemisphere's histogram to a list of arrays
    left_all = np.array([counts_by_subject[subj]['L'] for subj in subjects])
    right_all = np.array([counts_by_subject[subj]['R'] for subj in subjects])

    # Compute mean across subjects
    left_mean = np.mean(left_all, axis=0)
    right_mean = np.mean(right_all, axis=0)

    # Compute sem
    n_subjects = len(subjects)
    left_sem = np.std(left_all, axis=0, ddof=1) / np.sqrt(n_subjects)
    right_sem = np.std(right_all, axis=0, ddof=1) / np.sqrt(n_subjects)

    # Plot
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # pandas dataframe for linear mixed effects model
    rows = []
    for subj in subjects:
        for hemi in hemis:
            values = counts_by_subject[subj][hemi]  # replace with whatever measure you're using
            for i, val in enumerate(values):
                rows.append({
                    'subject': subj,
                    'hemisphere': hemi,
                    'bin_center': bin_centers[i],
                    'value': val
                })

    df = pd.DataFrame(rows)

    # Model: fixed effects for hemisphere and bin_center, random intercepts for subject
    pvals = []
    for hemi in hemis:
        print(f"\n--- Hemisphere: {hemi} ---")
        
        df_hemi = df[df['hemisphere'] == hemi]
        
        # Fit LMM with random intercept per subject
        model = smf.mixedlm("value ~ bin_center", df_hemi, groups=df_hemi["subject"])
        result = model.fit()
        
        pvals.append(result.pvalues['bin_center'])
        print(result.summary())


    # correlation coefficients
    r_left, p_left = pearsonr(bin_centers, left_mean)
    r_right, p_right = pearsonr(bin_centers, right_mean)

    # Fit linear (degree=1) and quadratic (degree=2) models for each hemisphere
    left_lin_fit = np.polyfit(bin_centers, left_mean, 1)
    left_quad_fit = np.polyfit(bin_centers, left_mean, 2)

    right_lin_fit = np.polyfit(bin_centers, right_mean, 1)
    right_quad_fit = np.polyfit(bin_centers, right_mean, 2)

    # Generate smooth predictions
    x_fit = np.linspace(bin_centers[0], bin_centers[-1], 200)
    left_lin_pred = np.polyval(left_lin_fit, x_fit)
    left_quad_pred = np.polyval(left_quad_fit, x_fit)

    right_lin_pred = np.polyval(right_lin_fit, x_fit)
    right_quad_pred = np.polyval(right_quad_fit, x_fit)


    # param for plot
    larger   = np.maximum(left_mean, right_mean)
    ymax     = int(larger.max())
    path     = '/data/u_kazuki_software/EMPRISE_2/figures/Sub-all/'
    fig      = f'LMM-mu-{ROI}-{Group}.png'
    pathfig  = os.path.join(path, fig)
    if Group == 'control':
        color_1 = 'blue'
        color_2 = 'skyblue'
    else:
        color_1 = 'red'
        color_2 = 'firebrick'
    # initialize plot
    plt.figure(figsize=(10, 6))

    plt.errorbar(bin_centers, left_mean, yerr=left_sem, label='hemi-L', fmt='o', capsize=2, color=color_1)
    plt.errorbar(bin_centers, right_mean, yerr=right_sem, label='hemi-R', fmt='o', capsize=2, color=color_2)

    # Fit lines
    plt.plot(x_fit, left_lin_pred, '-', color=color_1, label=f'hemi-L, fit-linear (r = {r_left:.2f}, p = {pvals[0]:.3f}, n = {n_subjects})')
    # plt.plot(x_fit, left_quad_pred, '-', color='blue', label='hemi-L, fit-quadratic (R2)')
    plt.plot(x_fit, right_lin_pred, '-', color=color_2, label=f'hemi-R, fit-linear (r = {r_right:.2f}, p = {pvals[0]:.3f}, n = {n_subjects})')
    # plt.plot(x_fit, right_quad_pred, '-', color='skyblue',label='hemi-R, fit-quadratic (R2)')

    plt.xlim(1,max_num)
    if ROI == 'NPO':
        plt.ylim(0,90)
    elif ROI == 'NPC':
        plt.ylim(0,700)
    elif ROI == 'NF':
        plt.ylim(0,80)   
    else:
        plt.ylim(0,ymax+50)
    plt.xlabel('preferred numerosity')
    plt.ylabel('frequency of vertices')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig(pathfig, dpi=300, bbox_inches='tight')
    plt.show()