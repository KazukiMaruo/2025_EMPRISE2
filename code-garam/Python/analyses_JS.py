# -*- coding: utf-8 -*-
"""
Analyses of the EMPRISE data set
by Joram Soch <soch@cbs.mpg.de>
"""


# import modules
import os
import time
import shutil
import NumpRF
import EMPRISE
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


### NumpRF empirical analysis (True_False_ar1_3_V2), 07/12/2023 ###############

# define analyses
subs   = EMPRISE.adults
sess   = ['visual', 'audio']
spaces = EMPRISE.spaces
model  = {'avg': [True, False], 'noise': 'ar1', 'hrfs': 3}
ver    =  'V2'

# specify folder
targ_dir = EMPRISE.tool_dir + '../../../Analyses/'

# perform analyses
for sub in subs:
    for ses in sess:
        for space in spaces:
                
            # determine results filename
            mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])+'_'+ver
            subj_dir = EMPRISE.Model(sub, ses, mod_name, space).get_model_dir() + '/'
            filepath = 'sub-' + sub + '_ses-' + ses + '_model-' + mod_name + '_hemi-' + 'R' + '_space-' + space + '_'
            res_file = filepath + 'numprf.mat'
            para_map = filepath + 'mu.surf.gii'
            
            # if results file already exists, let the user know
            if os.path.isfile(subj_dir+res_file):
                
                # display message
                print('\n\n-> Subject "{}", Session "{}", Model "{}", Space "{}":'. \
                      format(sub, ses, mod_name, space))
                print('   - results file does already exist, model is not estimated!')
                if not os.path.isfile(targ_dir+para_map):
                    shutil.copy(subj_dir+para_map, targ_dir)
            
            # if results file does not yet exist, perform analysis
            else:
            
                # run numerosity model
                try:
                    mod = EMPRISE.Model(sub, ses, mod_name, space)
                    mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver)
                    shutil.copy(subj_dir+para_map, targ_dir)
                except FileNotFoundError:
                    continue


### threshold surface images, 29/11/2023 ######################################

# Step 1: threshold tuning maps (17/11/2023)
# mod = EMPRISE.Model('003','visual','True_False_iid_1','fsnative')
# mod.threshold_maps('Rsqmb,0.2')
# mod = EMPRISE.Model('009','audio','True_False_iid_1','fsnative')
# mod.threshold_maps('Rsqmb,0.2')

# Step 2: cluster using SurfClust
# AFNI /data/hu_soch/ownCloud/MPI/EMPRISE/tools/EMPRISE/code/Shell/cluster_surface.sh 003 visual True_False_iid_1 space-fsnative_Rsq_thr-Rsqmb,0.2 ses-visual
# AFNI /data/hu_soch/ownCloud/MPI/EMPRISE/tools/EMPRISE/code/Shell/cluster_surface.sh 009 audio True_False_iid_1 space-fsnative_Rsq_thr-Rsqmb,0.2 ses-audio

# Step 3: visualize surface maps
# mod = EMPRISE.Model('003','visual','True_False_iid_1','fsnative')
# mod.visualize_maps(img='space-fsnative_Rsq_thr-Rsqmb,0.2_cls-SurfClust_cls')
# mod = EMPRISE.Model('009','audio','True_False_iid_1','fsnative')
# mod.visualize_maps(img='space-fsnative_Rsq_thr-Rsqmb,0.2_cls-SurfClust_cls')


### NumpRF empirical analysis (V2), 23/11/2023 ################################

# # define analyses
# subs   = EMPRISE.adults
# sess   = ['visual', 'audio']
# spaces = EMPRISE.spaces
# model  = {'avg': [True, False], 'noise': 'iid', 'hrfs': 1}
# ver    =  'V2'

# # specify folder
# targ_dir = EMPRISE.tool_dir + '../../../Analyses/'

# # perform analyses
# for sub in subs:
#     for ses in sess:
#         for space in spaces:
                
#             # determine results filename
#             mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])+'_'+ver
#             subj_dir = EMPRISE.Model(sub, ses, mod_name, space).get_model_dir() + '/'
#             filepath = 'sub-' + sub + '_ses-' + ses + '_model-' + mod_name + '_hemi-' + 'R' + '_space-' + space + '_'
#             res_file = filepath + 'numprf.mat'
#             para_map = filepath + 'mu.surf.gii'
            
#             # if results file already exists, let the user know
#             if os.path.isfile(subj_dir+res_file):
                
#                 # display message
#                 print('\n\n-> Subject "{}", Session "{}", Model "{}", Space "{}":'. \
#                       format(sub, ses, mod_name, space))
#                 print('   - results file does already exist, model is not estimated!')
#                 if not os.path.isfile(targ_dir+para_map):
#                     shutil.copy(subj_dir+para_map, targ_dir)
            
#             # if results file does not yet exist, perform analysis
#             else:
            
#                 # run numerosity model
#                 try:
#                     mod = EMPRISE.Model(sub, ses, mod_name, space)
#                     mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'], ver=ver)
#                     shutil.copy(subj_dir+para_map, targ_dir)
#                 except FileNotFoundError:
#                     continue


### NumpRF: old vs. new grid, 22/11/2023 ######################################

# # specify old grid
# mu_old   = np.concatenate((np.arange(0.05, 6.05, 0.05), \
#                            10*np.power(2, np.arange(0,8))))
# fwhm_old = np.concatenate((np.arange(0.3, 18.3, 0.3), \
#                            24*np.power(2, np.arange(0,4))))

# # specify new grid
# mu_new   = np.concatenate((np.arange(0.80, 5.25, 0.05), np.array([20])))
# mu_log   = np.log(mu_new)
# sig_log  = np.arange(0.05, 3.05, 0.05)
# fwhm_new = np.zeros((mu_new.size, sig_log.size))
# for i in range(mu_new.size):
#     mu, fwhm_new[i,:] = NumpRF.log2lin(mu_log[i], sig_log)

# # plot old grid
# fig = plt.figure(figsize=(16,9))
# ax  = fig.add_subplot(111)
# for i in range(mu_old.size):
#     ax.plot(fwhm_old, mu_old[i]*np.ones(fwhm_old.shape), '.b', markersize=2)
# ax.axis([0, 100, -0.1, 6.1])
# ax.set_xlabel('FWHM tuning width (0-18, 24, 48, 96, 192 [{} values])'. \
#               format(fwhm_old.size), fontsize=16)
# ax.set_ylabel('preferred numerosity (0-6, 10, 20, ..., 640, 1280 [{} values])'. \
#               format(mu_old.size), fontsize=16)
# ax.set_title('parameter grid: old version', fontsize=24)
# fig.savefig('Figure_outputs/NumpRF_estimate_MLE_old.png', dpi=150)

# # plot new grid
# fig = plt.figure(figsize=(16,9))
# ax  = fig.add_subplot(111)
# for i in range(mu_new.size):
#     ax.plot(fwhm_new[i,:], mu_new[i]*np.ones(sig_log.shape), '.b', markersize=2)
# ax.axis([0, 100, -0.1, 6.1])
# ax.set_xlabel('FWHM tuning width (0-3 in logarithmic space) [{} values])'. \
#               format(sig_log.size), fontsize=16)
# ax.set_ylabel('preferred numerosity (0.8-5.2, 20) [{} values])'. \
#               format(mu_new.size), fontsize=16)
# ax.set_title('parameter grid: new version', fontsize=24)
# fig.savefig('Figure_outputs/NumpRF_estimate_MLE_new.png', dpi=150)