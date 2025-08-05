# -*- coding: utf-8 -*-
"""
Analyses of the EMPRISE data set
by Joram Soch <soch@cbs.mpg.de>
"""


# import modules
import os
import time
import EMPRISE
import NumpRF
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


### Simulation C: NumpRF/popeye results, 25/09/2023 ###########################

# specify possible settings
avg1s  = [True, False]
avg2s  = [True, False]
corrs  = ['iid', 'ar1']
orders = [1, 2, 3]

# load simualtion data
filename  = r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\Simulation_C.mat'
folder    = os.path.dirname(filename)
file, ext = os.path.splitext(filename)
Sim       = sp.io.loadmat(filename)
mu_true   = np.squeeze(Sim['mu'])
fwhm_true = np.squeeze(Sim['fwhm'])
SNR_true  = Sim['SNR']
num_vox   = mu_true.size

# load simulation results (NumpRF)
Res   = []
names = []
times = []
for avg1 in avg1s:
    for avg2 in avg2s:
        for noise in corrs:
            for hrfs in orders:
                
                # specify results file
                res_file = file+'_'+str(avg1)+'_'+str(avg2)+'_'+str(noise)+'_'+str(hrfs)+'.mat'
                Sim      = sp.io.loadmat(res_file)
                
                # load simulation results
                names.append(str(avg1)+' , '+str(avg2)+' , '+str(noise)+' , '+str(hrfs))
                times.append(np.squeeze(Sim['time'])/60)
                res_dict = {'mu_est':    np.squeeze(Sim['mu_est']), \
                            'fwhm_est':  np.squeeze(Sim['fwhm_est']), \
                            'beta_est':  np.squeeze(Sim['beta_est']), \
                            'MLL_est':   np.squeeze(Sim['MLL_est']), \
                            'MLL_null':  np.squeeze(Sim['MLL_null']), \
                            'MLL_const': np.squeeze(Sim['MLL_const']), \
                            'k_est':     Sim['k_est'][0,0], \
                            'k_null':    Sim['k_null'][0,0], \
                            'k_const':   Sim['k_const'][0,0]}
                Res.append(res_dict)

# load simulation results (popeye)
filenames =[r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulation_results\Simulation_C_popeye.csv', \
            r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulation_results\Simulation_C_popeye_True_False.csv', \
            r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulation_results\Simulation_C_popeye_True_False_iid_3.csv']
settings  = ['True , True , iid , 1', \
             'True , False , iid , 1', \
             'True , False , iid , 3']
times_pe  = [13, 13, 60]
for s, filename in enumerate(filenames):
    Sim      = pd.read_csv(filename)
    res_dict = {'mu_est':   np.array(Sim['nums_est']), \
                'fwhm_est': np.array(Sim['fwhm_est']), \
                'Rsq_est':  np.array(Sim['rsq_est'])}
    Res.append(res_dict)
    names.append('popeye: '+settings[s])
    times.append(times_pe[s]/60)

# summarize simulation results
del Sim, res_file, res_dict, settings, times_pe
E        = len(names)
methA    = range(0,24)
methB    = range(24,E)
ind0     = range(0,500)
ind1     = range(500,1500)
ind2     = range(1500,2000)
cols     = 'rgb'
titles   =['no-signal voxels', 'low-numerosity voxels', 'high-numerosity voxels']
times    = np.array(times)
mu_thr   = [0.5, 5.5]
fwhm_thr = [1, 15]
beta_thr = [0, np.inf]
Rsq_thr  =  0.3

# identify voxel categories (NumpRF)
TPR_A  = []
TNR0_A = []
TNR2_A = []
TPR_B  = []
TNR0_B = []
TNR2_B = []
for i in methA:
    R = Res[i]
    mu_est   = R['mu_est']
    fwhm_est = R['fwhm_est']
    beta_est = R['beta_est']
    AIC_est  = -2*R['MLL_est']  + 2*R['k_est']
    AIC_null = -2*R['MLL_null'] + 2*R['k_null']
    ind_mfb  = np.logical_or(
                    np.logical_or(
                        np.logical_or(mu_est<mu_thr[0], mu_est>mu_thr[1]),
                        np.logical_or(fwhm_est<fwhm_thr[0], fwhm_est>fwhm_thr[1])),
                    np.logical_or(beta_est<beta_thr[0], beta_est>beta_thr[1]))
    TPR_A.append(np.sum(AIC_est[ind1]<AIC_null[ind1])/len(ind1))
    TNR0_A.append(np.sum(AIC_null[ind0]<AIC_est[ind0])/len(ind0))
    TNR2_A.append(np.sum(AIC_null[ind2]<AIC_est[ind2])/len(ind2))
    TPR_B.append(np.sum(~ind_mfb[ind1])/len(ind1))
    TNR0_B.append(np.sum(ind_mfb[ind0])/len(ind0))
    TNR2_B.append(np.sum(ind_mfb[ind2])/len(ind2))

# identify voxel categories (popeye)
for i in methB:
    R = Res[i]
    mu_est   = R['mu_est']
    fwhm_est = R['fwhm_est']
    Rsq_est  = R['Rsq_est']
    ind_mf   = np.logical_or(
                   np.logical_or(mu_est<mu_thr[0], mu_est>mu_thr[1]),
                   np.logical_or(fwhm_est<fwhm_thr[0], fwhm_est>fwhm_thr[1]))
    TPR_A.append(np.sum(Rsq_est[ind1]>Rsq_thr)/len(ind1))
    TNR0_A.append(np.sum(Rsq_est[ind0]<Rsq_thr)/len(ind0))
    TNR2_A.append(np.sum(Rsq_est[ind2]<Rsq_thr)/len(ind2))
    TPR_B.append(np.sum(~ind_mf[ind1])/len(ind1))
    TNR0_B.append(np.sum(~ind_mf[ind0])/len(ind0))
    TNR2_B.append(np.sum(~ind_mf[ind2])/len(ind2))

# calculate correlations (NumpRF & popeye)
mu_corr   = [np.corrcoef(mu_true[ind1], R['mu_est'][ind1])[0,1] for R in Res]
fwhm_corr = [np.corrcoef(fwhm_true[ind1], R['fwhm_est'][ind1])[0,1] for R in Res]
mu_MAE    = [np.mean(np.abs(R['mu_est'][ind1]-mu_true[ind1])) for R in Res]
fwhm_MAE  = [np.mean(np.abs(R['fwhm_est'][ind1]-fwhm_true[ind1])) for R in Res]

# plot simulation results (computation time)
fig, ax = plt.subplots(figsize=(16,9))
ax.barh(np.arange(E), times, color='w', edgecolor='k')
ax.set_yticks(np.arange(E), labels=names, fontsize=12)
ax.axis([0, np.max(times)+1, 0-1, E])
ax.invert_yaxis() 
ax.set_xlabel('estimation time [min]', fontsize=16)
ax.set_ylabel('estimation settings', fontsize=16)
ax.set_title('total computation time ({} voxels)'.format(num_vox), fontsize=24)
fig.savefig('Simulation_C_Results_Fig_1.png', dpi=150)

# plot simulation results (NumpRF estimates)
for h, ind in enumerate([ind0, ind1, ind2]):
    fig, axs = plt.subplots(4, 6, figsize=(32,18))
    for i, axi in enumerate(axs):
        for j, axj in enumerate(axi):
            n = i*6+j
            axj.plot(Res[n]['mu_est'][ind], Res[n]['fwhm_est'][ind], '.'+cols[h])
            axj.axis([-0.5, 6.5, -0.5, 18.5])
            if i == 3 and j == 0:
                axj.set_xlabel('estimated preferred numerosity', fontsize=16)
                axj.set_ylabel('estimated tuning width', fontsize=16)
            axj.set_title(names[n], fontsize=24)
    fig.savefig('Simulation_C_Results_Fig_2'+'abc'[h]+'.png', dpi=150)

# plot simulation results (popeye estimates)
fig, axs = plt.subplots(3, 3, figsize=(32,18))
for s in range(len(methB)):
    for h, ind in enumerate([ind0, ind1, ind2]):
        axs[s,h].plot(Res[methB[s]]['mu_est'][ind], Res[methB[s]]['fwhm_est'][ind], '.'+cols[h])
        axs[s,h].axis([-0.5, 6.5, -0.5, 18.5])
        if s == 2 and h == 1:
            axs[s,h].set_xlabel('estimated preferred numerosity', fontsize=16)
            axs[s,h].set_ylabel('estimated tuning width', fontsize=16)
        if s == 0:
            axs[s,h].set_title(titles[h], fontsize=24)
        if h == 0:
            axs[s,h].set_ylabel(names[methB[s]], fontsize=20)
fig.savefig('Simulation_C_Results_Fig_2d.png', dpi=150)

# plot simulation results (parameters correlations)
fig, axs = plt.subplots(2, 2, figsize=(32,18))
for i, axi in enumerate(axs):
    for j, axj in enumerate(axi):
        if i == 0:
            if j == 0:
                axj.barh(np.arange(E), np.array(mu_corr), color='g')
                axj.set_yticks(np.arange(E), labels=names, fontsize=16)
                axj.set_ylabel('estimation settings', fontsize=24)
                axj.set_title('preferred numerosity', fontsize=24)
            else:
                axj.barh(np.arange(E), np.array(fwhm_corr), color='g')
                axj.set_title('tuning widths', fontsize=24)
            axj.axis([0, 1, 0-1, E])
            axj.set_xlabel('correlation coefficient', fontsize=24)
        else:
            if j == 0:
                axj.barh(np.arange(E), np.array(mu_MAE), color='g')
                axj.set_yticks(np.arange(E), labels=names, fontsize=16)
                axj.set_ylabel('estimation settings', fontsize=24)
                axj.axis([0, 1, 0-1, E])
            else:
                axj.barh(np.arange(E), np.array(fwhm_MAE), color='g')
                axj.axis([0, 3, 0-1, E])
            axj.set_xlabel('mean absolute error', fontsize=24)
        axj.invert_yaxis()
fig.savefig('Simulation_C_Results_Fig_3.png', dpi=150)

# plot simulation results (identification accuracies, AIC/R^2)
fig, axs = plt.subplots(1, 3, figsize=(32,18))
for i, ax in enumerate(axs):
    if i == 0:
        ax.barh(np.arange(E), np.array(TNR0_A), color='r', edgecolor='k')
        ax.set_yticks(np.arange(E), labels=names, fontsize=16)
        ax.set_xlabel('true negative rate (AIC/R^2)', fontsize=24)
        ax.set_ylabel('estimation settings', fontsize=24)
        ax.set_title('no-signal voxels', fontsize=24)
    elif i == 1:
        ax.barh(np.arange(E), np.array(TPR_A), color='g', edgecolor='k')
        ax.set_title('low-numerosity voxels', fontsize=24)
        ax.set_xlabel('true positive rate (AIC/R^2)', fontsize=24)
    elif i == 2:
        ax.barh(np.arange(E), np.array(TNR2_A), color='b', edgecolor='k')
        ax.set_title('high-numerosity voxels', fontsize=24)
        ax.set_xlabel('true negative rate (AIC/R^2)', fontsize=24)
    ax.axis([0, 1, 0-1, E])
    ax.invert_yaxis()
fig.savefig('Simulation_C_Results_Fig_4a.png', dpi=150)

# plot simulation results (identification accuracies, range)
fig, axs = plt.subplots(1, 3, figsize=(32,18))
for i, ax in enumerate(axs):
    if i == 0:
        ax.barh(np.arange(E), np.array(TNR0_B), color='r', edgecolor='k')
        ax.set_yticks(np.arange(E), labels=names, fontsize=16)
        ax.set_xlabel('true negative rate (range)', fontsize=24)
        ax.set_ylabel('estimation settings', fontsize=24)
        ax.set_title('no-signal voxels', fontsize=24)
    elif i == 1:
        ax.barh(np.arange(E), np.array(TPR_B), color='g', edgecolor='k')
        ax.set_title('low-numerosity voxels', fontsize=24)
        ax.set_xlabel('true positive rate (range)', fontsize=24)
    elif i == 2:
        ax.barh(np.arange(E), np.array(TNR2_B), color='b', edgecolor='k')
        ax.set_title('high-numerosity voxels', fontsize=24)
        ax.set_xlabel('true negative rate (range)', fontsize=24)
    ax.axis([0, 1, 0-1, E])
    ax.invert_yaxis()
fig.savefig('Simulation_C_Results_Fig_4b.png', dpi=150)