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


### Simulation C: NumpRF/BayespRF/popeye results, 11,12,18/09/2023 ############

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
                # names.append('avg=[{}, {}], corr={}, order={}:'.format(avg1, avg2, noise, hrfs))
                names.append(str(avg1)+' , '+str(avg2)+' , '+str(noise)+' , '+str(hrfs))
                times.append(np.squeeze(Sim['time'])/60)
                res_dict = {'mu_est':    np.squeeze(Sim['mu_est']), \
                            'fwhm_est':  np.squeeze(Sim['fwhm_est']), \
                            'MLL_est':   np.squeeze(Sim['MLL_est']), \
                            'MLL_null':  np.squeeze(Sim['MLL_null']), \
                            'MLL_const': np.squeeze(Sim['MLL_const']), \
                            'k_est':     Sim['k_est'][0,0], \
                            'k_null':    Sim['k_null'][0,0], \
                            'k_const':   Sim['k_const'][0,0]}
                Res.append(res_dict)

# load simulation results (BayespRF)
filenames   =[r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\spm12\sub-EDY7\ses-visual\model-pRF\PRF_Simulation_C_1_0.mat', \
              r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\spm12\sub-EDY7\ses-visual\model-pRF\PRF_Simulation_C_1_1.mat']
Res_Bp      =[sp.io.loadmat(filenames[0]), sp.io.loadmat(filenames[1])]
mu_est_Bp   = np.zeros((2,num_vox))
fwhm_est_Bp = np.zeros((2,num_vox))
PP_est_Bp   = np.zeros((2,num_vox))
for b in range(2):
    for i in range(num_vox):
        mu_log,sig_lat,beta_lat,transit,decay,epsilon = Res_Bp[b]['PRF']['Ep'][0,0][0,i][0,0]
        mu_est_Bp[b,i],fwhm_est_Bp[b,i] = NumpRF.log2lin(mu_log[0,0], np.exp(sig_lat[0,0]))
        mu_log,sig_lat,beta_lat,transit,decay,epsilon = Res_Bp[b]['PRF']['Pp'][0,0][0,i][0,0]
        PP_est_Bp[b,i] = beta_lat[0,0]
    if b == 0: names.append('BayespRF: w/o conf')
    if b == 1: names.append('BayespRF: with conf')
    times.append(Res_Bp[b]['PRF']['est_time'][0,0][0,0]/60)
del mu_log,sig_lat,beta_lat,transit,decay,epsilon

# load simulation results (popeye)
filename    = r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulation_results\Simulation_C_popeye.csv'
Res_pe      = pd.read_csv(filename)
mu_est_pe   = np.array(Res_pe['nums_est'])
fwhm_est_pe = np.array(Res_pe['fwhm_est'])
Rsq_est_pe  = np.array(Res_pe['rsq_est'])
names.append('popeye')
times.append(0)

# summarize simulation results
del Sim, res_file, res_dict, Res_Bp, Res_pe
E        = len(names)
ind0     = range(0,500)
ind1     = range(500,1500)
ind2     = range(1500,2000)
cols     = 'rgb'
titles   =['no-signal voxels', 'low-numerosity voxels', 'high-numerosity voxels']
times    = np.array(times)
mu_thr   = [0.5, 5.5]
fwhm_thr = [1, 15]
Rsq_thr  =  0.3
PP_thr   =  0.5

# identify voxel categories (NumpRF)
TPR10 = []
TPR12 = []
TNR01 = []
TNR21 = []
for R in Res:
    mu_est   = R['mu_est']
    fwhm_est = R['fwhm_est']
    AIC_est  = -2*R['MLL_est']  + 2*R['k_est']
    AIC_null = -2*R['MLL_null'] + 2*R['k_null']
    ind_mf   = np.logical_or(np.logical_or(mu_est<mu_thr[0], mu_est>mu_thr[1]),\
                             np.logical_or(fwhm_est<fwhm_thr[0], fwhm_est>fwhm_thr[1]))
    TPR10.append(np.sum(AIC_est[ind1]<AIC_null[ind1])/len(ind1))
    TNR01.append(np.sum(AIC_null[ind0]<AIC_est[ind0])/len(ind0))
    TPR12.append(np.sum(~ind_mf[ind1])/len(ind1))
    TNR21.append(np.sum(ind_mf[ind2])/len(ind2))
    # TPR12.append(np.sum(AIC_est[ind1]<AIC_null[ind1])/len(ind1))
    # TNR21.append(np.sum(AIC_null[ind2]<AIC_est[ind2])/len(ind2))

# identify voxel categories (BayespRF)
for b in range(2):
    ind_mf_Bp = np.logical_or(np.logical_or(mu_est_Bp[b,:]<mu_thr[0], mu_est_Bp[b,:]>mu_thr[1]),\
                              np.logical_or(fwhm_est_Bp[b,:]<fwhm_thr[0], fwhm_est_Bp[b,:]>fwhm_thr[1]))
    TPR10.append(np.sum(PP_est_Bp[b,ind1]>PP_thr)/len(ind1))
    TNR01.append(np.sum(PP_est_Bp[b,ind0]<PP_thr)/len(ind0))
    TPR12.append(np.sum(~ind_mf_Bp[ind1])/len(ind1))
    TNR21.append(np.sum(ind_mf_Bp[ind2])/len(ind2))
    # TPR12.append(np.sum(PP_est_Bp[b,ind1]>PP_thr)/len(ind1))
    # TNR21.append(np.sum(PP_est_Bp[b,ind2]<PP_thr)/len(ind2))

# identify voxel categories (popeye)
ind_mf_pe = np.logical_or(np.logical_or(mu_est_pe<mu_thr[0], mu_est_pe>mu_thr[1]),\
                          np.logical_or(fwhm_est_pe<fwhm_thr[0], fwhm_est_pe>fwhm_thr[1]))
TPR10.append(np.sum(Rsq_est_pe[ind1]>Rsq_thr)/len(ind1))
TNR01.append(np.sum(Rsq_est_pe[ind0]<Rsq_thr)/len(ind0))
TPR12.append(np.sum(~ind_mf_pe[ind1])/len(ind1))
TNR21.append(np.sum(ind_mf_pe[ind2])/len(ind2))
# TPR12.append(np.sum(Rsq_est_pe[ind1]>Rsq_thr)/len(ind1))
# TNR21.append(np.sum(Rsq_est_pe[ind2]<Rsq_thr)/len(ind2))

# calculate correlations (NumpRF)
mu_corr   = [np.corrcoef(mu_true[ind1], R['mu_est'][ind1])[0,1] for R in Res]
fwhm_corr = [np.corrcoef(fwhm_true[ind1], R['fwhm_est'][ind1])[0,1] for R in Res]
mu_MAE    = [np.mean(np.abs(R['mu_est'][ind1]-mu_true[ind1])) for R in Res]
fwhm_MAE  = [np.mean(np.abs(R['fwhm_est'][ind1]-fwhm_true[ind1])) for R in Res]

# calculate correlations (BayespRF)
mu_corr.extend([np.corrcoef(mu_true[ind1], x[ind1])[0,1] for x in mu_est_Bp])
fwhm_corr.extend([np.corrcoef(fwhm_true[ind1], x[ind1])[0,1] for x in fwhm_est_Bp])
mu_MAE.extend([np.mean(np.abs(x[ind1]-mu_true[ind1])) for x in mu_est_Bp])
fwhm_MAE.extend([np.mean(np.abs(x[ind1]-fwhm_true[ind1])) for x in fwhm_est_Bp])

# calculate correlations (popeye)
mu_corr.append(np.corrcoef(mu_true[ind1], mu_est_pe[ind1])[0,1])
fwhm_corr.append(np.corrcoef(fwhm_true[ind1], fwhm_est_pe[ind1])[0,1])
mu_MAE.append(np.mean(np.abs(mu_est_pe[ind1]-mu_true[ind1])))
fwhm_MAE.append(np.mean(np.abs(fwhm_est_pe[ind1]-fwhm_true[ind1])))

# plot simulation results (computation time, all pipelines)
fig, ax = plt.subplots(figsize=(16,9))
ax.barh(np.arange(E), times, color='w', edgecolor='k')
ax.set_yticks(np.arange(E), labels=names, fontsize=12)
ax.axis([0, np.max(times)+1, 0-1, E])
ax.invert_yaxis() 
ax.set_xlabel('estimation time [min]', fontsize=16)
ax.set_ylabel('estimation settings', fontsize=16)
ax.set_title('total computation time ({} voxels)'.format(num_vox), fontsize=24)
ax.text(0, E-1, ' unknown', fontsize=12, horizontalalignment='left', verticalalignment='center')
# fig.savefig('Simulation_C_Results_Fig_1a.png', dpi=150)

# plot simulation results (computation time, NumpRF only)
fig, ax = plt.subplots(figsize=(16,9))
ax.barh(np.arange(E-3), times[:-3], color='w', edgecolor='k')
ax.set_yticks(np.arange(E-3), labels=names[:-3], fontsize=12)
ax.axis([0, np.max(times[:-3])+1, 0-1, E-3])
ax.invert_yaxis() 
ax.set_xlabel('estimation time [min]', fontsize=16)
ax.set_ylabel('estimation settings', fontsize=16)
ax.set_title('total computation time ({} voxels)'.format(num_vox), fontsize=24)
# fig.savefig('Simulation_C_Results_Fig_1b.png', dpi=150)

# plot simulation results (popeye estimates)
fig, axs = plt.subplots(1, 3, figsize=(16,9))
for h, ind in enumerate([ind0, ind1, ind2]):
    axs[h].plot(mu_est_pe[ind], fwhm_est_pe[ind], '.'+cols[h])
    if h == 0: axs[h].axis([-0.5, 6.5, -0.5, 20.5])
    if h == 1: axs[h].axis([-0.5, 6.5, -0.5, 20.5])
    if h == 2: axs[h].axis([-0.5, 20.5, -0.5, 20.5])
    axs[h].set_xlabel('estimated preferred numerosity', fontsize=12)
    if h == 0: axs[h].set_ylabel('estimated tuning width', fontsize=12)
    axs[h].set_title(titles[h]+' (popeye)', fontsize=16)
# fig.savefig('Simulation_C_Results_Fig_2a.png', dpi=150)
    
# plot simulation results (BayespRF estimates)
fig, axs = plt.subplots(2, 3, figsize=(24,13.5))
for b in range(2):
    for h, ind in enumerate([ind0, ind1, ind2]):
        axs[b,h].plot(mu_est_Bp[b,ind], fwhm_est_Bp[b,ind], '.'+cols[h])
        if h == 0:
            axs[b,h].axis([-0.5, 11.5, -0.5, 7.5])
        if h == 1:
            if b == 0: axs[b,h].axis([-0.5, 8.5, -0.5, 50.5])
            if b == 1: axs[b,h].axis([-0.5, 8.5, -0.5, 20.5])
        if h == 2:
            if b == 0: axs[b,h].axis([-0.5, 25.5, -0.5, 0.5])
            if b == 1: axs[b,h].axis([-0.5, 25.5, -0.5, 40.5])
        axs[b,h].set_xlabel('estimated preferred numerosity', fontsize=12)
        if h == 0: axs[b,h].set_ylabel('estimated tuning width', fontsize=12)
        axs[b,h].set_title(titles[h]+' ('+names[24+b]+')', fontsize=20)
# fig.savefig('Simulation_C_Results_Fig_2b.png', dpi=150)

# plot simulation results (NumpRF estimates)
for h, ind in enumerate([ind0, ind1, ind2]):
    fig, axs = plt.subplots(4, 6, figsize=(32,18))
    for i, axi in enumerate(axs):
        for j, axj in enumerate(axi):
            n = i*6+j
            axj.plot(Res[n]['mu_est'][ind], Res[n]['fwhm_est'][ind], '.'+cols[h])
            axj.axis([-0.5, 6.5, -0.5, 20.5])
            if i == 3 and j == 0:
                axj.set_xlabel('estimated preferred numerosity', fontsize=16)
                axj.set_ylabel('estimated tuning width', fontsize=16)
            axj.set_title(names[n], fontsize=24)
    # fig.savefig('Simulation_C_Results_Fig_3'+'abc'[h]+'.png', dpi=150)

# plot simulation results (popeye correlations)
fig, axs = plt.subplots(1, 2, figsize=(16,9))
axs[0].plot(mu_true[ind1], mu_est_pe[ind1], '.g')
axs[0].axis([0, 6, 0, 6])
axs[0].set_xlabel('true value', fontsize=12)
axs[0].set_ylabel('estimated value', fontsize=12)
axs[0].set_title('preferred numerosity (popeye)', fontsize=16)
axs[0].text(6, (1/20)*6, 'r = {:.4f}, MAE = {:.2f}   '.format(mu_corr[-1], mu_MAE[-1]), \
            fontsize=12, horizontalalignment='right', verticalalignment='center')
axs[1].plot(fwhm_true[ind1], fwhm_est_pe[ind1], '.g')
axs[1].axis([0, 20, 0, 20])
axs[1].set_xlabel('true value', fontsize=12)
axs[1].set_ylabel('estimated value', fontsize=12)
axs[1].set_title('tuning width (popeye)', fontsize=16)
axs[1].text(20, 1, 'r = {:.4f}, MAE = {:.2f}   '.format(fwhm_corr[-1], fwhm_MAE[-1]), \
            fontsize=12, horizontalalignment='right', verticalalignment='center')
# fig.savefig('Simulation_C_Results_Fig_4a.png', dpi=150)

# plot simulation results (BayespRF correlations)
fig, axs = plt.subplots(2, 2, figsize=(24,13.5))
for b in range(2):
    axs[b,0].plot(mu_true[ind1], mu_est_Bp[b,ind1], '.g')
    axs[b,0].axis([0, 6, 0, 6])
    axs[b,0].set_xlabel('true value', fontsize=12)
    axs[b,0].set_ylabel('estimated value', fontsize=12)
    axs[b,0].set_title('preferred numerosity ('+names[24+b]+')', fontsize=20)
    axs[b,0].text(6, (1/20)*6, 'r = {:.4f}, MAE = {:.2f}   '.format(mu_corr[24+b], mu_MAE[24+b]), \
                  fontsize=16, horizontalalignment='right', verticalalignment='center')
    axs[b,1].plot(fwhm_true[ind1], fwhm_est_Bp[b,ind1], '.g')
    axs[b,1].axis([0, 20, 0, 20])
    axs[b,1].set_xlabel('true value', fontsize=12)
    axs[b,1].set_ylabel('estimated value', fontsize=12)
    axs[b,1].set_title('tuning width ('+names[24+b]+')', fontsize=20)
    axs[b,1].text(20, 1, 'r = {:.4f}, MAE = {:.2f}   '.format(fwhm_corr[24+b], fwhm_MAE[24+b]), \
                  fontsize=16, horizontalalignment='right', verticalalignment='center')
# fig.savefig('Simulation_C_Results_Fig_4b.png', dpi=150)

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
# fig.savefig('Simulation_C_Results_Fig_5.png', dpi=150)

# plot simulation results (identifiction accuracies)
fig, axs = plt.subplots(2, 2, figsize=(32,18))
for i, axi in enumerate(axs):
    for j, axj in enumerate(axi):
        if i == 0:
            if j == 0:
                axj.barh(np.arange(E), np.array(TPR10), color='y', edgecolor='k')
                axj.set_yticks(np.arange(E), labels=names, fontsize=16)
                axj.set_ylabel('estimation settings', fontsize=24)
                axj.set_title('low-numerosity vs. no-signal voxels', fontsize=24)
            else:
                axj.barh(np.arange(E), np.array(TPR12), color='c', edgecolor='k')
                axj.set_title('low-numerosity vs. high-numerosity voxels', fontsize=24)
            axj.axis([0, 1, 0-1, E])
            axj.set_xlabel('true positive rate', fontsize=24)
        else:
            if j == 0:
                axj.barh(np.arange(E), np.array(TNR01), color='y', edgecolor='k')
                axj.set_yticks(np.arange(E), labels=names, fontsize=16)
                axj.set_ylabel('estimation settings', fontsize=24)
            else:
                axj.barh(np.arange(E), np.array(TNR21), color='c', edgecolor='k')
            axj.axis([0, 1, 0-1, E])
            axj.set_xlabel('true negative rate', fontsize=24)
        axj.invert_yaxis()
# fig.savefig('Simulation_C_Results_Fig_6.png', dpi=150)