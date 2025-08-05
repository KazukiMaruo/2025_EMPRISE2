# -*- coding: utf-8 -*-
"""
Simulation study for EMPRISE fMRI data

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-07-03, 14:51: first version
2023-08-10, 14:45: generated new signals
"""


### Step 1: specify settings ##################################################

# import modules
import EMPRISE
import NumpRF
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# set number of voxels
v = 1000

# set target folder
targ_dir = r'C:/Users/sochj/ownCloud_MPI/MPI/EMPRISE/tools/Python/simulated_data/'

# set subject and session
sub = 'EDY7'
ses = 'visual'

# get onsets and durations
ons, dur, stim = EMPRISE.get_onsets(sub, ses)
ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
TR             = EMPRISE.TR

# get confound variables
labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
          'white_matter', 'csf', 'global_signal', \
          'cosine00', 'cosine01', 'cosine02']
X_c    = EMPRISE.get_confounds(sub, ses, labels)
for j in range(X_c.shape[2]):
    for k in range(X_c.shape[1]):
        X_c[:,k,j] = X_c[:,k,j] - np.mean(X_c[:,k,j])
        X_c[:,k,j] = X_c[:,k,j] / np.max(X_c[:,k,j])

# specify tuning parameters
mu   = np.random.uniform(0.5, 5.5, size=v)
fwhm = np.random.uniform(1, 10, size=v)

# specify scaling parameters
mu_b = 10                       # mean signal beta
mu_c = 5                        # mean confound beta
s2_k = 1                        # between-voxel variance
s2_j = 0.1                      # between-run variance
s2_i = 0.25                     # within-run variance
tau  = 0.1                      # time constant

# prepare time vector for visualization
t = np.arange(0, EMPRISE.n*EMPRISE.TR, EMPRISE.TR)


### Step 2: perform simulations ###############################################

# perform simulation
np.random.default_rng(seed=1)
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)

# find voxels to plot
mus = np.arange(1,6,1)
ind = np.zeros(mus.shape, dtype=int)
for j in range(len(mus)):
    ind[j] = np.absolute(mu-mus[j]).argmin()

# plot simulated signals
fig = plt.figure(figsize=(32,18))
axs = fig.subplots(len(ind),1)
for j, ax in enumerate(axs):
    title  = ''; xlabel = ''; ylabel = 'signal [a.u.]'
    if j == 0: title = 'EMPRISE BOLD signals'
    if j == len(axs)-1: xlabel = 'time [s]'
    Y_ax = np.squeeze(Y[:,ind[j],:])
    NumpRF.plot_signals_axis(ax, Y_ax, t, ons[0], dur[0], stim[0])
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=24)
    ax.text(np.mean(t), np.min(Y_ax)-(1/2)*(1/20)*(np.max(Y_ax)-np.min(Y_ax)), \
            'voxel: {}; mu = {:.2f}, fwhm = {:.2f}'.format(ind[j]+1, mu[ind[j]], fwhm[ind[j]]), \
            fontsize=16, horizontalalignment='center', verticalalignment='center')

# show/save simulation
fig.show()
filename = targ_dir+'Simulation_B.png'
fig.savefig(filename, dpi=150)
filename = targ_dir+'Simulation_B.mat'
res_dict = {'Y': Y, 'S': S, 'X': X, 'B': B, 'mu': mu, 'fwhm': fwhm, \
            'settings': {'sub': sub, 'ses': ses, 'labels': labels, \
                         'onsets': ons, 'durations': dur, 'stimuli': stim, \
                         'mu_b': mu_b, 'mu_c': mu_c, 's2_k': s2_k, 's2_j': s2_j, 's2_i': s2_i, 'tau': tau}}
sp.io.savemat(filename, res_dict)