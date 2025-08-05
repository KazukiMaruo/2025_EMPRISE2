# -*- coding: utf-8 -*-
"""
Simulation study regarding the hemodynamic response function

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-08-24, 19:20: first version
2023-08-28, 10:15: minor changes
2023-08-28, 15:38: corrected derivatives
"""


### Step 0: specify settings ##################################################

# import modules
import random
import numpy as np
import matplotlib.pyplot as plt
import PySPMs as PySPM

# specify parameters
T    =  32
dt   =  0.1
std  = [6,16,1,1,6,0,32]
pars = ['delay of response [sec]', 'delay of undershoot [sec]', \
        'dispersion of response', 'dispersion of undershoot', \
        'ratio of response to undershoot', 'onset [sec]']


### Scenario 1: one-parameter changes #########################################

# specify parameter changes
par = [[4, 16, 1,   1,   6, 0,   T], \
       [8, 16, 1,   1,   6, 0,   T], \
       [6, 12, 1,   1,   6, 0,   T], \
       [6, 20, 1,   1,   6, 0,   T], \
       [6, 16, 0.5, 1,   6, 0,   T], \
       [6, 16, 2.0, 1,   6, 0,   T], \
       [6, 16, 1,   0.5, 6, 0,   T], \
       [6, 16, 1,   2.0, 6, 0,   T], \
       [6, 16, 1,   1,   3, 0,   T], \
       [6, 16, 1,   1,  12, 0,   T], \
       [6, 16, 1,   1,   6, 2.5, T], \
       [6, 16, 1,   1,   6, 5.0, T]]

# calculate HRFs
t       = np.arange(0, T, dt)
hrf_std = PySPM.spm_get_bf(dt, order=3)
hrf_par = np.array([PySPM.spm_hrf(dt, p) for p in par])

# fit derivatives to HRFs
hrf_fit = np.zeros(hrf_par.shape)
for i in range(len(par)):
    y = np.array([hrf_par[i,:]]).T
    X = hrf_std
    b_est, s2_est = PySPM.GLM(y, X).MLE()
    hrf_fit[i,:]  = (X @ b_est).squeeze()

# visualize HRFs
fig = plt.figure(figsize=(32,16))
axs = fig.subplots(3,int(len(par)/2))
for i, axi in enumerate(axs):
    for j, ax in enumerate(axi):
        if i == 0:
            ax.plot(t, hrf_std[:,0], '-b')
            if j == 0:
                ax.plot(t, hrf_std[:,1], '--b')
                ax.plot(t, hrf_std[:,2], ':b')
                ax.legend(['canonical HRF', 'first derivative', 'second deriative'], \
                          loc='upper right')
                ax.text(30, -0.2, 'p = '+str(std), fontsize=12, \
                        horizontalalignment='right', verticalalignment='center')
            ax.axis([0, T, -0.6, 1.2])
            ax.set_title(pars[j]+' = '+str(std[j]))
            
        else:
            n = j*2 + i-1
            ax.plot(t, hrf_std[:,0], '-b')
            ax.plot(t, hrf_par[n,:], '-r')
            ax.plot(t, hrf_fit[n,:], '-g')
            ax.axis([0, T, -0.4, 1.2])
            ax.set_title(pars[j]+' = '+str(par[n][j]))
            if i == len(axs)-1 and j == len(axi)-1:
                ax.legend(['canonical HRF', 'observed HRF', 'fitted HRF'], \
                          loc='upper right')
fig.savefig('Simulation_HRF_Fig_1.png', dpi=150)


### Scenario 2: many-parameter changes ########################################

# specify parameter changes
var = [[4, 6, 8], [12, 16, 20], [0.5, 1, 2], [0.5, 1, 2], [3, 6, 12]]

# calculate HRFs
par     = [[] for x in range(18)]
hrf_par = np.zeros((len(par),hrf_std.shape[0]))
for i in range(hrf_par.shape[0]):
    par[i] = std.copy()
    for j in range(len(var)):
        par[i][j] = random.choice(var[j])
    hrf_par[i,:] = PySPM.spm_hrf(dt, par[i])

# fit derivatives to HRFs
hrf_fit = np.zeros(hrf_par.shape)
for i in range(len(par)):
    y = np.array([hrf_par[i,:]]).T
    X = hrf_std
    b_est, s2_est = PySPM.GLM(y, X).MLE()
    hrf_fit[i,:]  = (X @ b_est).squeeze()

# visualize HRFs
fig = plt.figure(figsize=(32,16))
axs = fig.subplots(3,int(len(par)/3))
for i, axi in enumerate(axs):
    for j, ax in enumerate(axi):
        n = i*6 + j
        ax.plot(t, hrf_std[:,0], '-b')
        ax.plot(t, hrf_par[n,:], '-r')
        ax.plot(t, hrf_fit[n,:], '-g')
        ax.axis([0, T, -0.6, 1.2])
        ax.set_title('p = '+str(par[n]))
        if i == 0 and j == 0:
            ax.legend(['canonical HRF', 'observed HRF', 'fitted HRF'], \
                      loc='upper right')
fig.savefig('Simulation_HRF_Fig_2.png', dpi=150)