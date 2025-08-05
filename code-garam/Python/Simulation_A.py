# -*- coding: utf-8 -*-
"""
Simulation study for EMPRISE fMRI data

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-06-29, 17:07: first version
2023-07-03, 10:27: adapted parameters
2023-08-17, 11:35: saved results
"""


### Step 1: specify settings ##################################################

# import modules
import EMPRISE
import NumpRF
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

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
mu   = np.array([1,3,5])
fwhm = np.array([1,1,1])

# specify scaling parameters (default scenario)
mu_b = 10                       # mean signal beta
mu_c = 1                        # mean confound beta
s2_k = 1                        # between-voxel variance
s2_j = 0.1                      # between-run variance
s2_i = 0.25                     # within-run variance
tau  = 0.1                      # time constant

# prepare time vector for visualization
t = np.arange(0, EMPRISE.n*EMPRISE.TR, EMPRISE.TR)


### Step 2: perform simulations ###############################################

# define function for plotting
def save_simulation(Y, t, ons, dur, stim, mu, fwhm, title, fname):
    fig = plt.figure(figsize=(32,18))
    NumpRF.plot_signals_figure(fig, Y, t, ons, dur, stim, mu, fwhm, \
                               avg=[False, False], title=title)
    # fig.savefig(fname, dpi=150)
    
# define function for collecting
def collect_results(Y_all, S_all, mu_all, fwhm_all, Y, S, mu, fwhm):
    Y_all    = np.concatenate((Y_all, Y), axis=1)
    S_all    = np.concatenate((S_all, S), axis=1)
    mu_all   = np.concatenate((mu_all, mu),     axis=0)
    fwhm_all = np.concatenate((fwhm_all, fwhm), axis=0)
    return Y_all, S_all, mu_all, fwhm_all

# Simulation 1: default scenario
#-----------------------------------------------------------------------------#
np.random.default_rng(seed=1)
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)
save_simulation(Y, t, ons[0], dur[0], stim[0], mu, fwhm, \
                'default scenario', 'Figure_outputs/Simulation_A_Sim_1.png')
Y_all, S_all, mu_all, fwhm_all = collect_results(np.zeros((Y.shape[0],0,Y.shape[2])), \
                                                 np.zeros((S.shape[0],0,S.shape[2])), \
                                                 np.zeros(0), np.zeros(0), \
                                                 Y, S, mu, fwhm)

# Simulation 2: different preferred numerosities
#-----------------------------------------------------------------------------#
mu = np.array([3,3,3])
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)
save_simulation(Y, t, ons[0], dur[0], stim[0], mu, fwhm, \
                'different preferred numerosities', 'Figure_outputs/Simulation_A_Sim_2.png')
Y_all, S_all, mu_all, fwhm_all = collect_results(Y_all, S_all, mu_all, fwhm_all, Y, S, mu, fwhm)

# Simulation 3: different tuning widths
#-----------------------------------------------------------------------------#
mu   = np.array([1,3,5])
fwhm = np.array([10,10,10])
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)
save_simulation(Y, t, ons[0], dur[0], stim[0], mu, fwhm, \
                'different tuning widths', 'Figure_outputs/Simulation_A_Sim_3.png')
Y_all, S_all, mu_all, fwhm_all = collect_results(Y_all, S_all, mu_all, fwhm_all, Y, S, mu, fwhm)

# Simulation 4: stronger signal
#-----------------------------------------------------------------------------#
fwhm = np.array([1,1,1])
mu_b = 50
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)
save_simulation(Y, t, ons[0], dur[0], stim[0], mu, fwhm, \
                'stronger signal', 'Figure_outputs/Simulation_A_Sim_4.png')
Y_all, S_all, mu_all, fwhm_all = collect_results(Y_all, S_all, mu_all, fwhm_all, Y, S, mu, fwhm)

# Simulation 5: stronger confounds
#-----------------------------------------------------------------------------#
mu_b = 10
mu_c = 5
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)
save_simulation(Y, t, ons[0], dur[0], stim[0], mu, fwhm, \
                'stronger confounds', 'Figure_outputs/Simulation_A_Sim_5.png')
Y_all, S_all, mu_all, fwhm_all = collect_results(Y_all, S_all, mu_all, fwhm_all, Y, S, mu, fwhm)

# Simulation 6: higher between-voxel variance
#-----------------------------------------------------------------------------#
mu_c = 1
s2_k = 4
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)
save_simulation(Y, t, ons[0], dur[0], stim[0], mu, fwhm, \
                'higher between-voxel variance', 'Figure_outputs/Simulation_A_Sim_6.png')
Y_all, S_all, mu_all, fwhm_all = collect_results(Y_all, S_all, mu_all, fwhm_all, Y, S, mu, fwhm)

# Simulation 7: higher between-run variance
#-----------------------------------------------------------------------------#
s2_k = 1
s2_j = 0.4
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)
save_simulation(Y, t, ons[0], dur[0], stim[0], mu, fwhm, \
                'higher between-run variance', 'Figure_outputs/Simulation_A_Sim_7.png')
Y_all, S_all, mu_all, fwhm_all = collect_results(Y_all, S_all, mu_all, fwhm_all, Y, S, mu, fwhm)

# Simulation 8: higher within-run variance
#-----------------------------------------------------------------------------#
s2_j = 1
s2_i = 1
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)
save_simulation(Y, t, ons[0], dur[0], stim[0], mu, fwhm, \
                'higher within-run variance', 'Figure_outputs/Simulation_A_Sim_8.png')
Y_all, S_all, mu_all, fwhm_all = collect_results(Y_all, S_all, mu_all, fwhm_all, Y, S, mu, fwhm)

# Simulation 9: higher serial correlations
#-----------------------------------------------------------------------------#
s2_i = 0.25
tau  = 0.9
Y, S, X, B = NumpRF.simulate(ons, dur, stim, TR, X_c, mu, fwhm, mu_b, mu_c, s2_k, s2_j, s2_i, tau)
save_simulation(Y, t, ons[0], dur[0], stim[0], mu, fwhm, \
                'higher serial correlations', 'Figure_outputs/Simulation_A_Sim_9.png')
Y_all, S_all, mu_all, fwhm_all = collect_results(Y_all, S_all, mu_all, fwhm_all, Y, S, mu, fwhm)


### Step 3: store simulations #################################################

# save simulation results
tau = 0.1
filename = targ_dir+'Simulation_A.mat'
res_dict = {'Y': Y_all, 'S': S_all, 'X': X, 'B': B, 'mu': mu_all, 'fwhm': fwhm_all, \
            'settings': {'sub': sub, 'ses': ses, 'labels': labels, \
                         'onsets': ons, 'durations': dur, 'stimuli': stim, \
                         'mu_b': mu_b, 'mu_c': mu_c, 's2_k': s2_k, 's2_j': s2_j, 's2_i': s2_i, 'tau': tau}}
sp.io.savemat(filename, res_dict)