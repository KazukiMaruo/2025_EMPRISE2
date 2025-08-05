# -*- coding: utf-8 -*-
"""
Simulation study for EMPRISE fMRI data

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-09-04, 14:52: first version
2023-09-04, 16:27: fine-tuned parameters
2023-09-06, 11:55: controlled SNR
2023-09-06, 15:59: implemented saving
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
sub = 'EDY7'; ses = 'visual'

# get onsets and durations
sess           = EMPRISE.Session(sub, ses)
ons, dur, stim = sess.get_onsets()
ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
TR             = EMPRISE.TR

# get confound variables
labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
          'white_matter', 'csf', 'global_signal', \
          'cosine00', 'cosine01', 'cosine02']
X_c    = sess.get_confounds(labels)
X_c    = EMPRISE.standardize_confounds(X_c)

# set number of voxels
v0 = 500                        # voxels without numerosity-related signal
v1 = 1000                       # voxels with preferred numerosity 0.5 < mu < 5.5
v2 = 500                        # voxels with preferred numerosity mu ca. 20

# sample preferred numerosity
mu = np.concatenate((np.zeros(v0), 
                     np.random.uniform(0.5, 5.5, size=v1), 
                     np.random.normal(20, np.sqrt(5), size=v2)))
# Explanation: The simulation consists of
# - v0 voxels with no signal (to assess false-positive rate)
# - v1 voxels with low preferred numerosity (uniform between 0.5 and 5.5)
# - v2 voxels with high preferred numerosity (normal with mean 20 and variance 5)

# sample tuning width
m_xy = 1.49
n_xy = 1.06
fwhm = m_xy*mu + n_xy + np.random.normal(0, 1, mu.shape)
fwhm[fwhm<0] = m_xy*mu[fwhm<0] + n_xy
# Explanation: Figure 4B from Harvey et al. (2013) shows the linear
# relationship between mu and fwhm "for a representative subject". Assuming
# fwhm = m*mu + n, m and n were estimated from the data in this figure and
# used for sampling fwhm from mu, adding standard normal random noise.

# specify between-voxel variance
s2_k = 1
# Explanation: Each voxel will be simulated seperately and before that,
# regression coefficients will be sampled from normal distributions with
# variance 1 (see below).

# specify between-run variance
s2_j = s2_k/10
# Explanation: The between-run variance is assumed to be one magnitude smaller
# than the between-voxel variance. This means that regression coefficients vary
# considerably between voxels, but less so within one voxel, enforcing a
# reasonable level of stationarity across runs.

# specify autocorrelation
tau = 0.2
# Explanation: SPM estimates an AR(1) process with AR parameter p = 0.2. This
# corresponds to a correlation of 0.2 between directly adjacent fMRI scans.
# Sources:
# - https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=ind1607&L=SPM&P=R124872
# - https://www.jiscmail.ac.uk/cgi-bin/wa-jisc.exe?A2=ind1610&L=SPM&P=R67251

# set HRF parameters
hrf_mean = np.array([  6, 16,  1,  1,  6,  0, 32])
hrf_std  = np.array([  1,  2,0.1,0.1,0.5,  0,  0])
# Explanation: HRF parameters are sampled from N(hrf_mean, hrf_std). Thus,
# 99.7% of values will be within [hrf_mean-3*hrf_std, hrf_mean+3*hrf_std]:
# - delay of response will be approximately 3 < p < 9
# - delay of undershoot will be 10 < p < 22
# - dispersion of response will be 0.7 < p < 1.3
# - dispersion of undershoot will be 0.7 < p < 1.3
# - ratio of response to undershoot will be 4.5 < p < 7.5
# - onset of function will always be 0
# - length of kernel will always be 32
# Sources:
# - https://statproofbook.github.io/P/norm-probstd
# - https://github.com/SkeideLab/EMPRISE/blob/akieslinger/data/code/harveymodel.py#L549

# preallocate scaling parameters
mu_b = np.zeros(mu.shape)
mu_c = np.zeros(mu.shape)
s2_i = np.zeros(mu.shape)


### Step 2: perform simulations ###############################################

# prepare simulation
ds  = NumpRF.DataSet(0, ons, dur, stim, TR, X_c)
n   = X_c.shape[0]
p   = X_c.shape[1]+2
v   = v0+v1+v2
r   = len(EMPRISE.runs)

# preallocate simulanda
Y   = np.zeros((n,v,r))
S   = np.zeros((n,v,r))
B   = np.zeros((p,v,r))
SNR = np.zeros((r,v))
HRF = np.zeros((v,hrf_mean.size))

# for all voxels
print('\n-> Simulate {} voxels (s2_k = {}, s2_j = {}, tau = {}):'.
      format(v0+v1+v2, s2_k, s2_j, tau))
for i in range(v):
    
    # sample task-related scaling
    print('   - Voxel {}: '.format(i+1), end='')
    if mu[i] == 0:
        mu_b[i] = 0
        mu[i]   = 0.01
        fwhm[i] = 0.01
    else:
        mu_b[i] = np.random.normal(5, np.sqrt(s2_k))
    print('mu = {:.2f}, fwhm = {:.2f}, mu_b = {:.2f}, '.format(mu[i], fwhm[i], mu_b[i]), end='')
    # Explanation: If mu is 0, this is a noise voxel. In this case, set the
    # tuning parameters to very small and set the scaling parameter to 0 in
    # order to produce no numerosity-related signal. If mu is not 0, sample
    # the scaling parameter from N(5,1), i.e. approximately within [2, 8].
    
    # sample nuisance scaling
    mu_c[i] = np.random.normal(0, np.sqrt(s2_k))
    print('mu_c = {:.2f}, '.format(mu_c[i]), end='')
    # Explanation: Confound variables are added regardless of tuning parameters.
    # However, confounds are assumed to be zero across voxels. In each voxel, 
    # sample the scaling parameters from N(0,1), i.e. within [-3, +3].
    
    # sample hemodynamic parameters
    HRF[i,:] = np.random.normal(hrf_mean, hrf_std)
    hrf_str  = '['+', '.join(['{:.2f}'.format(x) for x in HRF[i,:]])+']'
    # Explanation: see detailed explanation of hemodynamic parameters above.
    
    # pre-simulate to calculate SNR
    y, s, x, b = ds.simulate(mu[[i]], fwhm[[i]], mu_b[i], mu_c[i], 0, 0, 0, tau, HRF[i,:])
    if mu[i] == 0.01:
        VXb = np.var(x[:,1:,0] @ b[1:,0,0])
    else:
        x[:,0,0] = s[:,0,0]
        VXb = np.var(x[:,:,0] @ b[:,0,0])
    
    # sample within-run variance term
    snr     = np.random.gamma(shape=1.5, scale=1)
    s2_i[i] = VXb/snr
    print('s2_i = {:.2f},'.format(s2_i[i]), end='\n')
    print('     HRF: {}; '.format(hrf_str), end='')
    # Explanation: The signal to noise-ratio is sampled from Gam(1.5,1), such
    # that mean(SNR) = 1.5 and var(SNR) = 1.5. The within-run variance is then
    # calculated as s2_i = Var(Xb)/SNR (because SNR = Var(Xb)/Var(e)).
    
    # sample simulated data
    Y[:,[i],:], S[:,[i],:], X, B[:,[i],:] = ds.simulate(
        mu[[i]], fwhm[[i]], mu_b[i], mu_c[i],
        0, s2_j, s2_i[i], tau, HRF[i,:])
    # Explanation: see NumpRF.DataSet.simulate; s2_k is set to 0, because
    # it was already applied above and each voxel is simulated separately.
    
    # calculate SNR
    for j in range(r):
        if mu[i] == 0.01:
            Y[:,i,j] = Y[:,i,j] - S[:,i,j] * B[0,i,j]
            SNR[j,i] = np.var(X[:,1:,j] @ B[1:,i,j])/s2_i[i]
        else:
            X[:,0,j] = S[:,i,j]
            SNR[j,i] = np.var(X[:,:,j] @ B[:,i,j])/s2_i[i]
    print('mean SNR = {:.2f}.'.format(np.mean(SNR[:,i])), end='\n')
    # Explanation: see above; SNR = Var(Xb)/Var(e).

# clean up simulation
del hrf_str, ds, y, s, x, b, VXb, snr
t = np.arange(0, EMPRISE.n*EMPRISE.TR, EMPRISE.TR)
print('\n-> Simulated SNRs:')
print('   - no-signal voxels: mean SNR = {:.2f};'.format(np.mean(SNR[:,:v0])))
print('   - low-numerosity voxels: mean SNR = {:.2f};'.format(np.mean(SNR[:,v0:v0+v1])))
print('   - high-numerosity voxels: mean SNR = {:.2f}.\n'.format(np.mean(SNR[:,v0+v1:v0+v1+v2])))


### Step 3: visualize simulations #############################################

# find voxels to plot
inds = []
inds.append([x for x in range(5)])
mus = np.arange(1,6,1)
inds.append([np.absolute(mu-x).argmin() for x in mus])
mus = np.arange(18,23,1)
inds.append([np.absolute(mu-x).argmin() for x in mus])
titles = ['no-signal voxels', 'low-numerosity voxels', 'high-numerosity voxels']

# plot simulated signals
for i, ind in enumerate(inds):
    fig = plt.figure(figsize=(32,18))
    axs = fig.subplots(len(ind),1)
    for j, ax in enumerate(axs):
        title  = ''; xlabel = ''; ylabel = 'signal [a.u.]'
        if j == 0: title = titles[i]
        if j == len(axs)-1: xlabel = 'time [s]'
        Y_ax = np.squeeze(Y[:,ind[j],:])
        ds = NumpRF.DataSet(Y_ax, ons, dur, stim, TR, X_c)
        ds.plot_signals_axis(ax)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=24)
        ax.text(np.mean(t), np.min(Y_ax)-(1/2)*(1/20)*(np.max(Y_ax)-np.min(Y_ax)), \
                'voxel: {}; mu = {:.2f}, fwhm = {:.2f}, SNR = {:.2f}'.
                format(ind[j]+1, mu[ind[j]], fwhm[ind[j]], np.mean(SNR,axis=0)[ind[j]]), \
                fontsize=16, horizontalalignment='center', verticalalignment='center')
    fig.show()
    filename = targ_dir+'Simulation_C_Fig_1'+['a','b','c'][i]+'.png'
    fig.savefig(filename, dpi=150)
    
# specify indices
inds = [range(0,v0), range(v0,v0+v1), range(v0+v1,v0+v1+v2)]

# plot tuning parameters
fig = plt.figure(figsize=(18,6))
axs = fig.subplots(1,len(inds))
for j, ax in enumerate(axs):
    ax.plot(mu[inds[j]], fwhm[inds[j]], 'o')
    if j == 0:
        ax.axis([-0.1, 1, -0.1, 1])
    elif j == 1:
        ax.axis([0, 6, 0, 12])
    elif j == 2:
        ax.axis([0, 25, 0, 40])
    ax.set_xlabel('true preferred numerosity', fontsize=16)
    ax.set_ylabel('true tuning width (FWHM)', fontsize=16)
    ax.set_title(titles[j], fontsize=24)
fig.show()
filename = targ_dir+'Simulation_C_Fig_2.png'
fig.savefig(filename, dpi=150)

# plot signal-to-noise ratios
fig = plt.figure(figsize=(18,6))
axs = fig.subplots(1,len(inds))
for j, ax in enumerate(axs):
    bins = np.arange(0, 10.25, 0.25)
    SNRs = np.mean(SNR, axis=0)[inds[j]]
    ax.hist(SNRs, bins)
    ax.set_xlabel('signal-to-noise ratio', fontsize=16)
    ax.set_ylabel('number of occurences', fontsize=16)
    ax.set_title(titles[j], fontsize=24)
fig.show()
filename = targ_dir+'Simulation_C_Fig_3.png'
fig.savefig(filename, dpi=150)

# save simulation data (Python)
settings = {'sub': sub, 'ses': ses, 'labels': labels, \
            'onsets': ons, 'durations': dur, 'stimuli': stim, \
            'mu_b': mu_b, 'mu_c': mu_c, 's2_k': s2_k, 's2_j': s2_j, 's2_i': s2_i, \
            'tau': tau, 'HRF': HRF}
filename = targ_dir+'Simulation_C.npz'
np.savez(filename, Y=Y, S=S, X=X, B=B, mu=mu, fwhm=fwhm, SNR=SNR)

# save simulation data (MATLAB)
filename = targ_dir+'Simulation_C.mat'
res_dict = {'Y': Y, 'S': S, 'X': X, 'B': B, \
            'mu': mu, 'fwhm': fwhm, 'SNR': SNR, \
            'settings': settings}
sp.io.savemat(filename, res_dict)