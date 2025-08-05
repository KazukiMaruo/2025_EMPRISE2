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


### Simulation C: NumpRF, 07,21/09/2023 #######################################

# specify possible settings
avg1s  = [True, False]
avg2s  = [True, False]
corrs  = ['iid', 'ar1']
orders = [1, 2, 3]

# load simulation results
filename =r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\Simulation_C.mat'
file,ext = os.path.splitext(filename)
Sim      = sp.io.loadmat(filename)
sub      = Sim['settings']['sub'][0][0][0]
ses      = Sim['settings']['ses'][0][0][0]

# retrieve simulation parameters
sess           = EMPRISE.Session(sub, ses)
ons, dur, stim = sess.get_onsets()
ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
TR             = EMPRISE.TR
X_c            = Sim['X'][:,1:13,:]
mu_true        = np.squeeze(Sim['mu'])
fwhm_true      = np.squeeze(Sim['fwhm'])

# load simulated data
ind = np.arange(0,Sim['Y'].shape[1])
Y   = Sim['Y'][:,ind,:]

# iterate through settings
for avg1 in avg1s:
    for avg2 in avg2s:
        for noise in corrs:
            for hrfs in orders:
                
                # specify target file
                res_file = file+'_'+str(avg1)+'_'+str(avg2)+'_'+str(noise)+'_'+str(hrfs)+'.mat'
                print('\n-> ESTIMATION VARIANT: avg=[{}, {}], corr={}, order={}:'.\
                      format(avg1, avg2, noise, hrfs))
                
                # if file already exists, report this
                if os.path.isfile(res_file):
                    print('   - has already been run and saved to disk!\n')
                
                # if file does not exist, run analysis
                else:
                    print('   - has not been saved to disk and will now be run:')
                    
                    # initialize data set
                    ds = NumpRF.DataSet(Y, ons, dur, stim, TR, X_c)
                    
                    # estimate parameters
                    start_time = time.time()
                    mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                        ds.estimate_MLE(avg=[avg1,avg2], meth='fgs', corr=noise, order=hrfs)
                    k_est, k_null, k_const = ds.free_parameters(avg=[avg1,avg2], corr=noise, order=hrfs)
                    end_time   = time.time()
                    difference = end_time - start_time
                    
                    # save estimated parameters
                    res_dict = {'mu_est': mu_est, 'fwhm_est': fwhm_est, 'beta_est': beta_est, \
                                'MLL_est': MLL_est, 'MLL_null': MLL_null, 'MLL_const': MLL_const, \
                                'k_est': k_est, 'k_null': k_null, 'k_const': k_const, \
                                'corr_est': corr_est, 'time': difference}
                    sp.io.savemat(res_file, res_dict)