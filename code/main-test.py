# install library
#-------------------------------------------------------------------------#
import os
import numpy as np
import pandas as pd
import time
import scipy as sp
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Parameters
#-------------------------------------------------------------------------#
"""
Modify here only for your analysis
condition: digit, spoken, visual, audio1, audio2
code_dir: specify the directory containing "config.json" file.
"""
subject  = "N001"
session  = "visual"
model    = "VolumetricFWHM3" # model name.sh
target   = "all" # "all" or "split"
code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"


# Install functions
#-------------------------------------------------------------------------#
import EMPRISE
import NumpRF


# start Session class
#-------------------------------------------------------------------------#
sess = EMPRISE.Session(subject, session)
print(sess.get_anat_nii()) # example usage


# start Model class
#-------------------------------------------------------------------------#
mod = EMPRISE.Model(subject, session, model, space_id='T1w')
print(mod.get_model_dir()) # example usage

covs = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
        'white_matter', 'csf', 'global_signal', \
        'cosine00', 'cosine01', 'cosine02']
runs  =[1,2,3,4,5,6,7,8,9,10]

# specify thresholding parameters
#-----------------------------------------------------------------------------#
dAIC_thr  = 0                   # AIC diff must be larger than this
dBIC_thr  = 0                   # BIC diff must be larger than this
Rsq_def   = 0.3                 # R-squared must be larger than this
alpha_def = 0.05                # p-value must be smaller than this
mu_thr    =[1, 9]               # numerosity must be inside this range (EMPRISE-[1,5],EMPRISE2-[1,9])
fwhm_thr  =[0, 45]              # tuning width must be inside this range (EMPRISE-[0,24], EMPRISE2-[0,45])
beta_thr  =[0, np.inf]          # scaling parameter must be inside this range
crit_def  = 'Rsqmb'             # default thresholding option (see "threshold_maps")

##########################################################################
##########################################################################
##########################################################################



folds=['all']
stim = False

# part 1: prepare calculations
#---------------------------------------------------------------------#
print('\n\n-> Subject "{}", Session "{}", Model "{}":'.format(mod.sub, mod.ses, mod.model))
mod_dir = mod.get_model_dir()

# specify slices
i = -1                              # slice index (3rd dim)
slices = {'all': [], 'odd': [], 'even': []}
X_c, v_run = mod.get_confounds(covs)

for idx, run in enumerate(v_run):                    # for all possible runs
        filename = mod.get_confounds_tsv(run)
        if os.path.isfile(filename):    # if data from this run exist
                i = i + 1                   # increase slice index
                slices['all'].append(i)
                if idx % 2 == 1: slices['odd'].append(idx)
                else:            slices['even'].append(idx)

# part 2: analyze both hemispheres
#---------------------------------------------------------------------#
maps  = {}
for fold in folds: maps[fold] = {}

# for all folds
for fold in folds:
        
        # load analysis results
        filepath = mod_dir + '/sub-' + mod.sub + '_ses-' + mod.ses + \
                                '_model-' + mod.model + '_space-' + mod.space + '_'
        if fold in ['odd', 'even']:
                filepath = filepath + 'runs-' + fold + '_'
        res_file = filepath + 'numprf.mat'
        NpRF     = sp.io.loadmat(res_file)



# model.analyze_numerosity_volumetric(params)
avg=[True, False]
corr='iid'
order=1
ver='V2'
stim = False
sh=False 

# part 1: load subject data
#---------------------------------------------------------------------#
print('\n\n-> Subject "{}", Session "{}":'.format(mod.sub, mod.ses))
mod_dir = mod.get_model_dir()
if not os.path.isdir(mod_dir): os.makedirs(mod_dir)

# load confounds
print('   - Loading confounds ... ', end='')
X_c, v_run = mod.get_confounds(covs)

# remove the runs wiht high FD
if mod.ses ==  'digit':
        if mod.sub == 'N012':
                del v_run[1]
                X_c = np.delete(X_c, 1, axis=2)
elif mod.ses == 'visual':
        if mod.sub == 'N009':
                for idx in sorted([3, 6], reverse=True):
                        del v_run[idx]
                        X_c = np.delete(X_c, idx, axis=2)
        elif mod.sub == 'N012':
                del v_run[5]
                X_c = np.delete(X_c, 5, axis=2)

X_c = standardize_confounds(X_c)
print(f'\n    - Valid runs:{v_run}')
print('successful!')

# load onsets
print('   - Loading onsets ... ', end='')
ons, dur, stim = mod.get_onsets(v_run)
if not stim:
        ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
else:    
        if mod.ses in ['visual','digit','spoken']:
                ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
        elif mod.ses == 'audio1': 
                ons, dur, stim = onsets_trials2trials(ons, dur, stim, mode = True)
        elif mod.ses == 'audio2':
                ons, dur, stim = onsets_trials2trials(ons, dur, stim, mode = False)
print('successful!')


# specify grids
if ver == 'V0':
        mu_grid   = [ 3.0, 1.0]
        fwhm_grid = [10.1, 5.0]  
elif ver == 'V1':
        mu_grid   = np.concatenate((np.arange(0.05, 6.05, 0.05), \
                                10*np.power(2, np.arange(0,8))))
        fwhm_grid = np.concatenate((np.arange(0.3, 18.3, 0.3), \
                                24*np.power(2, np.arange(0,4))))
elif ver == 'V2' or ver == 'V2-lin':
        mu_grid   = np.concatenate((np.arange(0.80, 9.25, 0.05), \
                                np.array([20]))) # EMPRISE np.arange(0.80, 5.25, 0.05) 
        sig_grid  = np.arange(0.05, 5.05, 0.05) # EMPRISE np.arange(0.05, 3.05, 0.05)
else:
        err_msg = 'Unknown version ID: "{}". Version must be "V0" or "V1" or "V2"/"V2-lin".'
        raise ValueError(err_msg.format(ver))

# specify folds
if not sh: folds = {'all': []}        # all runs vs. split-half
else:      folds = {'odd': [], 'even': []}
for idx, run in enumerate(v_run):     # for all possible runs
        filename = mod.get_confounds_tsv(run)
        if os.path.isfile(filename):      # if data from this run exist
                if not sh:
                        folds['all'].append(idx)  # add slice to all runs
                else:
                        if idx % 2 == 1:          # add slice to odd runs
                                folds['odd'].append(idx)
                        else:                     # add slice to even runs
                                folds['even'].append(idx)

# part 2: analyze one whole brain
#---------------------------------------------------------------------#
results = {}
                
# load data
print('\n-> Space "{}":'.format(mod.space))
print('   - Loading fMRI data ... ', end='')

# fMRI data = scans * voxels * runs
Y     = mod.load_data_all(v_run, mod.space)

# Spatial smoothing

# Load binary brain mask
M     = mod.get_volumetric_mask(mod.space, v_run)

# Mask out non-brain voxels
Y     = Y[:,M,:]                
Y     = standardize_signals(Y)
V     = M.size
print('successful!')
print('\n-> Compare the temporal dimension.')

t_dif = Y.shape[0] - X_c.shape[0]
if t_dif == 0:
        print('   - Dimension matched.')
elif t_dif == -1:
        X_c = X_c[1:,:,:] # cut the t=1 confound since we dropped the first scan
        print('   - Cut the first time point confound.')
else:
        print('   -Temporal dimension of BOLD and confounds are not matched: \n')
        print(f'   Bold: {Y.shape[0]} \n Cov: {X_c.shape[0]}')



# function: threshold tuning maps
#-------------------------------------------------------------------------#
# threshold_maps(mod,
crit='Rsqmb,p=0.05B'
cv =False

"""
Threshold Numerosity, FWHM and Scaling Maps based on Criterion
maps = mod.threshold_maps(crit)

        crit - string; criteria used for thresholding maps
                        (default: "Rsqmb"; see below for details)
        cv   - bool; indicating whether cross-validated R-squared is used
        
        maps - dict of dicts; thresholded tuning maps
        o mu   - dict of strings; estimated numerosity maps
        o fwhm - dict of strings; FWHM tuning widths maps
        o beta - dict of strings; scaling parameter maps
        o Rsq  - dict of strings; variance explained maps
        o left  - thresholded parameter map for left hemisphere
        o right - thresholded parameter map for right hemisphere
        
maps = mod.threshold_maps(crit) loads tuning parameter maps from
analysis mod and thresholds these maps according to criteria crit.

The input parameter "crit" is a string that can contain the following:
o "AIC": numerosity model better than no-numerosity model according to AIC
o "BIC": numerosity model better than no-numerosity model according to BIC
o "Rsq": variance explained of numerosity model larger than specified value
o "m"  : value of numerosity estimate within specified range
o "f"  : value of tuning width estimate within specified range
o "b"  : value of scaling parameter estimate within specified range
o ","  : preceeding user-defined R^2 threshold (e.g. "Rsqmb,0.25")
o "p=" : specifying user-defined signifance level (e.g. "Rsq,p=0.05")
o "BHS": specifying multiple comparison correction (e.g. "Rsq,p=0.05B")

Note: "threshold_maps" uses the results dictionary keys "left" and "right"
which are identical to the hemisphere labels used by surfplot.
"""

# part 1: prepare thresholding
#---------------------------------------------------------------------#
print('\n\n-> Subject "{}", Session "{}", Model "{}":'.format(model.sub, model.ses, model.model))
mod_dir = model.get_model_dir()

X_c, v_run = model.get_confounds(covs)
# get runs and scans
r0, n0 = model.calc_runs_scans()
n1     = r0*n0          # effective number of observations in model
print(f'Effective number of observations in model: {n1}')
# number of explanatory variables used for R^2, 
p1     = [4,2][int(cv)] # 4 = B0, B1, mu, fwhm; 2 = B0, B1, mu and fwhm is estimatd from the other runs

# extract thresholds
Rsq_thr = Rsq_def
if ',' in crit:
        if 'p=' in crit:
                if crit[-1] in ['B', 'S', 'H']:
                        alpha = float(crit[(crit.find('p=')+2):-1])
                        meth  = crit[-1]
                else:
                        alpha = float(crit[(crit.find('p=')+2):])
                        meth  = ''
        else:
                Rsq_thr = float(crit.split(',')[1])


# part 2: threshold both hemispheres
#---------------------------------------------------------------------#
# hemis = {'L': 'left', 'R': 'right'}
maps  = {'mu': {}, 'fwhm': {}, 'beta': {}, 'Rsq': {}}
if cv: maps['cvRsq'] = maps.pop('Rsq')
# for hemi in hemis.keys():
        
# load numerosity map
filepath = mod_dir + '/sub-' + model.sub + '_ses-' + model.ses + \
                        '_model-' + model.model + '_space-' + model.space + '_'
res_file = filepath + 'numprf.mat'

# mu_map   = filepath + 'mu.surf.gii'
NpRF     = sp.io.loadmat(res_file)
# surface  = nib.load(mu_map)
# mask     = surface.darrays[0].data != 0

# load estimation results
mu    = np.squeeze(NpRF['mu_est'])
fwhm  = np.squeeze(NpRF['fwhm_est'])
beta  = np.squeeze(NpRF['beta_est'])

# brain mask in volumetric space
M       = model.get_volumetric_mask(model.space, v_run)
M_subco = model.get_volumetric_subcortmask(model.space, v_run)

# convert all parameters into the brain (x, y, z)
ref_path  = model.get_bold_nii(v_run[0], model.space) # Load your one functional image for reference
ref_img   = nib.load(ref_path)
shape     = ref_img.shape[:3]
n_voxels  = np.prod(shape)

# empty brain
mu_data   = np.zeros(n_voxels, dtype=mu.dtype)
fwhm_data = np.zeros(n_voxels, dtype=fwhm.dtype)
beta_data = np.zeros(n_voxels, dtype=beta.dtype)

# estimated parameters only in empty brain
mu_data[M]   = mu.flatten()
fwhm_data[M] = fwhm.flatten()
beta_data[M] = beta.flatten()

mu_subco = mu_data[M_subco]     # mu in subcortical
fwhm_subco = fwhm_data[M_subco] # fwhm in subcortical
beta_subco = beta_data[M_subco] # beta in subcortical

# if CV, load cross-validated R^2
if cv:
        Rsq_map = filepath + 'cvRsq.surf.gii'
        cvRsq   = nib.load(Rsq_map).darrays[0].data
        Rsq     = cvRsq[mask]

# otherwise, calculate total R^2
else:
        MLL1  = np.squeeze(NpRF['MLL_est'])
        MLL0  = np.squeeze(NpRF['MLL_null'])
        MLL00 = np.squeeze(NpRF['MLL_const'])

        MLL1_data  = np.zeros(n_voxels, dtype=MLL1.dtype)
        MLL0_data  = np.zeros(n_voxels, dtype=MLL0.dtype)
        MLL00_data = np.zeros(n_voxels, dtype=MLL00.dtype)

        # estimated parameters only in empty brain
        MLL1_data[M]  = MLL1.flatten()
        MLL0_data[M]  = MLL0.flatten()
        MLL00_data[M] = MLL00.flatten()

        MLL1_subco  = MLL1_data[M_subco]     
        MLL0_subco  = MLL0_data[M_subco] 
        MLL00_subco = MLL00_data[M_subco] 



        k1    = NpRF['k_est'][0,0]
        k0    = NpRF['k_null'][0,0]
        Rsq   = NumpRF.MLL2Rsq(MLL1_subco, MLL00_subco, n1)
        # See: https://statproofbook.github.io/P/rsq-mll
        dAIC  = (-2*MLL0_subco + 2*k0) - (-2*MLL1_subco + 2*k1)
        # See: https://statproofbook.github.io/P/mlr-aic
        dBIC  = (-2*MLL0_subco + k0*np.log(n1)) - (-2*MLL1_subco + k1*np.log(n1))
        # See: https://statproofbook.github.io/P/mlr-bic

# compute quantities for thresholding
print('     - Applying threshold criteria "{}" ... '.format(crit), end='')
ind_m = np.logical_or(mu_subco<mu_thr[0], mu_subco>mu_thr[1])
ind_f = np.logical_or(fwhm_subco<fwhm_thr[0], fwhm_subco>fwhm_thr[1])
ind_b = np.logical_or(beta_subco<beta_thr[0], beta_subco>beta_thr[1])

# apply conditions for exclusion
ind = mu_subco > np.inf
if 'AIC' in crit:
        ind = np.logical_or(ind, dAIC<dAIC_thr)
if 'BIC' in crit:
        ind = np.logical_or(ind, dBIC<dBIC_thr)
if 'Rsq' in crit:
        if not 'p=' in crit:
                ind = np.logical_or(ind, Rsq < Rsq_thr)
        else:
                ind = np.logical_or(ind, ~NumpRF.Rsqsig(Rsq, n1, p1, alpha, meth))
if 'm' in crit:
        ind = np.logical_or(ind, ind_m)
if 'f' in crit:
        ind = np.logical_or(ind, ind_f)
if 'b' in crit:
        ind = np.logical_or(ind, ind_b)
print('successful!')


# threshold tuning maps
para_est = {'mu': mu_subco, 'fwhm': fwhm_subco, 'beta': beta_subco, 'Rsq': Rsq}
if cv: para_est['cvRsq'] = para_est.pop('Rsq')
for name in para_est.keys():
        print('     - Saving thresholded "{}" image ... '.format(name), end='')
        para_map          = np.zeros(n_voxels, dtype=np.float32)
        para_thr          = para_est[name].copy()
        para_thr[ind]     = np.nan
        # para_map[M_subco] = para_thr
        filename          = filepath + name + '_thr-' + crit + '.nii.gz'
        para_img          = save_vol_mask(para_thr, ref_img, filename, M_subco)
        maps[name]        = filename
        print('successful!')
del para_est, para_map, para_thr,filename, para_img

# return results filename
return maps


# function: save single volume image (3D)
#-----------------------------------------------------------------------------#
def save_vol_mask(data, img, fname, mask):
    """
    Save Volume with Masked Data
    
    data  - 1D array (length = number of valid voxels)
    img   - Nifti1Image template (for shape, affine, header)
    fname - Output filename
    mask  - 1D boolean array or 3D boolean array; same shape as img
    
    Returns: new Nifti1Image
    """
    # Flatten mask if it's 3D
    if mask.ndim == 3:
        mask = mask.flatten(order='C')

    # Create empty volume
    spatial_shape = img.shape[:3]
    n_voxels = np.prod(spatial_shape)
    full_data = np.zeros(n_voxels, dtype=data.dtype)

    # Fill in only the masked voxels
    full_data[mask] = data.flatten()

    # create and save image
    data_map = full_data.reshape(spatial_shape, order='C')
    data_img = nib.Nifti1Image(data_map, img.affine, img.header)
    nib.save(data_img, fname)
    
    # load and return image
    data_img = nib.load(fname)
    return data_img


##########################################################################
##########################################################################
##########################################################################


# Parameters used for function input
ver='V2'
sh=False
        
# part 1: load subject data
#---------------------------------------------------------------------#
print('\n\n-> Subject "{}", Session "{}":'.format(model.sub, model.ses))
mod_dir = model.get_model_dir()
if not os.path.isdir(mod_dir): os.makedirs(mod_dir)

# load confounds
print('   - Loading confounds ... ', end='')
X_c, v_run = model.get_confounds(covs)

# remove the runs wiht high FD
if model.ses ==  'digit':
        if model.sub == 'N012':
                del v_run[1]
                X_c = np.delete(X_c, 1, axis=2)
elif model.ses == 'visual':
        if model.sub == 'N009':
                for idx in sorted([3, 6], reverse=True):
                        del v_run[idx]
                        X_c = np.delete(X_c, idx, axis=2)
        elif model.sub == 'N012':
                del v_run[5]
                X_c = np.delete(X_c, 5, axis=2)

X_c = standardize_confounds(X_c)
print(f'\n    - Valid runs:{v_run}')
print('successful!')

# load onsets
print('   - Loading onsets ... ', end='')
ons, dur, stim = model.get_onsets(v_run)
if not stim:
        ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
else:    
        if model.ses in ['visual','digit','spoken']:
                ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
        elif model.ses == 'audio1': 
                ons, dur, stim = onsets_trials2trials(ons, dur, stim, mode = True)
        elif model.ses == 'audio2':
                ons, dur, stim = onsets_trials2trials(ons, dur, stim, mode = False)
print('successful!')



# specify grids
if ver == 'V0':
        mu_grid   = [ 3.0, 1.0]
        fwhm_grid = [10.1, 5.0]  
elif ver == 'V1':
        mu_grid   = np.concatenate((np.arange(0.05, 6.05, 0.05), \
                                10*np.power(2, np.arange(0,8))))
        fwhm_grid = np.concatenate((np.arange(0.3, 18.3, 0.3), \
                                24*np.power(2, np.arange(0,4))))
elif ver == 'V2' or ver == 'V2-lin':
        mu_grid   = np.concatenate((np.arange(0.80, 9.25, 0.05), \
                                np.array([20]))) # EMPRISE np.arange(0.80, 5.25, 0.05) 
        sig_grid  = np.arange(0.05, 5.05, 0.05) # EMPRISE np.arange(0.05, 3.05, 0.05)
else:
        err_msg = 'Unknown version ID: "{}". Version must be "V0" or "V1" or "V2"/"V2-lin".'
        raise ValueError(err_msg.format(ver))

# specify folds
if not sh: folds = {'all': []}        # all runs vs. split-half
else:      folds = {'odd': [], 'even': []}
for idx, run in enumerate(v_run):     # for all possible runs
        filename = model.get_confounds_tsv(run)
        if os.path.isfile(filename):      # if data from this run exist
                if not sh:
                        folds['all'].append(idx)  # add slice to all runs
                else:
                        if idx % 2 == 1:          # add slice to odd runs
                                folds['odd'].append(idx)
                        else:                     # add slice to even runs
                                folds['even'].append(idx)


# part 2: analyze both hemispheres
#---------------------------------------------------------------------#
#hemis   = ['L', 'R'] # hemis is given as input variable
results = {}
      
# load data
print('\n-> Space "{}":'.format(model.space))
print('   - Loading fMRI data ... ', end='')
Y     = model.load_data_all(v_run, model.space)
M     = np.all(Y, axis=(0,2))
Y     = Y[:,M,:]
Y     = standardize_signals(Y)
V     = M.size 
print('successful!')
print('\n-> Compare the temporal dimension.')

t_dif = Y.shape[0] - X_c.shape[0]
if t_dif == 0:
        print('   - Dimension matched.')
elif t_dif == -1:
        X_c = X_c[1:,:,:] # cut the t=1 confound since we dropped the first scan
        print('   - Cut the first time point confound.')
else:
        print('   -Temporal dimension of BOLD and confounds are not matched: \n')
        print(f'   Bold: {Y.shape[0]} \n Cov: {X_c.shape[0]}')


# analyze all folds
# results[hemi] = {}
for fold in folds:

        # if fold contains runs
        num_runs = len(folds[fold])
        if num_runs > 0:

                # get fold data
                print('\n-> Runs "{}" ({} run{}: slice{} {}):'. \
                        format(fold, num_runs, ['','s'][int(num_runs>1)], \
                                ['','s'][int(num_runs>1)], ','.join([str(i) for i in folds[fold]])))
                print('   - Estimating parameters ... ', end='\n')
                Y_f    = Y[:,:,folds[fold]]
                ons_f  = [ons[i]  for i in folds[fold]]
                dur_f  = [dur[i]  for i in folds[fold]]
                stim_f = [stim[i] for i in folds[fold]]
                Xc_f   = X_c[:,:,folds[fold]]

                # analyze data
                ds = NumpRF.DataSet(Y_f, ons_f, dur_f, stim_f, TR, Xc_f)
                start_time = time.time()
                if ver == 'V0':
                        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                        ds.estimate_MLE_rgs(avg=avg, corr=corr, order=order, mu_grid=mu_grid, fwhm_grid=fwhm_grid)
                elif ver == 'V1':
                        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                        ds.estimate_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, fwhm_grid=fwhm_grid)
                elif ver == 'V2':
                        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                        ds.estimate_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, sig_grid=sig_grid, lin=False)
                elif ver == 'V2-lin':
                        mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                        ds.estimate_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, sig_grid=sig_grid, lin=True)
                if True:
                        k_est, k_null, k_const = \
                        ds.free_parameters(avg, corr, order)
                end_time   = time.time()
                difference = end_time - start_time
                del start_time, end_time

                # save results (mat-file)
                sett = str(avg[0])+','+str(avg[1])+','+str(corr)+','+str(order)
                print('\n-> Runs "{}", Model "{}", Settings "{}":'.
                        format(fold, mod.model, sett))
                print('   - Saving results file ... ', end='')
                filepath = mod_dir  + '/sub-' + mod.sub + '_ses-' + mod.ses + \
                                        '_model-' + mod.model + '_space-' + mod.space + '_'
                if sh: filepath = filepath + 'runs-' + fold + '_'
                results[fold] = filepath + 'numprf.mat'
                res_dict = {'mod_dir': mod_dir, 'settings': {'avg': avg, 'corr': corr, 'order': order}, \
                        'mu_est':  mu_est,  'fwhm_est': fwhm_est, 'beta_est':  beta_est, \
                        'MLL_est': MLL_est, 'MLL_null': MLL_null, 'MLL_const': MLL_const, \
                        'k_est':   k_est,   'k_null':   k_null,   'k_const':   k_const, \
                        'corr_est':corr_est,'version':  ver,      'time':      difference}
                sp.io.savemat(results[fold], res_dict)
                print('successful!')
                del sett, res_dict

                # save results (surface images)
                para_est = {'mu': mu_est, 'fwhm': fwhm_est, 'beta': beta_est}

                # Load your structural image (T1-weighted)
                t1_path = sess.get_anat_nii()
                t1_img = nib.load(t1_path)

                for name, data in para_est.items():
                        print('   - Saving "{}" image ... '.format(name), end='')                                                
                        filename    = filepath + name + '.map.nii.gz'
                        para_img    = save_vol(data, t1_img, filename)
                        print('successful!')
                del para_est, filename, para_img, t1_img, t1_path

# return results filename
return results



# function: calculate R-squared maps
#-------------------------------------------------------------------------#
# def calculate_Rsq(mod, folds=['all', 'odd', 'even', 'cv'], stim = False):


# part 1: prepare calculations
#---------------------------------------------------------------------#
print('\n\n-> Subject "{}", Session "{}", Model "{}":'.format(model.sub, model.ses, model.model))
mod_dir = model.get_model_dir()

# specify slices
i = -1                              # slice index (3rd dim)
slices = {'all': [], 'odd': [], 'even': []}
X_c, v_run = model.get_confounds(covs)

for idx, run in enumerate(v_run):                    # for all possible runs
        filename = model.get_confounds_tsv(run)
        if os.path.isfile(filename):    # if data from this run exist
                i = i + 1                   # increase slice index
                slices['all'].append(i)
                if idx % 2 == 1: slices['odd'].append(idx)
                else:            slices['even'].append(idx)

# part 2: analyze both hemispheres
#---------------------------------------------------------------------#
hemis = {'L': 'left', 'R': 'right'}
maps  = {}
folds = ['all']
for fold in folds: maps[fold] = {}

# for both hemispheres
# for hemi in hemis.keys():
# print('   - {} hemisphere:'.format(hemis[hemi]))


# for all folds
for fold in folds:

        # load analysis results
        filepath = mod_dir + '/sub-' + model.sub + '_ses-' + model.ses + \
                                '_model-' + model.model + '_space-' + model.space + '_'
        if fold in ['odd', 'even']:
                filepath = filepath + 'runs-' + fold + '_'
        res_file = filepath + 'numprf.mat'
        NpRF     = sp.io.loadmat(res_file)

        # calculate R-squared (all, odd, even)
        avg   = list(NpRF['settings']['avg'][0,0][0,:])
        MLL1  = np.squeeze(NpRF['MLL_est'])
        MLL00 = np.squeeze(NpRF['MLL_const'])
        r0,n0 = model.calc_runs_scans(fold)
        n1    = r0*n0
        Rsq   = NumpRF.MLL2Rsq(MLL1, MLL00, n1)

        # threshold tuning maps
        print('     - Saving R-squared image for {} runs ... '.format(fold), end='')

        # load your one functional image for reference
        ref_path = model.get_bold_nii(v_run[0], model.space)
        ref_img  = nib.load(ref_path) 

        # Load binary brain mask
        M     = model.get_volumetric_mask(model.space, v_run)


        if fold in ['all', 'odd', 'even']:
                filename   = filepath + 'Rsq.volumetric.nii.gz'
        else:
                filename   = filepath + 'cvRsq.volumetric.nii.gz'
        

        para_img       = save_vol_mask(Rsq, ref_img, filename, M)
        maps[fold] = filename
        print('successful!')
        del para_map, voxel, filename, para_img
        
# return results filename
# return maps


# get model directory path
mod_dir  = model.get_model_dir()
filepath = mod_dir  + '/sub-' + model.sub + '_ses-' + model.ses + '_model-' + model.model + \
                        '_space-' + model.space + '_'
file     = filepath + 'numprf.mat'

# extract valid runs
X_c, v_run = model.get_confounds(covs)

# load your one functional image for reference
ref_path = model.get_bold_nii(v_run[0], model.space)
ref_img  = nib.load(ref_path) 

# Load binary brain mask
M = model.get_volumetric_mask(model.space, v_run)

# load estimated data
NumpRF = sp.io.loadmat(file)

param_lists = ['fwhm_est', 'mu_est', 'beta_est']
for param in param_lists:
        data     = NumpRF[param]
        prefix   = param.split('_')[0]
        filename = filepath + prefix + '.volumetric.nii.gz'
        para_img = save_vol_mask(data, ref_img, filename, M)
        print(f'   - Saving "{prefix}" image ... successful!')

# function: save single volume image (3D)
#-----------------------------------------------------------------------------#
def save_vol_mask(data, img, fname, mask):
    """
    Save Volume with Masked Data
    
    data  - 1D array (length = number of valid voxels)
    img   - Nifti1Image template (for shape, affine, header)
    fname - Output filename
    mask  - 1D boolean array or 3D boolean array; same shape as img
    
    Returns: new Nifti1Image
    """
    # Flatten mask if it's 3D
    if mask.ndim == 3:
        mask = mask.flatten(order='C')

    # Create empty volume
    spatial_shape = img.shape[:3]
    n_voxels = np.prod(spatial_shape)
    full_data = np.zeros(n_voxels, dtype=data.dtype)

    # Fill in only the masked voxels
    full_data[mask] = data.flatten()

    # create and save image
    data_map = full_data.reshape(spatial_shape, order='C')
    data_img = nib.Nifti1Image(data_map, img.affine, img.header)
    nib.save(data_img, fname)
    
    # load and return image
    data_img = nib.load(fname)
    return data_img
#
# Uncomment analysis you want to implement
#-------------------------------------------------------------------------#
"""
        1. "analyze_numerosity": Estimate Numerosities and FWHMs for Surface-Based Data
        2. "calculate_Rsq": Calculate R-Squared Maps for Numerosity Model
        3. "threshold_maps": Threshold Numerosity, FWHM and Scaling Maps based on Criterion: Bonferoni correction on p = .05
        4. "": 
"""



# analyze numerosities
#-------------------------------------------------------------------------#
if   target == "all":
        # analyze numerosities
        model.analyze_numerosity(avg=[True, False], corr='iid', order=1, ver='V2', stim = False, hemis = ['L','R'], sh=False)
        
        # model.analyze_numerosity(avg=[True, False], corr='iid', order=1, ver='V2', stim = False, hemis = ['L','R'], sh=False) # all runs
        # calculate R-squared
        # model.calculate_Rsq(folds=['all'], stim = False) # all runs

elif target == "split":
        # analyze numerosities
        model.analyze_numerosity(avg=[True, False], corr='iid', order=1, ver='V2', stim = False, hemis = ['L','R'], sh=True) # even and odd runs
        # cross-validated R-squared
        model.calculate_Rsq(folds=['cv'], stim = False)# cross-validated
        # function: threshold tuning maps
        model.threshold_maps(crit='Rsqmb,p=0.05B', cv=True)# cross-validated
else:
        raise ValueError(f"Invalid target '{target}'. Expected 'all' or 'split'.")



#~~~~~~~~~~~ Function ~~~~~~~~~~~~~

# function: standardize confounds
#-----------------------------------------------------------------------------#
def standardize_confounds(X, std=[True, True]):
    """
    Standardize Confound Variables for GLM Modelling
    X = standardize_confounds(X, std)
    
        X   - n x c x r array; scan-by-variable-by-run signals
        std - list of bool; indicating which operations to perform (see below)
    
    X = standardize_confounds(X, std) standardizes confounds, i.e. subtracts
    the mean from each variable (in each run), if the first entry of std is
    true, and divides by the mean from each variable (in each run), if the
    second entry of std is true. By default, both entries are true.
    """
    
    # if X is a 2D matrix
    if len(X.shape) < 3:
        for k in range(X.shape[1]):
            if std[0]:          # subtract mean
                X[:,k] = X[:,k] - np.mean(X[:,k])
            if std[1]:          # divide by max
                X[:,k] = X[:,k] / np.max(X[:,k])
    
    # if X is a 3D array
    else:
        for j in range(X.shape[2]):
            for k in range(X.shape[1]):
                if std[0]:      # subtract mean
                    X[:,k,j] = X[:,k,j] - np.mean(X[:,k,j])
                if std[1]:      # divide by max
                    X[:,k,j] = X[:,k,j] / np.max(X[:,k,j])
    
    # return standardized confounds
    return X

# function: correct onsets
#-----------------------------------------------------------------------------#
def correct_onsets(ons, dur, stim):
    """
    Correct Onsets Measured during EMPRISE Task
    ons, dur, stim = correct_onsets(ons, dur, stim)
    
        ons    - b x 1 vector; block-wise onsets [s]
        dur    - b x 1 vector; block-wise durations [s]
        stim   - b x 1 vector; block-wise stimuli (b = blocks)
        
        ons    - b0 x 1 vector; block-wise onsets [s]
        dur    - b0 x 1 vector; block-wise durations [s]
        stim   - b0 x 1 vector; block-wise stimuli (b0 = blocks per epoch)
    
    ons, dur, stim = correct_onsets(ons, dur, stim) corrects onsets ons,
    durations dur and stimuli stim, if signals are averaged across epochs
    within run. This is done by only using onsets, durations and stimuli from
    the first epoch and subtracting the discarded scan time from the onsets.
    """
    
    # correct for epochs
    ons  = ons[:blocks_per_epoch] - num_scan_disc*TR
    dur  = dur[:blocks_per_epoch]
    stim = stim[:blocks_per_epoch]
        
    # return corrected onsets
    return ons, dur, stim


def get_onsets(mod, valid_runs, filenames=None):
        """
        Get Onsets and Durations for Single Subject and Session, all Runs
        ons, dur, stim = sess.get_onsets(valid_runs,filenames)
        
            valid_runs- list of integers; from 1 to 9, valid run numbers
            filenames - list of strings; "events.tsv" filenames
        
            ons       - list of arrays of floats; t x 1 vectors of onsets [s]
            dur       - list of arrays of floats; t x 1 vectors of durations [s]
            stim      - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
            
        ons, dur, stim = sess.get_onsets(valid_runs,filenames) loads the "events.tsv" file
        belonging to session sess and returns lists of length number of runs,
        containing, as each element, lists of length number of trials per run,
        containing onsets and durations in seconds as well as stimuli in
        numerosity.
        """
        
        # prepare onsets, durations, stimuli as empty lists
        ons  = []
        dur  = []
        stim = []
        
        # prepare labels for trial-wise extraction
        if mod.ses == 'visual':
            stimuli = {'1_dot': 1, '2_dot': 2, '3_dot': 3, '4_dot': 4, '5_dot': 5, \
                       '6_dot': 6, '7_dot': 7, '8_dot': 8, '9_dot': 9, '20_dot': 20}
        elif mod.ses == 'digit':
            stimuli = {'1_digit': 1, '2_digit': 2, '3_digit': 3, '4_digit': 4, '5_digit': 5, \
                       '6_digit': 6, '7_digit': 7, '8_digit': 8, '9_digit': 9, '20_digit': 20}
        elif mod.ses in ['audio1','audio2','spoken']:
            stimuli = {'1_audio': 1, '2_audio': 2, '3_audio': 3, '4_audio': 4, '5_audio': 5, \
                       '6_audio': 6, '7_audio': 7, '8_audio': 8, '9_audio': 9, '20_audio': 20}
        elif mod.ses == 'congruent' or mod.ses == 'incongruent':
            stimuli = {              '2_mixed': 2, '3_mixed': 3, '4_mixed': 4, '5_mixed': 5, '20_mixed': 20}
        
        # for all runs
        for j, run in enumerate(valid_runs):
            
            # extract filename
            if filenames is None:
                filename = mod.get_events_tsv(run)
            else:
                filename = filenames[j]
            
            # if onset file exists
            if os.path.isfile(filename):
                validity_check = mod.get_confounds_tsv(run)
                if os.path.isfile(validity_check):
                    print('\n     run ', run, ' is included to analysis.')
                    # extract events of interest
                    events = pd.read_csv(filename, sep='\t')
                    events = events[events['trial_type']!='button_press']
                    for code in stimuli.keys():
                        events.loc[events['trial_type']==code+'_attn','trial_type'] = code
                    
                    # save onsets, durations, stimuli
                    stims = [stimuli[trl] for trl in events['trial_type']]
                    ons.append(np.array(events['onset']))
                    dur.append(np.array(events['duration']))
                    stim.append(np.array(stims))
                
        # return onsets
        return ons, dur, stim


def onsets_trials2blocks(ons, dur, stim, mode='true'):
    """
    Transform Onsets and Durations from Trials to Blocks
    ons, dur, stim = onsets_trials2blocks(ons, dur, stim, mode)
    
        ons  - list of arrays of floats; t x 1 vectors of onsets [s]
        dur  - list of arrays of floats; t x 1 vectors of durations [s]
        stim - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
        mode - string; duration conversion ("true" or "closed")

        ons  - list of arrays of floats; b x 1 vectors of onsets [s]
        dur  - list of arrays of floats; b x 1 vectors of durations [s]
        stim - list of arrays of floats; b x 1 vectors of stimuli (b = blocks)
        
    ons, dur, stim = onsets_trials2blocks(ons, dur, stim, mode) transforms
    onsets ons, durations dur and stimuli stim from trial-wise vectors to
    block-wise vectors.
    
    If mode is "true" (default), then the actual durations are used. If mode is
    "closed", then each block ends not earlier than when the next block starts.
    """
    
    # prepare onsets, durations, stimuli as empty lists
    ons_in  = ons; dur_in  = dur; stim_in = stim
    ons     = [];  dur     = [];  stim    = []
    
    # for all runs
    for j in range(len(ons_in)):
        
        # prepare onsets, durations, stimuli for this run
        ons.append([])
        dur.append([])
        stim.append([])
        
        # for all trials
        for i in range(len(ons_in[j])):
            
            # detect first block, last block and block change
            if i == 0:
                ons[j].append(ons_in[j][i])
                stim[j].append(stim_in[j][i])
            elif i == len(ons_in[j])-1:
                if mode == 'true':
                    dur[j].append((ons_in[j][i]+dur_in[j][i]) - ons[j][-1])
                elif mode == 'closed':
                    dur[j].append(max(dur[j]))
            elif stim_in[j][i] != stim_in[j][i-1]:
                if mode == 'true':
                    dur[j].append((ons_in[j][i-1]+dur_in[j][i-1]) - ons[j][-1])
                elif mode == 'closed':
                    dur[j].append(ons_in[j][i] - ons[j][-1])
                ons[j].append(ons_in[j][i])
                stim[j].append(stim_in[j][i])
        
        # convert lists to vectors
        ons[j]  = np.array(ons[j])
        dur[j]  = np.array(dur[j])
        stim[j] = np.array(stim[j])
    
    # return onsets
    return ons, dur, stim


# function: get "timeseries.tsv" filenames
#-------------------------------------------------------------------------#
def get_confounds_tsv(mod, run_no):
        """
        Get Filename for Confounds TSV File
        filename = sess.get_confounds_tsv(run_no)

                run_no   - int; run number (e.g. 1)
                
                filename - string; filename of "timeseries.tsv"

        filename = get_confounds_tsv(run_no) returns the filename of the
        tab-separated confounds file belonging to session sess and run run_no.
        """

        # create filename
        filename = deri_dir + 'fmriprep' + \
                        '/sub-' + mod.sub + '/ses-' + mod.ses + '/func' + \
                        '/sub-' + mod.sub + '_ses-' + mod.ses + '_task-' + task + \
                        '_acq-' + acq[1] + '_run-' + str(run_no) + '_desc-' + desc[2] + '_timeseries.tsv'
        return filename


# function: load in-mask data
#-------------------------------------------------------------------------#
def load_mask_data(mod, hemi='L'):
        """
        Load Functional MRI GIfTI Files and Mask
        Y = sess.load_mask_data(hemi, space)
                
                hemi  - string; brain hemisphere (e.g. "L")
                
                Y     - n x v x r array; scan-by-vertex-by-run fMRI data
        """

        # load and mask data
        Y = mod.load_surf_data_all(hemi, mod.space)
        M = np.all(Y, axis=(0,2))
        Y = Y[:,M,:]

        # return data and mask
        return Y, M


# function: load surface fMRI data (all runs)
#-------------------------------------------------------------------------#
def load_surf_data_all(mod, hemi='L', space='fsnative'):
        """
        Load Functional MRI GIfTI Files from All Runs
        Y = sess.load_surf_data_all(hemi, space)
                
                hemi  - string; brain hemisphere (e.g. "L")
                space - string; image space (e.g. "fsnative")
                
                Y     - n x V x r array; scan-by-vertex-by-run fMRI data
        """

        # Collect valid runs and their scan numbers
        valid_runs = []
        r_scan = []
        for run in runs:
                filename = mod.get_bold_gii(run, hemi, space)
                if os.path.isfile(filename):
                        Y = mod.load_surf_data(run, hemi, space)
                        valid_runs.append(run)
                        r_scan.append(Y.shape[0])

        # Determine minimum scan number
        min_scans = min(r_scan)

        # Filter out runs with insufficient scans
        if mod.ses in ['visual','audio1','digit','spoken']:
                if min_scans <= int(n*0.75) : # EMPRISE: 108 EMPRISE2: 156 (3/4 of the scans of an entire run)
                        valid_runs = [run for run, scans in zip(valid_runs, r_scan) if scans > int(n*0.75)]
                        r_scan = [scans for scans in r_scan if scans > int(n*0.75)]
                        min_scans = min(r_scan) if r_scan else 0

        elif mod.ses in ['audio2']:
                if min_scans <= int(n2*0.75) : #EMPRISE2 audio2: 142 (3/4 of the scans of an entire run)
                        valid_runs = [run for run, scans in zip(valid_runs, r_scan) if scans > int(n2*0.75)]
                        r_scan = [scans for scans in r_scan if scans > int(n2*0.75)]            
                        min_scans = min(r_scan) if r_scan else 0


        # Prepare output array
        if not r_scan:
                return None

        Y = np.zeros((min_scans, Y.shape[1], len(r_scan)))

        # Load data
        for j, run in enumerate(valid_runs):
                Y[:,:,j] = mod.load_surf_data(run, hemi, space)[:min_scans,:]

        # Remove empty runs
        Y = Y[:,:,np.any(Y, axis=(0,1))]


        return Y

# function: load fMRI data (all runs)
#-------------------------------------------------------------------------#
def load_data_all(mod, runs, space=''):
        """
        Load Functional MRI NIfTI Files from All Runs
        Y = sess.load_data_all(space)
                
                space - string; image space (e.g. "T1w")
                
                Y     - n x V x r array; scan-by-voxel-by-run fMRI data
        """

        # prepare 3D array
        for j, run in enumerate(runs):
                filename = mod.get_bold_nii(run, space)
                if os.path.isfile(filename):
                        Y_tmp = mod.load_data(run, space)
                        Y_tmp = Y_tmp[1:, :]
                break
        Y = np.zeros((Y_tmp.shape[0], Y_tmp.shape[1], len(runs)))

        # load fMRI data
        for j, run in enumerate(runs):
                filename = mod.get_bold_nii(run, space)
                if os.path.isfile(filename):
                        Y[:,:,j] = mod.load_data(run, space)[1:, :]

        # select available runs
        Y = Y[:,:,np.any(Y, axis=(0,1))]
        return Y


# function: standardize signals
#-----------------------------------------------------------------------------#
def standardize_signals(Y, std=[True, True]):
    """
    Standardize Measured Signals for ReML Estimation
    Y = standardize_signals(Y, std)
    
        Y   - n x v x r array; scan-by-voxel-by-run signals
        std - list of bool; indicating which operations to perform (see below)
    
    Y = standardize_signals(Y, std) standardizes signals, i.e. it sets the mean
    of each time series (in each run) to 100, if the first entry of std is
    true, and scales the signal to percent signal change (PSC), if the second
    entry of std is true. By default, both entries are true.
    """
    
    # if Y is a 2D matrix
    if len(Y.shape) < 3:
        for k in range(Y.shape[1]):
            mu     = np.mean(Y[:,k])
            Y[:,k] = Y[:,k] - mu
            if std[1]:
                Y[:,k] = Y[:,k]/mu * 100
            if std[0]:
                Y[:,k] = Y[:,k] + 100
            else:
                Y[:,k] = Y[:,k] + mu
    
    # if Y is a 3D array
    else:
        for j in range(Y.shape[2]):
            for k in range(Y.shape[1]):
                mu      = np.mean(Y[:,k,j])
                Y[:,k,j] = Y[:,k,j] - mu
                if std[1]:
                    Y[:,k,j] = Y[:,k,j]/mu * 100
                if std[0]:
                    Y[:,k,j] = Y[:,k,j] + 100
                else:
                    Y[:,k,j] = Y[:,k,j] + mu
    
    # return standardized signals
    return Y


# function: save single volume image (3D)
#-----------------------------------------------------------------------------#
def save_vol_mask(data, img, fname, mask):
    """
    Save Volume with Masked Data
    
    data  - 1D array (length = number of valid voxels)
    img   - Nifti1Image template (for shape, affine, header)
    fname - Output filename
    mask  - 1D boolean array or 3D boolean array; same shape as img
    
    Returns: new Nifti1Image
    """
    # Flatten mask if it's 3D
    if mask.ndim == 3:
        mask = mask.flatten(order='C')

    # Create empty volume
    spatial_shape = img.shape[:3]
    n_voxels = np.prod(spatial_shape)
    full_data = np.zeros(n_voxels, dtype=data.dtype)

    # Fill in only the masked voxels
    full_data[mask] = data.flatten()

    # create and save image
    data_map = full_data.reshape(spatial_shape, order='C')
    data_img = nib.Nifti1Image(data_map, img.affine, img.header)
    nib.save(data_img, fname)
    
    # load and return image
    data_img = nib.load(fname)
    return data_img

