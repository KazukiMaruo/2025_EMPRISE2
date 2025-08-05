# install library
#-------------------------------------------------------------------------#
import os
import re
import numpy as np
import scipy as sp
import nibabel as nib
import math
import matplotlib.pyplot as plt

# Parameters
#-------------------------------------------------------------------------#
"""
Modify parameters for your analysis in config.json and Figures.py
code_dir: specify the directory containing "config.json" file.
"""
code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"

# # Install functions
#-------------------------------------------------------------------------#
import Figures
import json
import EMPRISE
import NumpRF
import PySPMs as PySPM

# function: p-value string
#-----------------------------------------------------------------------------#
def pvalstr(p, p_thr=0.001, sig_thr=[]):
    """
    Convert p-Value to p-Value String
    p_str = pvalstr(p, p_thr, sig_thr)
    
        p       - float; p-value
        p_thr   - float; p-value threshold under which not to display
        sig_thr - list; significance thresholds defining asterisks
        
        p_str   - string; p-value string
    """
    
    # get number of significant digits
    num_dig = -math.floor(math.log10(p_thr))
    
    # create p-value string
    if p < p_thr:
        p_str = ('p < {' + ':.{}f'.format(num_dig) + '}').format(p_thr)
    else:
        p_str = ('p = {' + ':.{}f'.format(num_dig) + '}').format(p)
    
    # add significance markers
    sig_str = '*' * np.sum(p<np.array(sig_thr))
    p_str   = p_str + sig_str
    return p_str

# import config.json
#-----------------------------------------------------------------------------#
with open ("config.json") as f:
    config = json.load(f)
globals().update(config)

# specify default model
#-----------------------------------------------------------------------------#
model_def = MODEL
session   = SESSION
subject   = SUBJECT
space     = SPACE
cv        = CV
filepath  = f"{DATA_ANALYSIS_DIR}/Figures/{SESSION}/{model_def}/"


# Figure S2
cv = CV
model = model_def

fig = plot_tuning_function_time_course(subject, session, model, 'L', space, cv=cv, verts=[0], col='b')
if cv:
    filename = filepath+'WP1_Figure_S2'+'_ses-'+session+'_sub-'+subject+'_cv.png'
else:
    filename = filepath+'WP1_Figure_S2'+'_ses-'+session+'_sub-'+subject+'.png'

os.makedirs(os.path.dirname(filename), exist_ok=True)
fig.savefig(filename, dpi=150, transparent=True)

# Figures.WP1_Fig1(Figure='S2')


# sub-function: plot tuning functions & time courses
#-------------------------------------------------------------------------#
# def plot_tuning_function_time_course_volumetric(sub, ses, model, space, cv=True, verts=[0], col='b'):
# hemi = 'L'
# verts = [0]
# col = 'b'
# load session data
mod             = EMPRISE.Model(subject, session, model, space)
# Y, M            = mod.load_mask_data(hemi)
labels          = EMPRISE.covs
X_c, valid_runs = mod.get_confounds(labels)
X_c             = EMPRISE.standardize_confounds(X_c)
X_c             = X_c[1:,:,:] # converting from 210 to 209
ons, dur, stim  = mod.get_onsets(valid_runs)
ons, dur, stim  = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
ons0,dur0,stim0 = ons[0], dur[0], stim[0]
ons, dur, stim  = EMPRISE.correct_onsets(ons0, dur0, stim0)

#####MODIFIED###### 
# fMRI data = scans * voxels * runs
Y               = mod.load_data_all(valid_runs, mod.space)
# Load binary brain mask
M               = mod.get_volumetric_mask(mod.space, valid_runs) 

# load model results
if model_def == 'Volumetric': 
    res_file = mod.get_results_file_volumetric()
else:
    res_file = mod.get_results_file(hemi)

# debug: remove 'model'
res_file = re.sub(r'model-model-', 'model-', res_file)
NpRF     = sp.io.loadmat(res_file)
"""
mu       = np.squeeze(NpRF['mu_est'])
fwhm     = np.squeeze(NpRF['fwhm_est'])
beta     = np.squeeze(NpRF['beta_est'])
"""
mu_map = res_file[:res_file.find('numprf.mat')] + 'mu_thr-Rsqmb,p=0.05B.nii.gz'
mu     = nib.load(mu_map).get_fdata()
ori_shape  = mu.shape
mu     = mu.flatten(order='C')

fwhm_map = res_file[:res_file.find('numprf.mat')] + 'fwhm_thr-Rsqmb,p=0.05B.nii.gz'
fwhm     = nib.load(fwhm_map).get_fdata()
fwhm     = fwhm.flatten(order='C')

beta_map = res_file[:res_file.find('numprf.mat')] + 'beta_thr-Rsqmb,p=0.05B.nii.gz'
beta     = nib.load(beta_map).get_fdata()
beta     = beta.flatten(order='C')
#####MODIFIED######

# determine tuning function
try:
    ver = NpRF['version'][0]
except KeyError:
    ver = 'V2'
linear_model = 'lin' in ver


""" XYZ coordinates information   
# load vertex-wise coordinates
hemis     = {'L': 'left', 'R': 'right'}
para_map  = res_file[:res_file.find('numprf.mat')] + 'mu.surf.gii'
para_mask = nib.load(para_map).darrays[0].data != 0
mesh_file = mod.get_mesh_files(space,'pial')[hemis[hemi]]
mesh_gii  = nib.load(mesh_file)
XYZ       = mesh_gii.darrays[0].data
XYZ       = XYZ[para_mask,:]
del para_map, mesh_file, mesh_gii
# Remark: the vertex indices get changed when the mask is applied. 
# Therefore, the marked vertex number on our figure is not matched to the original vertex indices.
# If you want to find the corresponding (max Rsq) verex index on fsnative or fsaverage surface, 
# use the coordinate of max Rsq vertex and find corresponding vertex index on unmasked surface.
"""

# if CV, load cross-validated R^2
if cv:
    Rsq_map = res_file[:res_file.find('numprf.mat')] + 'cvRsq.surf.gii'
    cvRsq   = nib.load(Rsq_map).darrays[0].data
    Rsq     = cvRsq[para_mask]
    del Rsq_map, para_mask, cvRsq
        
# otherwise, calculate total R^2
else:
    Rsq_map = res_file[:res_file.find('numprf.mat')] + 'Rsq_thr-Rsqmb,p=0.05B.nii.gz'
    Rsq     = nib.load(Rsq_map).get_fdata()
    Rsq     = Rsq.flatten(order='C')
    del Rsq_map


# select vertices for plotting
for k, vertex in enumerate(verts):
    if vertex == 0:
        if k == 0 and len(verts) == 1:
            # select maximum R^2 in vertices with numerosity 1 < mu < 5
            vertex = np.nanargmax(Rsq) ####MODIFIED
        elif k == 0 and len(verts) == 2:
            # select maximum R^2 in vertices with numerosity 1 < mu < 2
            vertex = np.argmax(Rsq + np.logical_and(mu>1, mu<2)+ (beta>0))
        elif k == 1:
            # select maximum R^2 in vertices with numerosity 4 < mu < 5 or 3 < mu < 5
            if ses == 'visual': vertex = np.argmax(Rsq + np.logical_and(mu>4, mu<5) + (fwhm<EMPRISE.fwhm_thr[1]) + (beta>0))
            if ses == 'audio':  vertex = np.argmax(Rsq + np.logical_and(mu>3, mu<5) + (fwhm<EMPRISE.fwhm_thr[1]) + (beta>0))
        verts[k] = vertex
    elif vertex == 1: # vert = [1]
        # select the top 5 maximum R^2 in vertices with numerosity 1 < mu < 2
        vert_mask = np.logical_and(mu > 1, mu < 2)
        top_5_indices = np.argpartition(-Rsq[vert_mask], 5)[:5]
        vertex = np.where(vert_mask)[0][top_5_indices]
        vertex = vertex.tolist()
        verts  = vertex
    
# plot selected vertices
fig = plt.figure(figsize=(24,len(verts)*8))
axs = fig.subplots(len(verts), 2, width_ratios=[4,6])
if len(verts) == 1: axs = np.array([axs])
xr  = EMPRISE.mu_thr        # numerosity range
xm  = xr[1]+1               # maximum numerosity
dx  = 0.05                  # numerosity delta
    
# Figure 1A: estimated tuning functions
for k, vertex in enumerate(verts):

    # compute vertex tuning function
    x  = np.arange(dx, xm+dx, dx)
    xM = mu[vertex]
    if not linear_model:
        mu_log, sig_log = NumpRF.lin2log(mu[vertex], fwhm[vertex])
        y  = NumpRF.f_log(x, mu_log, sig_log)
        x1 = np.exp(mu_log - math.sqrt(2*math.log(2))*sig_log)
        x2 = np.exp(mu_log + math.sqrt(2*math.log(2))*sig_log)
    else:
        mu_lin, sig_lin = (mu[vertex], NumpRF.fwhm2sig(fwhm[vertex]))
        y  = NumpRF.f_lin(x, mu_lin, sig_lin)
        x1 = mu[vertex] - fwhm[vertex]/2
        x2 = mu[vertex] + fwhm[vertex]/2
    x1 = np.max(np.array([0,x1])) 
    x2 = np.min(np.array([x2,xm])) 
    
    # plot vertex tuning function
    hdr  = 'sub-{}, ses-{}, space-{}'.format(subject, session, space) ### MODIFIED
    xxx, yyy, zzz = np.unravel_index(vertex, ori_shape, order='C')
    txt1 = 'XYZ = [{:.0f}, {:.0f}, {:.0f}])'. \
            format(xxx, yyy, zzz)
    txt2 =('mu = {:.2f}\n fwhm = {' + ':.{}f'.format([1,2][int(fwhm[vertex]<10)]) + '}'). \
            format(mu[vertex], fwhm[vertex])
    axs[k,0].plot(x[x<=xr[0]], y[x<=xr[0]], '--'+col, linewidth=2)
    axs[k,0].plot(x[x>=xr[1]], y[x>=xr[1]], '--'+col, linewidth=2)
    axs[k,0].plot(x[np.logical_and(x>=xr[0],x<=xr[1])], y[np.logical_and(x>=xr[0],x<=xr[1])], '-'+col, linewidth=2)
    if k == 0:
        axs[k,0].plot([xM,xM], [0,1], '-', color='gray', linewidth=2)
        axs[k,0].plot([x1,x2], [0.5,0.5], '-', color='gray', linewidth=2)
        axs[k,0].text(xM, 0+0.01, ' mu = preferred numerosity', fontsize=18, \
                        horizontalalignment='left', verticalalignment='bottom')
        axs[k,0].text((x1+x2)/2, 0.5-0.01, 'fwhm = tuning width', fontsize=18, \
                        horizontalalignment='center', verticalalignment='top')
    axs[k,0].axis([0, xm, 0, 1+(1/20)])
    if k == len(verts)-1:
        axs[k,0].set_xlabel('presented numerosity', fontsize=32)
    axs[k,0].set_ylabel('neuronal response', fontsize=32)
    if k == 0:
        axs[k,0].set_title('', fontweight='bold', fontsize=32)
    axs[k,0].tick_params(axis='both', labelsize=18)
    axs[k,0].text(xm-(1/20)*xm, 0.85, txt1, fontsize=18,
                    horizontalalignment='right', verticalalignment='top')
    axs[k,0].text(xm-(1/20)*xm, 0.05, txt2, fontsize=18,
                    horizontalalignment='right', verticalalignment='bottom')
    del x, y, xM, x1, x2
    
# Figure 1B: predicted time courses
for k, vertex in enumerate(verts):

    # compute "observed" signal
    y     = EMPRISE.standardize_signals(Y[:,[vertex],:]) - 100
    y_reg = np.zeros(y.shape)
    for j in range(y.shape[2]): # run loop
        glm          = PySPM.GLM(y[:,:,j], np.c_[X_c[:,:,j], np.ones((y.shape[0],1))])
        b_reg        = glm.OLS()
        y_reg[:,:,j] = glm.Y - glm.X @ b_reg
    
    # get vertex tuning parameters
    if not linear_model:
        mu_k, sig_k = NumpRF.lin2log(mu[vertex], fwhm[vertex])
    else:
        mu_k, sig_k = (mu[vertex], NumpRF.fwhm2sig(fwhm[vertex]))
    
    # compute predicted signal (run)
    y_run, t = EMPRISE.average_signals(y_reg, None, [True, False])
    z, t = NumpRF.neuronal_signals(ons0, dur0, stim0, EMPRISE.TR, EMPRISE.mtr, np.array([mu_k]), np.array([sig_k]), lin=linear_model)
    s, t = NumpRF.hemodynamic_signals(z, t, y_run.shape[0], EMPRISE.mtr)
    glm  = PySPM.GLM(y_run, np.c_[s[:,:,0], np.ones((y_run.shape[0], 1))])
    b_run= glm.OLS()
    s_run= glm.X @ b_run
    
    # compute predicted signal (epoch)
    y_avg, t = EMPRISE.average_signals(y_reg, None, [True, True])
    # Note: For visualization purposes, we here apply "avg = [True, True]".
    z, t = NumpRF.neuronal_signals(ons, dur, stim, EMPRISE.TR, EMPRISE.mtr, np.array([mu_k]), np.array([sig_k]), lin=linear_model)
    s, t = NumpRF.hemodynamic_signals(z, t, y_avg.shape[0], EMPRISE.mtr)
    glm  = PySPM.GLM(y_avg, np.c_[s[:,:,0], np.ones((y_avg.shape[0], 1))])
    b_avg= glm.OLS()
    s_avg= glm.X @ b_avg
    
    # assess statistical significance
    Rsq_run = Rsq[vertex]
    Rsq_avg = NumpRF.yp2Rsq(y_avg, s_avg)[0]
    p       = [4,2][int(cv)]
    # Note: Typically, we loose 4 degrees of freedom for estimating 4
    # parameters (mu, fwhm, beta, beta_0). But if tuning parameters
    # (mu, fwhm) come from independent data, we only loose 2 degrees
    # of freedom for estimating 2 parameters (beta, beta_0).
    p_run = NumpRF.Rsq2pval(Rsq_run, EMPRISE.n, p)
    p_avg = NumpRF.Rsq2pval(Rsq_avg, y_avg.size, p)
    
    # prepare axis limits
    y_min = np.min(y_avg)
    y_max = np.max(y_avg)
    #y_min  = np.min(y_run)
    #y_max = np.max(y_run)
    y_rng = y_max-y_min
    t_max = np.max(t)
    xM    = t[np.argmax(s_avg)]
    #xM    = t[np.argmax(s_run)]
    yM    = np.max(s_avg)
    #yM    = np.max(s_run)
    y0    = b_avg[1,0]
    #y0    = b_run[1,0]
    
    # plot hemodynamic signals
    if cv:
        txt = 'beta = {:.2f}\ncvR² = {:.2f}, {}\n '. \
            format(beta[vertex], Rsq_run, pvalstr(p_run))
    else:
        txt = 'beta = {:.2f}\nR² = {:.2f}, {}\n '. \
            format(beta[vertex], Rsq_run, pvalstr(p_run))
    # Note: For visualization purposes, we here apply "avg = [True, True]".
    axs[k,1].plot(t, y_avg[:,0], ':ok', markerfacecolor='k', markersize=8, linewidth=1, label='measured signal')
    #axs[k,1].plot(t, y_run[:,0], ':ok', markerfacecolor='k', markersize=8, linewidth=1, label='measured signal')
    axs[k,1].plot(t, s_avg[:,0], '-'+col, linewidth=2, label='predicted signal')
    #axs[k,1].plot(t, s_run[:,0], '-'+col, linewidth=2, label='predicted signal')
    for i in range(len(ons)):
        axs[k,1].plot(np.array([ons[i],ons[i]]), \
                        np.array([y_max+(1/20)*y_rng, y_max+(3/20)*y_rng]), '-k')
        axs[k,1].text(ons[i]+(1/2)*dur[i], y_max+(2/20)*y_rng, str(stim[i]), fontsize=18,
                        horizontalalignment='center', verticalalignment='center')
    axs[k,1].plot(np.array([ons[-1]+dur[-1],ons[-1]+dur[-1]]), \
                    np.array([y_max+(1/20)*y_rng, y_max+(3/20)*y_rng]), '-k')
    axs[k,1].plot(np.array([ons[0],ons[-1]+dur[-1]]), \
                    np.array([y_max+(1/20)*y_rng, y_max+(1/20)*y_rng]), '-k')
    if k == 0:
        axs[k,1].plot([xM,xM], [y0,yM], '-', color='gray', linewidth=2)
        axs[k,1].text(xM, (y0+yM)/2, 'beta = scaling factor', fontsize=18, \
                        horizontalalignment='right', verticalalignment='center', rotation=90)
        axs[k,1].legend(loc='lower right', fontsize=18)
    axs[k,1].axis([0, t_max, y_min-(1/20)*y_rng, y_max+(3/20)*y_rng])
    if k == len(verts)-1:
        axs[k,1].set_xlabel('within-cycle time [s]', fontsize=32)
    axs[k,1].set_ylabel('hemodynamic signal [%]', fontsize=32)
    if k == 0:
        axs[k,1].set_title('presented numerosity', fontsize=32)
    axs[k,1].tick_params(axis='both', labelsize=18)
    axs[k,1].text((1/20)*t_max, y_min-(1/20)*y_rng, txt, fontsize=18, \
                    horizontalalignment='left', verticalalignment='bottom')
    del y, y_reg, y_avg, z, s, s_avg, t, b_reg, b_avg, mu_k, sig_k, y_min, y_max, y_rng, txt
    #del y, y_reg, y_run, z, s, s_run, t, b_reg, b_run, mu_k, sig_k, y_min, y_max, y_rng, txt
    
# return figure
return fig

