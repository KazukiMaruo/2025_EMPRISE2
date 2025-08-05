# -*- coding: utf-8 -*-
"""
Analyses of the EMPRISE data set
by Joram Soch <soch@cbs.mpg.de>
"""


# import modules
import os
import time
import NumpRF
import EMPRISE
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


### NumpRF empirical analysis, 12,16/10/2023 ##################################

# # define analyses
# # subs = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', 
# #         '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '116']
# subs = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012']
# # sess = ['visual', 'audio', 'digits', 'spoken']
# sess = ['visual', 'audio']
# # spcs = ['fsnative', 'fsaverage']
# spcs = ['fsnative', 'fsaverage']
# # mods = [{'avg': [True, False], 'noise': 'iid', 'hrfs': 1},
# #         {'avg': [True, False], 'noise': 'ar1', 'hrfs': 3}]
# mods = [{'avg': [True, False], 'noise': 'iid', 'hrfs': 1}]

# # specify folder
# targ_dir = EMPRISE.tool_dir + '../../../Analyses/'

# # perform analyses
# for sub in subs:
#     for ses in sess:
#         for space in spcs:
#             for model in mods:
                
#                 # determine results filename
#                 mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])
#                 subj_dir = EMPRISE.deri_out + 'numprf' + '/sub-' + sub + '/ses-' + ses + '/model-' + mod_name + '/'
#                 filepath = 'sub-' + sub + '_ses-' + ses + '_model-' + mod_name + '_hemi-' + 'L' + '_space-' + space + '_'
#                 res_file = filepath + 'numprf.mat'
#                 para_map = filepath + 'mu.surf.gii'
                
#                 # if results file already exists, let the user know
#                 if os.path.isfile(subj_dir+res_file):
                    
#                     # display message
#                     print('\n\n-> Subject "{}", Session "{}", Model "{}", Space "{}":'. \
#                           format(sub, ses, mod_name, space))
#                     print('   - results file does already exist, model is not estimated!')
#                     if not os.path.isfile(targ_dir+para_map):
#                         shutil.copy(subj_dir+para_map, targ_dir)
                
#                 # if results file does not yet exist, perform analysis
#                 else:
                
#                     # run numerosity model
#                     try:
#                         mod = EMPRISE.Model(sub, ses, mod_name, space)
#                         mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'])
#                         shutil.copy(subj_dir+para_map, targ_dir)
#                     except FileNotFoundError:
#                         continue


### threshold surface images, 17/11/2023 ######################################

# Step 1: threshold tuning maps
# mod = EMPRISE.Model('003','visual','True_False_iid_1','fsnative')
# mod.threshold_maps('Rsqmb,0.2')
# mod = EMPRISE.Model('009','audio','True_False_iid_1','fsnative')
# mod.threshold_maps('Rsqmb,0.2')

# Step 2: cluster using SurfClust
# AFNI /data/hu_soch/ownCloud/MPI/EMPRISE/tools/EMPRISE/code/Shell/cluster_surface.sh 003 visual True_False_iid_1 space-fsnative_Rsq_thr-Rsqmb,0.2 ses-visual
# AFNI /data/hu_soch/ownCloud/MPI/EMPRISE/tools/EMPRISE/code/Shell/cluster_surface.sh 003 visual True_False_iid_1 space-fsnative_mu_thr-Rsqmb,0.2 ses-visual
# AFNI /data/hu_soch/ownCloud/MPI/EMPRISE/tools/EMPRISE/code/Shell/cluster_surface.sh 009 audio True_False_iid_1 space-fsnative_Rsq_thr-Rsqmb,0.2 ses-audio
# AFNI /data/hu_soch/ownCloud/MPI/EMPRISE/tools/EMPRISE/code/Shell/cluster_surface.sh 009 audio True_False_iid_1 space-fsnative_mu_thr-Rsqmb,0.2 ses-audio

# Step 3: visualize surface maps
# mod = EMPRISE.Model('003','visual','True_False_iid_1','fsnative')
# mod.visualize_maps(img='space-fsnative_mu_thr-Rsqmb,0.2_cls-SurfClust')
# mod = EMPRISE.Model('009','audio','True_False_iid_1','fsnative')
# mod.visualize_maps(img='space-fsnative_mu_thr-Rsqmb,0.2_cls-SurfClust')


### NumpRF tuning parameters, 09,12/10/2023 ###################################

# # define model
# sub   = '001'
# ses   = 'audio'
# model = 'True_False_iid_1'
# space = 'fsnative'
# thr   = 'Rsq,0.15'

# # threshold maps
# mod = EMPRISE.Model(sub, ses, model, space)
# mod.visualize_maps(thr)


### NumpRF empirical analysis, 05,09/10/2023 ##################################

# # define analyses
# # subs = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', 
# #         '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', 116']
# subs = ['001', '101', '002', '102']
# # sess = ['visual', 'audio', 'digits', 'spoken']
# sess = ['visual', 'audio']
# # spcs = ['fsnative', 'fsaverage']
# spcs = ['fsnative']
# mods = [{'avg': [True, False], 'noise': 'iid', 'hrfs': 1},
#         {'avg': [True, False], 'noise': 'ar1', 'hrfs': 3}]

# # perform analyses
# for sub in subs:
#     for ses in sess:
#         for space in spcs:
#             for model in mods:
                
#                 # determine data filename
#                 sess_obj = EMPRISE.Session(sub, ses)
#                 filename = sess_obj.get_bold_gii(1, 'L', space)
#                 del sess_obj
                
#                 # perform analysis, if preprocessed 4D GIfTI file exists
#                 if os.path.isfile(filename):
                    
#                     # determine results filename
#                     mod_name = str(model['avg'][0])+'_'+str(model['avg'][1])+'_'+str(model['noise'])+'_'+str(model['hrfs'])
#                     filename = EMPRISE.deri_out + 'numprf' + \
#                                '/sub-' + sub + '/ses-' + ses + '/model-' + mod_name + \
#                                '/sub-' + sub + '_ses-' + ses + '_model-' + mod_name + '_hemi-' + 'L' + '_space-' + space + '_numprf.mat'
                    
#                     # perform analysis, if results file does not yet exist
#                     if not os.path.isfile(filename):
                    
#                         # run numerosity model
#                         mod = EMPRISE.Model(sub, ses, mod_name, space)
#                         mod.analyze_numerosity(avg=model['avg'], corr=model['noise'], order=model['hrfs'])


### threshold surface images, 19/10/2023 ######################################

# # specify surface images
# mesh_img = r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\fmriprep\sub-001\ses-visual\anat\sub-001_ses-visual_acq-mprageised_hemi-L_inflated.surf.gii'
# surf_img = r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\numprf\sub-001\ses-visual\model-True_False_iid_1\sub-001_ses-visual_model-True_False_iid_1_hemi-L_space-fsnative_mu_thr-Rsq.surf.gii'
# res_img  = r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\numprf\sub-001\ses-visual\model-True_False_iid_1\sub-001_ses-visual_model-True_False_iid_1_hemi-L_space-fsnative_mu_thr-Rsq_cls-coords.surf.gii'
# # mesh_img = r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\fmriprep\sub-001\ses-visual\anat\sub-001_ses-visual_acq-mprageised_hemi-R_inflated.surf.gii'
# # surf_img = r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\numprf\sub-001\ses-visual\model-True_False_iid_1\sub-001_ses-visual_model-True_False_iid_1_hemi-R_space-fsnative_mu_thr-Rsq.surf.gii'
# # res_img  = r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\numprf\sub-001\ses-visual\model-True_False_iid_1\sub-001_ses-visual_model-True_False_iid_1_hemi-R_space-fsnative_mu_thr-Rsq_cls-coords.surf.gii'
# surf_gii = nib.load(surf_img)
# mesh_gii = nib.load(mesh_img)

# # load surface data
# y    = surf_gii.darrays[0].data
# v    = y.size
# # This is a v x 1 vector which contains vertex values as such.
# XYZ  = mesh_gii.darrays[0].data
# # This is a v x 3 matrix of vertex coordinates (MNI in mm?).
# vert = mesh_gii.darrays[1].data
# # This is a v x 3 matrix of triangle vectors (vertex indices).

# # specify clustering
# d_e   = 1                       # maximum distance of vertex to cluster [edges]
# d_mm  = 3                       # maximum distance of vertex to cluster [mm]
# n_min = 100                     # minimum number of vertices in cluster [1]
# ctype = 'coords'                # clustering type ("coords" or "edges")
# c     = 0                       # cluster index

# # perform clustering (spatial-distance-based)
# print('\n-> Step 0: preallocate clusters ... ', end='')
# clst = np.nan * np.ones(y.shape)
# print('successful!')

# # Step 1: assign clusters
# print('\n-> Step 1: assign clusters ... ', end='')
# for j in range(v):
#     if not np.isnan(y[j]) and y[j] != 0:
#         if ctype == 'coords':
#             XYZ_j  = XYZ[j,:]
#         elif ctype == 'edges':
#             vert_j = vert[np.any(vert==j, axis=1),:]
#         new_clust = True
#         for k in range(c):
#             if ctype == 'coords':
#                 dist_clust = np.sqrt( np.sum( (XYZ[clst==k,:] - XYZ_j)**2, axis=1 ) )
#                 conn_clust = dist_clust<d_mm
#             elif ctype == 'edges':
#                 # currently, this only implements d_e = 1
#                 conn_clust = [np.any(vert_j==l) for l in np.where(clst==k)[0]]
#             if np.any(conn_clust):
#                 new_clust = False
#                 clst[j] = k
#                 break
#         if new_clust:
#             c = c + 1
#             clst[j] = c
# print('successful!')
# print(np.unique(clst))

# # Step 2: merge clusters
# print('\n-> Step 2: merge clusters ... ', end='')
# for k1 in range(1,c+1):
#     for k2 in range(k1+1,c+1):
#         if ctype == 'coords':
#             XYZ_k1  = XYZ[clst==k1,:]
#         elif ctype == 'edges':
#             ind_k1  = np.sum(vert, axis=1) > np.inf
#             for j in np.where(clst==k1)[0]:
#                 ind_k1 = np.logical_or(ind_k1, np.any(vert==j, axis=1))
#             vert_k1 = vert[ind_k1,:]
#             del ind_k1
#         single_clust = False
#         for j in np.where(clst==k2)[0]:
#             if ctype == 'coords':
#                 dist_clust = np.sqrt( np.sum( (XYZ_k1 - XYZ[j,:])**2, axis=1 ) )
#                 conn_clust = dist_clust<d_mm
#             elif ctype == 'edges':
#                 # currently, this only implements d_e = 1
#                 conn_clust = np.any(vert_k1==j, axis=1)
#             if np.any(conn_clust):
#                 single_clust = True
#                 break
#         if single_clust:
#             clst[clst==k2] = k1
# print('successful!') 
# print(np.unique(clst))

# # Step 3: remove clusters
# print('\n-> Step 3: remove clusters ... ', end='')
# for k in range(1,c+1):
#     if np.sum(clst==k) < n_min:
#         clst[clst==k] = np.nan
# print('successful!')
# print(np.unique(clst))

# # save clustered image
# y_clust = np.nan * np.ones(y.shape)
# for k in range(1,c+1):
#     y_clust[clst==k] = y[clst==k]
# y_clust = y_clust.astype(np.float32)    
# EMPRISE.save_surf(y_clust, surf_gii, res_img)

# # visualize clustered images
# mod = EMPRISE.Model('001', 'visual', 'True_False_iid_1', 'fsnative')
# mod.visualize_maps(img='space-fsnative_mu_thr-Rsq_cls-coords')


### Session EDY7-visual: NumpRF/BayespRF results, 26/09/2023 ##################

# # specify filenames
# res_dir = r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\spm12\sub-EDY7\ses-visual\model-pRF'
# Np_res  = ['NumpRF_True_False_iid_1.mat', 'NumpRF_False_False_ar1_3.mat']
# Bp_res  = ['PRF_Analysis_1_0.mat', 'PRF_Analysis_1_1.mat', \
#            'PRF_Analysis_0_0.mat', 'PRF_Analysis_0_1.mat']

# # load parameter estimates (NumpRF)
# times    = []
# Res_Np   = []
# names_Np = ['True, False, iid, 1', 'False, False, ar1, 3']
# for h, res_file in enumerate(Np_res):
#     Ana = sp.io.loadmat(res_dir+'/'+res_file)
#     num_vox  = Ana['mu_est'].size
#     res_dict = {'mu_est':    np.squeeze(Ana['mu_est']), \
#                 'fwhm_est':  np.squeeze(Ana['fwhm_est']), \
#                 'MLL_est':   np.squeeze(Ana['MLL_est']), \
#                 'MLL_null':  np.squeeze(Ana['MLL_null']), \
#                 'MLL_const': np.squeeze(Ana['MLL_const']), \
#                 'k_est':     Ana['k_est'][0,0], \
#                 'k_null':    Ana['k_null'][0,0], \
#                 'k_const':   Ana['k_const'][0,0]}
#     Res_Np.append(res_dict)
#     times.append(np.squeeze(Ana['time'])/60)
#     names_Np[h] = 'NumpRF: {} ({})'.format(names_Np[h], num_vox)

# # load parameter estimates (BayespRF)
# Res_Bp   = []
# names_Bp = ['True, w/o', 'True, with', 'False, w/o', 'False, with']
# for h, res_file in enumerate(Bp_res):
#     Ana = sp.io.loadmat(res_dir+'/'+res_file)
#     num_vox  = Ana['PRF']['F'][0,0].size
#     mu_est   = np.zeros(num_vox)
#     fwhm_est = np.zeros(num_vox)
#     beta_est = np.zeros(num_vox)
#     for i in range(num_vox):
#         mu_log,sig_lat,beta_lat,transit,decay,epsilon = Ana['PRF']['Ep'][0,0][0,i][0,0]
#         mu_est[i], fwhm_est[i] = NumpRF.log2lin(mu_log[0,0], np.exp(sig_lat[0,0]))
#         beta_est[i]            = np.exp(beta_lat)
#     res_dict = {'mu_est':   mu_est, \
#                 'fwhm_est': fwhm_est, \
#                 'beta_est': beta_est}
#     Res_Bp.append(res_dict)
#     times.append(Ana['PRF']['est_time'][0,0][0,0]/60)
#     names_Bp[h] = 'BayespRF: {} ({})'.format(names_Bp[h], num_vox)
# del mu_log,sig_lat,beta_lat,transit,decay,epsilon

# # summarize simulation results
# del Ana, res_file, res_dict
# E     = len(times)
# methA = range(0,2)
# methB = range(2,E)
# times = np.array(times)
# names = names_Np
# names.extend(names_Bp)
# mu_thr   = [0.5, 5.5]
# fwhm_thr = [1, 15]

# # plot simulation results (computation time)
# fig, ax = plt.subplots(figsize=(16,9))
# ax.barh(np.arange(E), times, color='w', edgecolor='k')
# ax.set_yticks(np.arange(E), labels=names, fontsize=12)
# ax.axis([0, np.max(times)*(11/10), 0-1, E])
# ax.invert_yaxis() 
# ax.set_xlabel('estimation time [min]', fontsize=16)
# ax.set_ylabel('estimation settings', fontsize=16)
# ax.set_title('total computation time', fontsize=24)
# fig.savefig('Session_EDY7_visual_Fig_1.png', dpi=150)

# # plot simulation results (NumpRF estimates)
# fig, axs = plt.subplots(1, 2, figsize=(16,9))
# for h, R in enumerate(Res_Np):
#     axs[h].plot(R['mu_est'], R['fwhm_est'], '.b')
#     axs[h].vlines(np.array(mu_thr), -100, +100, colors='k', linestyles='dashed')
#     axs[h].hlines(np.array(fwhm_thr), -100, +100, colors='k', linestyles='dashed')
#     axs[h].axis([-0.5, 6.5, -0.5, 20.5])
#     axs[h].set_xlabel('estimated preferred numerosity', fontsize=16)
#     axs[h].set_ylabel('estimated tuning width', fontsize=16)
#     axs[h].set_title(names_Np[h], fontsize=24)
# fig.savefig('Session_EDY7_visual_Fig_2a.png', dpi=150)

# # plot simulation results (BayespRF estimates)
# fig, axs = plt.subplots(2, 2, figsize=(16,9))
# for i, axi in enumerate(axs):
#     for j, axj in enumerate(axi):
#         h = i*2+j
#         axj.plot(Res_Bp[h]['mu_est'], Res_Bp[h]['fwhm_est'], '.b')
#         axj.vlines(np.array(mu_thr), -100, +100, colors='k', linestyles='dashed')
#         axj.hlines(np.array(fwhm_thr), -100, +100, colors='k', linestyles='dashed')
#         axj.axis([-0.5, 6.5, -0.5, 20.5])
#         if i == 1 and j == 0:
#             axj.set_xlabel('estimated preferred numerosity', fontsize=16)
#             axj.set_ylabel('estimated tuning width', fontsize=16)
#         axj.set_title(names_Bp[h], fontsize=24)
# fig.savefig('Session_EDY7_visual_Fig_2b.png', dpi=150)


### Session EDY7-visual: NumpRF, 14/09/2023 ###################################

# # specify data
# subj_id = 'EDY7'
# sess_id = 'visual'
# res_dir =r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\spm12\sub-EDY7\ses-visual\model-pRF'

# # specify parameters
# avg1  = True
# avg2  = False
# noise = 'ar1'
# hrfs  = 3

# # load onsets
# print('\n-> Loading onsets ... ', end='')
# sess = EMPRISE.Session(subj_id, sess_id)
# ons, dur, stim = sess.get_onsets()
# ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
# TR             = EMPRISE.TR
# print('successful!')

# # load confounds
# print('-> Loading confounds ... ', end='')
# labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
#           'white_matter', 'csf', 'global_signal', \
#           'cosine00', 'cosine01', 'cosine02']
# X_c = sess.get_confounds(labels)
# X_c = EMPRISE.standardize_confounds(X_c)
# print('successful!')

# # load data
# print('-> Loading fMRI data ... ', end='')
# Y = sess.load_surf_data_all('L','fsnative')
# V = Y.shape[1]
# M = np.all(Y, axis=(0,2))
# Y = Y[:,M,:]
# print('successful!')

# # analyze data
# print('-> Estimating parameters ... ', end='\n')
# Y  = EMPRISE.standardize_signals(Y)
# ds = NumpRF.DataSet(Y, ons, dur, stim, TR, X_c)
# start_time = time.time()
# mu_est, fwhm_est, MLL_est, MLL_null, MLL_const =\
#     ds.estimate_MLE(avg=[avg1,avg2], meth='fgs', corr=noise, order=hrfs)
# k_est, k_null, k_const = ds.free_parameters(avg=[avg1,avg2], corr=noise, order=hrfs)
# end_time   = time.time()
# difference = end_time - start_time

# # save results (mu)
# mu_map   = np.zeros(V, dtype=np.float32)
# mu_map[M]= mu_est
# surface  = nib.load(sess.get_bold_gii(1,'L','fsnative'))
# filename = res_dir+'/'+'NumpRF_surf_'+str(avg1)+'_'+str(avg2)+'_'+str(noise)+'_'+str(hrfs)+'_mu.surf.gii'
# mu_img   = EMPRISE.save_surf(mu_map, surface, filename)

# # save results (fwhm)
# fwhm_map   = np.zeros(V, dtype=np.float32)
# fwhm_map[M]= fwhm_est
# surface    = nib.load(sess.get_bold_gii(1,'L','fsnative'))
# filename   = res_dir+'/'+'NumpRF_surf_'+str(avg1)+'_'+str(avg2)+'_'+str(noise)+'_'+str(hrfs)+'_fwhm.surf.gii'
# fwhm_img   = EMPRISE.save_surf(fwhm_map, surface, filename)

# # save results (MLL)
# res_dict = {'res_dir': res_dir, 'settings': [avg1, avg2, noise, hrfs], \
#             'mu_est': mu_est, 'fwhm_est': fwhm_est, \
#             'MLL_est': MLL_est, 'MLL_null': MLL_null, 'MLL_const': MLL_const, \
#             'k_est': k_est, 'k_null': k_null, 'k_const': k_const, \
#             'time': difference}
# filename = res_dir+'/'+'NumpRF_surf_'+str(avg1)+'_'+str(avg2)+'_'+str(noise)+'_'+str(hrfs)+'.mat'
# sp.io.savemat(filename, res_dict)


### Session EDY7-visual: NumpRF, 07/09/2023 ###################################

# # specify data
# subj_id = 'EDY7'
# sess_id = 'visual'
# spm_dir =r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\spm12\sub-EDY7\ses-visual\model-pRF'
# thr_spm = 'spmF_0001_unc_0.001_10.nii'

# # specify parameters
# avg1  = True
# avg2  = False
# noise = 'iid'
# hrfs  = 1

# # load onsets
# print('\n-> Loading onsets ... ', end='')
# sess = EMPRISE.Session(subj_id, sess_id)
# ons, dur, stim = sess.get_onsets()
# ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
# TR             = EMPRISE.TR
# print('successful!')

# # load confounds
# print('-> Loading confounds ... ', end='')
# labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
#           'white_matter', 'csf', 'global_signal', \
#           'cosine00', 'cosine01', 'cosine02']
# X_c = sess.get_confounds(labels)
# X_c = EMPRISE.standardize_confounds(X_c)
# print('successful!')

# # load data
# print('-> Loading fMRI data ... ', end='')
# M = sess.load_mask(1,'T1w')
# Y = sess.load_data_all('T1w')
# print('successful!')

# # load mask image
# print('-> Loading voxel image ... ', end='')
# filename = spm_dir+'/'+thr_spm
# spm_nii  = nib.load(filename)
# SPM = spm_nii.get_fdata()
# SPM = SPM.reshape((np.prod(M.shape),), order='C')
# print('successful!')

# # analyze data
# print('-> Estimating parameters ... ', end='\n')
# Y  = Y[:,np.logical_and(M==1,SPM==1),:]
# Y  = EMPRISE.standardize_signals(Y)
# ds = NumpRF.DataSet(Y, ons, dur, stim, TR, X_c)
# start_time = time.time()
# mu_est, fwhm_est, MLL_est, MLL_null, MLL_const =\
#     ds.estimate_MLE(avg=[avg1,avg2], meth='fgs', corr=noise, order=hrfs)
# k_est, k_null, k_const = ds.free_parameters(avg=[avg1,avg2], corr=noise, order=hrfs)
# end_time   = time.time()
# difference = end_time - start_time

# # save results (mu)
# mu_map   = SPM.copy()
# mu_map[np.logical_and(M==1,SPM==1)] = mu_est
# mu_map   = mu_map.reshape(spm_nii.shape, order='C')
# mu_img   = nib.Nifti1Image(mu_map, spm_nii.affine, spm_nii.header)
# filename = spm_dir+'/'+'NumpRF_'+str(avg1)+'_'+str(avg2)+'_'+str(noise)+'_'+str(hrfs)+'_mu.nii'
# nib.save(mu_img, filename)

# # save results (fwhm)
# fwhm_map = SPM.copy()
# fwhm_map[np.logical_and(M==1,SPM==1)] = fwhm_est
# fwhm_map = fwhm_map.reshape(spm_nii.shape, order='C')
# fwhm_img = nib.Nifti1Image(fwhm_map, spm_nii.affine, spm_nii.header)
# filename = spm_dir+'/'+'NumpRF_'+str(avg1)+'_'+str(avg2)+'_'+str(noise)+'_'+str(hrfs)+'_fwhm.nii'
# nib.save(fwhm_img, filename)

# # save results (MLL)
# res_dict = {'mask_img': sess.get_mask_nii(1,'T1w'), 'vox_img': thr_spm, \
#             'res_dir': spm_dir, 'settings': [avg1, avg2, noise, hrfs], \
#             'mu_est': mu_est, 'fwhm_est': fwhm_est, \
#             'MLL_est': MLL_est, 'MLL_null': MLL_null, 'MLL_const': MLL_const, \
#             'k_est': k_est, 'k_null': k_null, 'k_const': k_const, \
#             'time': difference}
# filename = spm_dir+'/'+'NumpRF_'+str(avg1)+'_'+str(avg2)+'_'+str(noise)+'_'+str(hrfs)+'.mat'
# sp.io.savemat(filename, res_dict)


### NumpRF empirical analysis, 24/08/2023 #####################################

# # load onsets
# print('\n-> Loading onsets ... ', end='')
# sess = EMPRISE.Session('EDY7', 'visual')
# ons, dur, stim = sess.get_onsets()
# ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
# print('successful!')

# # load confounds
# print('-> Loading confounds ... ', end='')
# labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
#           'white_matter', 'csf', 'global_signal', \
#           'cosine00', 'cosine01', 'cosine02']
# X_c = sess.get_confounds(labels)
# X_c = EMPRISE.standardize_confounds(X_c)
# print('successful!')

# # load data
# print('-> Loading fMRI data ... ', end='')
# M = sess.load_mask(1,'T1w')
# Y = sess.load_data_all('T1w')
# Y = Y[:,M==1,:]
# Y = EMPRISE.standardize_confounds(Y)
# print('successful!')

# # analyze data
# print('-> Estimating parameters ... ', end='\n')
# ds = NumpRF.DataSet(Y, ons, dur, stim, EMPRISE.TR, X_c)
# mu_est, fwhm_est = ds.estimate_MLE(avg=[True, False], meth='fgs', corr='ar1')


### NumpRF simulation analysis, 17/08/2023 ####################################

# # load simulation data
# filename = r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\Simulation_B.mat'
# Sim      = sp.io.loadmat(filename)

# # retrieve simulation settings
# ons, dur, stim = EMPRISE.get_onsets('EDY7', 'visual')
# ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
# X_c            = Sim['X'][:,1:13,:]
# mu_true        = np.squeeze(Sim['mu'])
# fwhm_true      = np.squeeze(Sim['fwhm'])

# # analyze simulation data
# ind = np.arange(0,Sim['Y'].shape[1])
# Y   = Sim['Y'][:,ind,:]
# avg = [False, False]
# start_time       = time.time()
# mu_est, fwhm_est = NumpRF.estimate_MLE_fgs(Y, ons, dur, stim, EMPRISE.TR, X_c, avg)
# # mu_est, fwhm_est = NumpRF.estimate_MLE_rgs(Y, ons, dur, stim, EMPRISE.TR, X_c, avg)
# end_time         = time.time()

# # save estimated parameters
# difference = end_time - start_time
# # file, ext  = os.path.splitext(filename)
# # filename   = file + '_NumpRF_avg_' + str(avg[0]) + '_' + str(avg[1]) + ext
# # res_dict   = {'mu_est': mu_est, 'fwhm_est': fwhm_est, 'time': difference}
# # sp.io.savemat(filename, res_dict)
# print(difference)


### NumpRF simulation results, 14/08/2023 #####################################

# # load simulation data
# filename  = r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\Simulation_B.mat'
# folder    = os.path.dirname(filename)
# file, ext = os.path.splitext(filename)
# Sim       = sp.io.loadmat(filename)
# mu_true   = np.squeeze(Sim['mu'])
# fwhm_true = np.squeeze(Sim['fwhm'])
# del Sim

# # load simulation results
# avg      = ['across runs and epochs', 'across runs only', 'across epochs only', 'none']
# mu_est   = []
# fwhm_est = []
# time_est = []
# avgs     = [True, False]
# for avg_run in avgs:
#     for avg_epoc in avgs:
#         filename  = file + '_NumpRF_avg_' + str(avg_run) + '_' + str(avg_epoc) + ext
#         Res       = sp.io.loadmat(filename)
#         mu_est.append(np.squeeze(Res['mu_est']))
#         fwhm_est.append(np.squeeze(Res['fwhm_est']))
#         time_est.append(np.squeeze(Res['time']))
# del Res
        
# # plot estimated parameters (NumpRF)
# fig1 = plt.figure(figsize=(24,13.5))
# axs  = fig1.subplots(2,2)
# for i, axi in enumerate(reversed(axs)):
#     for j, axj in enumerate(axi):
#         n = (i-1)*2+j
#         axj.plot(mu_est[n], fwhm_est[n], 'ob')
#         axj.axis([0, 8, 0, 16])
#         axj.set_xlabel('estimated preferred numerosity', fontsize=16)
#         axj.set_ylabel('estimated tuning width', fontsize=16)
#         axj.set_title('averaging: '+avg[n], fontsize=24)
# filename = folder + '/' + 'Simulation_B_NumpRF_Figure_1.png'
# fig1.savefig(filename, dpi=150)

# # plot estimated parameters (NumpRF mu vs. truth)
# fig2 = plt.figure(figsize=(24,13.5))
# axs  = fig2.subplots(2,2)
# for i, axi in enumerate(reversed(axs)):
#     for j, axj in enumerate(axi):
#         n = (i-1)*2+j
#         r = np.corrcoef(mu_true, mu_est[n])[0,1]
#         MAE = np.mean(np.abs(mu_true-mu_est[n]))
#         axj.plot(mu_true, mu_est[n], 'ob')
#         axj.axis([0, 8, 0, 8])
#         axj.set_xlabel('ground truth value', fontsize=16)
#         axj.set_ylabel('NumpRF estimate', fontsize=16)
#         axj.set_title('averaging: '+avg[n], fontsize=24)
#         axj.text(8, 7, 'r = {:.4f}   \nMAE = {:.4f}   '.format(r, MAE), fontsize=16, \
#                  horizontalalignment='right', verticalalignment='center')
# filename = folder + '/' + 'Simulation_B_NumpRF_Figure_2.png'
# fig2.savefig(filename, dpi=150)

# # plot estimated parameters (NumpRF fwhm vs. truth)
# fig3 = plt.figure(figsize=(24,13.5))
# axs  = fig3.subplots(2,2)
# for i, axi in enumerate(reversed(axs)):
#     for j, axj in enumerate(axi):
#         n = (i-1)*2+j
#         r = np.corrcoef(fwhm_true, fwhm_est[n])[0,1]
#         MAE = np.mean(np.abs(fwhm_true-fwhm_est[n]))
#         axj.plot(fwhm_true, fwhm_est[n], 'ob')
#         axj.axis([0, 12, 0, 12])
#         axj.set_xlabel('ground truth value', fontsize=16)
#         axj.set_ylabel('NumpRF estimate', fontsize=16)
#         axj.set_title('averaging: '+avg[n], fontsize=24)
#         axj.text(12, 1, 'r = {:.4f}   \nMAE = {:.4f}   '.format(r, MAE), fontsize=16, \
#                  horizontalalignment='right', verticalalignment='center')
# filename = folder + '/' + 'Simulation_B_NumpRF_Figure_3.png'
# fig3.savefig(filename, dpi=150)
# del i, j, n, axi, axj, axs, r


### NumpRF simulation analysis, 10/08/2023 ####################################

# # load simulation data
# filename = r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\Simulation_B.mat'
# Sim      = sp.io.loadmat(filename)

# # retrieve simulation settings
# ons, dur, stim = EMPRISE.get_onsets('EDY7', 'visual')
# ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
# X_c            = Sim['X'][:,1:13,:]
# mu_true        = np.squeeze(Sim['mu'])
# fwhm_true      = np.squeeze(Sim['fwhm'])

# # analyze simulation data
# ind = np.arange(0,Sim['Y'].shape[1])
# Y   = Sim['Y'][:,ind,:]
# avg = [False, False]
# start_time       = time.time()
# mu_est, fwhm_est = NumpRF.estimate_MLE_rgs(Y, ons, dur, stim, EMPRISE.TR, X_c, avg)
# end_time         = time.time()

# # save estimated parameters
# file, ext  = os.path.splitext(filename)
# filename   = file + '_NumpRF_avg_' + str(avg[0]) + '_' + str(avg[1]) + ext
# difference = end_time - start_time
# res_dict   = {'mu_est': mu_est, 'fwhm_est': fwhm_est, 'time': difference}
# sp.io.savemat(filename, res_dict)


### NumpRF empirical analysis, 07/08/2023 #####################################

# # set subject and session
# sub = 'EDY7'
# ses = 'visual'
# v   = 717

# # load measured data
# data_dir = r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\spm12\sub-EDY7\ses-visual\model-pRF'
# Y = np.zeros((EMPRISE.n, v, len(EMPRISE.runs)))
# for j in range(len(EMPRISE.runs)):
#     filename = 'VOI_model-pRF_run-' + str(j+1) + '_timeseries.mat'
#     Data     = sp.io.loadmat(data_dir+'/'+filename)
#     Y[:,:,j] = Data['xY']['y'][0][0]
# del Data, v

# # get experimental design
# ons, dur, stim = EMPRISE.get_onsets(sub, ses)
# ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')

# # get confound variables
# labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
#           'white_matter', 'csf', 'global_signal', \
#           'cosine00', 'cosine01', 'cosine02']
# X_c    = EMPRISE.get_confounds(sub, ses, labels)
# for j in range(X_c.shape[2]):
#     for k in range(X_c.shape[1]):
#         X_c[:,k,j] = X_c[:,k,j] - np.mean(X_c[:,k,j])
#         X_c[:,k,j] = X_c[:,k,j] / np.max(X_c[:,k,j])

# # analyze measured data
# avg = [False, False]
# mu_est, fwhm_est = NumpRF.estimate_MLE_rgs(Y, ons, dur, stim, EMPRISE.TR, X_c, avg)

# # save estimation results
# filename = data_dir+'/'+'NumpRF.mat'
# res_dict = {'Y': Y, 'X_c': X_c, 'mu_est': mu_est, 'fwhm_est': fwhm_est, \
#             'settings': {'sub': sub, 'ses': ses, 'labels': labels, \
#                          'onsets': ons, 'durations': dur, 'stimuli': stim}}
# sp.io.savemat(filename, res_dict)


### NumpRF simulation analysis, 14/07/2023 ####################################

# # load simulation data
# filename = r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\simulated_data\Simulation_B.mat'
# Sim      = sp.io.loadmat(filename)

# # retrieve simulation settings
# ons, dur, stim = EMPRISE.get_onsets('EDY7', 'visual')
# ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
# X_c            = Sim['X'][:,1:13,:]
# mu_true        = np.squeeze(Sim['mu'])
# fwhm_true      = np.squeeze(Sim['fwhm'])

# # analyze simulation data
# ind = np.arange(0,Sim['Y'].shape[1])
# Y   = Sim['Y'][:,ind,:]
# avg = [False, False]
# mu_est, fwhm_est = NumpRF.estimate_MLE_rgs(Y, ons, dur, stim, EMPRISE.TR, X_c, avg)
# print('-> r(mu,   mu_est)   = {:.4f}'.format(np.corrcoef(mu_true[ind], mu_est)[0,1]))
# print('-> r(fwhm, fwhm_est) = {:.4f}'.format(np.corrcoef(fwhm_true[ind], fwhm_est)[0,1]))


### pilot analyses, 31/05/2023 ################################################

# # import modules
# import numpy as np
# import nibabel as nib

# # specify filename
# filename = r'C:\Joram\projects\MPI\EMPRISE\data\derivatives\fmriprep\sub-EDY7\ses-visual\func\sub-EDY7_ses-visual_task-harvey_acq-fMRI1p75TE24TR2100iPAT3FS_run-1_space-T1w_desc-preproc_bold.nii.gz'
# print(filename)

# # load images
# bold_nii = nib.load(filename)
# print(bold_nii.header)
# Y         = bold_nii.get_fdata()
# Y_shape   = Y.shape
# Y_reshape = Y.reshape((Y_shape[-1], np.prod(Y_shape[0:-1])), order='C')

# # nibabel documentation
# # https://nipy.org/nibabel/nifti_images.html
# # https://nilearn.github.io/dev/modules/generated/nilearn.image.load_img.html