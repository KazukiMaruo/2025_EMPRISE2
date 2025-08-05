"""
Figures - Figures within the EMPRISE project (Revision)

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-05-28, 15:45: inherited from "Figures_V1.py"
2024-05-28, 16:16: WP1_Fig1
2024-06-04, 15:47: WP1_Fig2
2024-06-19, 14:59: WP1_Fig3
2024-06-19, 15:27: WP1_Fig4
2024-06-25, 19:57: WP1_Fig2
2024-06-27, 13:18: WP1_Fig2
2024-07-10, 15:59: WP1_Fig4
2024-07-18, 15:34: WP1_Ana2
2024-07-24, 12:39: WP1_Ana2
2024-07-24, 13:48: WP1_Fig2
2024-07-24, 15:43: WP1_Fig1
2024-07-25, 17:58: WP1_Fig3, WP1_Fig4
2024-07-31, 18:12: WP1_Fig1/2/3/4, WP1_Tab1/2
2024-08-01, 18:28: WP1_Fig4
2024-08-01, 19:21: WP1_Fig5
2024-08-02, 16:53: WP1_Fig4
2024-08-02, 17:02: WP1_Fig1, WP1_Fig5
2024-09-11, 16:09: WP1_Fig6
2024-09-16, 15:58: WP1_Fig4
2024-09-27, 09:15: renamed "Figures_Rev" to "Figures_WP1"
2024-09-27, 09:24: WP1_Fig4, WP1_Fig5
"""


# import packages
#-----------------------------------------------------------------------------#
import os
import math
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
from nilearn import surface
from surfplot import Plot
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import PySPMs as PySPM
import NumpRF
import EMPRISE

# specify default model
#-----------------------------------------------------------------------------#
model_def = 'model-NumAna'

# specify results directory
#-----------------------------------------------------------------------------#
if EMPRISE.at_MPI:
    res_dir = '/data/pt_02495/emprise7t_2_analysis/derivatives/numprf/'
else:
    res_dir = r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\NumpRF_results/'

# define numerosity maps
#-----------------------------------------------------------------------------#
hemis= ['L', 'R']
maps = {'visual': {'labels': ['NTO', 'NPO', 'NPC1', 'NPC2', 'NPC3', 'NF'], \
                  # Source: Harvey & Dumoulin (2017), pp. 1-2
                   'mean'  : {'L': np.array([[-42, -77,  -3], \
                                             [-23, -80,  32], \
                                             [-22, -59,  61], \
                                             [-38, -43,  48], \
                                             [-48, -29,  34], \
                                             [-22, -11,  50]]), \
                              'R': np.array([[ 44, -75,  -4], \
                                             [ 25, -82,  34], \
                                             [ 22, -61,  60], \
                                             [ 33, -40,  52], \
                                             [ 45, -30,  40], \
                                             [ 24, -11,  52]])}, \
                   'std'   : {'L': np.array([[  3,   3,   8], \
                                             [  4,   5,   7], \
                                             [  4,  11,   8], \
                                             [  3,   8,   8], \
                                             [  6,   5,   6], \
                                             [  3,   6,   8]]), \
                              'R': np.array([[  7,   1,   3], \
                                             [  5,   4,   6], \
                                             [  5,   7,   5], \
                                             [  3,   4,   7], \
                                             [ 10,   6,   4], \
                                             [  3,   5,   6]])}}, \
        'audio' : {'labels': ['NaT', 'NaF'], \
                  # "Numerosity, auditory, temporal/frontal"
                   'mean'  : {'L': np.array([[np.nan, -30,  10], \
                                             [np.nan,  -5,  55]]), \
                              'R': np.array([[np.nan, -30,  10], \
                                             [np.nan,   0,  50]])}, \
                   'std'   : {'L': np.array([[np.nan, np.nan, np.nan], \
                                             [np.nan, np.nan, np.nan]]), \
                              'R': np.array([[np.nan, np.nan, np.nan], \
                                             [np.nan, np.nan, np.nan]])}}}

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

# function: simple linear regression
#-----------------------------------------------------------------------------#
def simplinreg(y, x):
    """
    Simple Linear Regression with Correlation
    r, p, m, n = simplinreg(y, x)
    
        y - n x 1 array; vector of observations
        x - n x 1 array; vector of predictions
        
        r - float; Pearson correlation between x and y
        p - float; p-value of the correlation coefficient
        m - float; slope of the regression line
        n - float; intercept of the regression line
    
    r, p, m, n = simplinreg(y, x) performs a simple linear regression of
    observations y on predictors x and returns correlation coefficient r,
    p-value p, slope and intercept of the regression line m and n.
    """
    
    # run linear regression
    slope, intercept, corr, pval, stderr = sp.stats.linregress(x, y)
    
    # return parameter estimates
    r = corr
    p = pval
    m = slope
    n = intercept
    return r, p, m, n

# function: calculate surface area
#-----------------------------------------------------------------------------#
def calc_surface_area(verts, trias):
    """
    areas = calc_surface_area(verts, trias)
    
        verts - v x 9 array of float; vertex properties (see "EMPRISE.threshold_and_cluster")
        trias - t x 3 array of int; triangle vertices (see "EMPRISE.threshold_and_cluster")
        
        areas - v x 7 array of float; supra-threshold triangles with area
        o 1st           column: triangle index
        o 2nd           column: cluster index
        o 3rd, 4th, 5th column: average mu, fwhm, beta
        o 6th           column: average R-squared
        o 7th           column: triangle area
    
    areas = calc_surface_area(verts, trias) takes vertex and triangle
    information, as provided by "EMPRISE.threshold_and_cluster" and returns
    all triangles that fully consist of supra-threshold vertices, including
    their surface area.
    
    Note: coordinates are a v x 3 array and triangles are a t x 3 array.
    Source: https://nben.net/MRI-Geometry/#surface-geometry-data
    """

    # get vertex indices
    verts_ind = verts[:,0].astype(np.int32)
    areas     = np.zeros((0,7))
    
    # cycle through triangles
    for j in range(trias.shape[0]):
        
        # if all three vertices are above threshold,
        # then the triangle is above threshold
        if trias[j,0] in verts_ind:
            if trias[j,1] in verts_ind:
                if trias[j,2] in verts_ind:
                    
                    # get all three vertices
                    ABC = np.r_[verts[verts_ind==trias[j,0],:], \
                                verts[verts_ind==trias[j,1],:], \
                                verts[verts_ind==trias[j,2],:]]
                    
                    # calculate triangle area
                    AB    = ABC[1,6:9] - ABC[0,6:9]
                    AC    = ABC[2,6:9] - ABC[0,6:9]
                    ABxAC = np.cross(AB, AC)
                    A     = np.sqrt(np.sum(np.power(ABxAC,2)))/2
                    # See: https://math.stackexchange.com/a/128999/480910
                    del AB, AC, ABxAC
                    
                    # collect triangle information
                    areas = np.r_[areas, \
                                  np.array([[j, ABC[0,1], \
                                             np.mean(ABC[:,2]), np.mean(ABC[:,3]), np.mean(ABC[:,4]), \
                                             np.mean(ABC[:,5]), A]])]
                    del ABC
    
    # return triangle information
    return areas

# function: filter supra-threshold clusters
#-----------------------------------------------------------------------------#
def filter_clusters(verts, areas, A_min=10, d_max=10, nmap='', hemi=''):
    """
    verts, areas = filter_clusters(verts, areas, A_min, d_max, nmap, hemi)
    
        verts - v x 9 array of float; vertex properties (see "EMPRISE.threshold_and_cluster")
        areas - v x 7 array of float; triangle properties (see "calc_surface_area")
        A_min - float; minimum cluster size [mm²]
        d_max - float; maximum distance to map [mm]
        nmap  - string; numerosity map identifier (see "maps[...]['labels']")
        hemi  - string; brain hemisphere identifier (e.g. "L")
        
        verts - w x 9 array of float; filtered vertices
        areas - w x 7 array of float; filtered triangles
    
    verts, areas = filter_clusters(verts, areas, A_min, d_max, nmap, hemi)
    calculates surface areas for all clusters in verts and areas and removes
    clusters which are smaller than A_min or further than d_max away from
    numerosity map nmap in hemisphere hemi.
    
    Note: "d_max" is only applied, if "nmap" and "hemi" are non-empty.
    If "nmap" is a modality (e.g. 'visual' or 'audio'), the clusters close
    to all maps from this modality are filtered.
    """
    
    # get number of clusters
    if verts.shape[0] > 0:
        num_clst = int(np.max(verts[:,1]))
    else:
        num_clst = 0
    cls_inds = np.array(range(1,num_clst+1))
    
    # filter clusters for area
    for i in cls_inds:
        
        # calculate area
        verts_cls = verts[:,1].astype(np.int32)
        areas_cls = areas[:,1].astype(np.int32)
        A = np.sum(areas[areas_cls==i,6])
        
        # remove cluster, if cluster area < minimum area
        if A < A_min:
            verts = verts[verts_cls!=i,:]
            areas = areas[areas_cls!=i,:]
    
    # assign to numerosity map
    if nmap and hemi:
        
        # get session and index
        if nmap in EMPRISE.sess:
            ses = nmap
            ind = [imap for imap in range(len(maps[ses]['labels']))]
        elif nmap in maps['visual']['labels']:
            ses = 'visual'
            ind = maps[ses]['labels'].index(nmap)
        elif nmap in maps['audio']['labels']:
            ses = 'audio'
            ind = maps[ses]['labels'].index(nmap)
        else:
            vis_maps = ','.join(maps['visual']['labels'])
            aud_maps = ','.join(maps['audio']['labels'])
            err_msg  = 'Unknown numerosity map: "{}". Map must be one of [{}] or [{}].'
            raise ValueError(err_msg.format(nmap, vis_maps, aud_maps))
        
        # if indices are a list
        if type(ind) == list:
            
            # check all numerosity maps
            verts_all = np.zeros((0,verts.shape[1]))
            areas_all = np.zeros((0,areas.shape[1]))
            for imap in ind:
                
                # filter clusters for this map
                verts_new, areas_new = filter_clusters(verts, areas, A_min, d_max, maps[ses]['labels'][imap], hemi)
                
                # add cluster, if not yet filtered
                verts_all_cls = verts_all[:,1].astype(np.int32)
                verts_new_cls = verts_new[:,1].astype(np.int32)
                areas_new_cls = areas_new[:,1].astype(np.int32)
                for i in cls_inds:
                    if i not in verts_all_cls:
                        verts_all = np.r_[verts_all, verts_new[verts_new_cls==i,:]]
                        areas_all = np.r_[areas_all, areas_new[areas_new_cls==i,:]]
            
            # store filtered clusters
            verts = verts_all
            areas = areas_all
            del verts_all, areas_all, verts_new, areas_new
        
        # if index is an integer
        elif type(ind) == int:
            
            # filter clusters for distance
            for i in cls_inds:
                
                # if cluster still exists
                verts_cls = verts[:,1].astype(np.int32)
                areas_cls = areas[:,1].astype(np.int32)
                if np.sum(verts_cls==i) > 0:
                
                    # calculate distance
                    if ses == 'visual':
                        c = maps['visual']['mean'][hemi][ind,:]
                        E = verts[verts_cls==i,6:9] - np.tile(c, (np.sum(verts_cls==i),1))
                        d = np.sqrt(np.sum(np.power(E, 2), axis=1))
                        d = np.min(d)
                    if ses == 'audio':
                        c = maps['audio']['mean'][hemi][ind,:]
                        E = verts[verts_cls==i,7:9] - np.tile(c[1:3], (np.sum(verts_cls==i),1))
                        E = np.concatenate((np.mean(np.matrix(E), axis=1), E), axis=1)
                        d = np.sqrt(np.sum(np.power(E, 2), axis=1))
                        d = np.min(d)
                    
                    # remove cluster, if cluster distance > maximum distance
                    if d > d_max:
                        verts = verts[verts_cls!=i,:]
                        areas = areas[areas_cls!=i,:]
    
    # return clusters
    return verts, areas

# function: Work Package 1, Table 1
#-----------------------------------------------------------------------------#
def WP1_Tab1():

    # read participant table
    filename  = EMPRISE.data_dir+'participants.tsv'
    part_info = pd.read_table(filename)
    subs      = EMPRISE.adults
    sess      = ['visual', 'audio']
    
    # preallocate table columns
    gender          = []
    age             = []
    hit_rates       = np.zeros((len(subs),len(EMPRISE.runs),len(sess)))
    visual_hit_rate = []
    audio_hit_rate  = []
    
    # collect subject information
    for i, sub in enumerate(subs):
        
        # get subject info
        sub_info = part_info[part_info['participant_id']=='sub-'+sub]
        gender.append(sub_info.iloc[0,2])
        age.append(np.mean(np.unique(sub_info['age'])))
        
        # for both sessions
        for k, ses in enumerate(sess):
            
            # for all runs
            for j, run in enumerate(EMPRISE.runs):
                
                # load logfile
                filename = EMPRISE.Session(sub, ses).get_events_tsv(run)
                events   = pd.read_csv(filename, sep='\t')
                
                # analyze all trials
                num_catch = 0
                num_resp  = 0
                for t in range(len(events)):
                    if events['trial_type'][t].find('_attn') > -1:
                        num_catch  = num_catch + 1
                        curr_time  = events['onset'][t]
                        time_after = events[(events['onset']>(curr_time+0.1)) & (events['onset']<(curr_time+2.1))]
                        if sum(time_after['trial_type']!='button_press') > 0:
                            num_resp = num_resp + 1
                
                # calculate hit rates
                print('Subject {}, Session {}, Run {}: {} catch trials, {} responses.'. \
                      format(sub, ses, run, num_catch, num_resp))
                hit_rates[i,j,k] = (num_resp/num_catch)*100
        
        # store hit rates
        visual_hit_rate.append(np.mean(hit_rates[i,:,0]))
        audio_hit_rate.append(np.mean(hit_rates[i,:,1]))
    
    # create data frame
    data = zip(subs, gender, age, visual_hit_rate, audio_hit_rate)
    cols = ['Subject_ID', 'gender', 'age', \
            'visual hit rate', 'auditory hit rate']
    df   = pd.DataFrame(data, columns=cols)
    
    # save data frame to CSV/XLS
    df.to_csv('Figures_WP1/WP1_Table_1_sub-all.csv', index=False)
    df.to_excel('Figures_WP1/WP1_Table_1_sub-all.xlsx', index=False)

# function: Work Package 1, Table 2
#-----------------------------------------------------------------------------#
def WP1_Tab2():
    
    # specify analysis
    sub_group  =[EMPRISE.adults, EMPRISE.childs]
    sess  =['visual']
    space = 'fsnative'
    mesh  = 'pial'
    
    # preallocate statistics
    df_ses  = []
    df_dep  = []
    df_ind  = []
    df_coef = []
    df_z    = []
    df_pval = []
    df_ci1  = []
    df_ci2  = []
    
    # for all sub group
    for gr in sub_group:
        for ses in sess:
        
            # load results
            if '00' in gr[0]:
                filepath = res_dir+'sub-adults'+'_ses-'+ses+'_space-'+space+'_mesh-'+mesh+'_AFNI_'
            else:
                filepath = res_dir+'sub-childs'+'_ses-'+ses+'_space-'+space+'_mesh-'+mesh+'_AFNI_'
            verts    = sp.io.loadmat(filepath+'verts.mat')
            areas    = sp.io.loadmat(filepath+'areas.mat')
            
            # specify mu grid
            d_mu   = 0.5
            mu_min = EMPRISE.mu_thr[0]
            mu_max = EMPRISE.mu_thr[1]
            mu_b   = np.arange(mu_min, mu_max+d_mu, d_mu)
            mu_c   = np.arange(mu_min+d_mu/2, mu_max+d_mu/2, d_mu)
            
            # preallocate data
            df_subs  = []
            df_hemis = []
            df_mus   = []
            df_fwhms = []
            df_areas = []
            
            # for all subjects
            for sub in gr:
                
                # for both hemispheres
                for j, hemi in enumerate(hemis):
                    
                    # get supra-threshold vertices/triangles
                    verts_sub = verts[sub][hemi][0,0]
                    areas_sub = areas[sub][hemi][0,0]
                    # verts_sub, areas_sub = \
                    #     filter_clusters(verts_sub, areas_sub, A_min[ses])
                    mu_vert   = verts_sub[:,2]
                    fwhm_vert = verts_sub[:,3]
                    mu_area   = areas_sub[:,2]
                    area_area = areas_sub[:,6]
                    
                    # if supra-threshold vertices exist
                    if verts_sub.shape[0] > 0:
                    
                        # go through numerosity bins
                        for k in range(mu_c.size):
                            
                            # extract tuning width
                            ind_k = np.logical_and(mu_vert>mu_b[k],mu_vert<mu_b[k+1])
                            if np.sum(ind_k) > 0:
                                fwhm_m = np.mean(fwhm_vert[ind_k])
                            else:
                                fwhm_m = np.nan
                            
                            # extract surface area
                            ind_k = np.logical_and(mu_area>mu_b[k],mu_area<mu_b[k+1])
                            if np.sum(ind_k) > 0:
                                area_s = np.sum(area_area[ind_k])
                            else:
                                area_s = 0
                            
                            # store into data frame
                            if not np.isnan(fwhm_m): # area_s > 0:
                                df_subs.append(sub)
                                df_hemis.append(hemi)
                                df_mus.append(mu_c[k])
                                df_fwhms.append(fwhm_m)
                                df_areas.append(area_s)
            
            # create data frame
            data = zip(df_subs, df_hemis, df_mus, df_fwhms, df_areas)
            cols = ['sub', 'hemi', 'mu', 'fwhm', 'area']
            Y    = pd.DataFrame(data, columns=cols)
            
            # run linear mixed model (area vs. mu)
            lmm = smf.mixedlm('area ~ mu:hemi + hemi', Y, groups=Y['sub'])
            lmm = lmm.fit()
            print(lmm.summary())
            
            # store statistical results (area vs. mu)
            results = lmm.summary().tables[1]
            paras   = {'hemisphere': 'hemi[T.R]', 'mu_left': 'mu:hemi[L]', 'mu_right': 'mu:hemi[R]'}
            for para in paras.keys():
                res = results.loc[paras[para]]
                if para == 'hemisphere':
                    df_ses.append(ses)
                    df_dep.append('area')
                else:
                    df_ses.append('')
                    df_dep.append('')
                df_ind.append(para)
                df_coef.append(float(res.loc['Coef.']))
                df_z.append(float(res.loc['z']))
                df_pval.append(float(res.loc['P>|z|']))
                df_ci1.append(float(res.loc['[0.025']))
                df_ci2.append(float(res.loc['0.975]']))
            
            # run linear mixed model (fwhm vs. mu)
            lmm = smf.mixedlm('fwhm ~ mu:hemi + hemi', Y, groups=Y['sub'])
            lmm = lmm.fit()
            print(lmm.summary())
            
            # store statistical results (fwhm vs. mu)
            results = lmm.summary().tables[1]
            paras   = {'hemisphere': 'hemi[T.R]', 'mu_left': 'mu:hemi[L]', 'mu_right': 'mu:hemi[R]'}
            for para in paras.keys():
                res = results.loc[paras[para]]
                if para == 'hemisphere':
                    df_ses.append(ses)
                    df_dep.append('fwhm')
                else:
                    df_ses.append('')
                    df_dep.append('')
                df_ind.append(para)
                df_coef.append(float(res.loc['Coef.']))
                df_pval.append(float(res.loc['P>|z|']))
                df_z.append(float(res.loc['z']))
                df_ci1.append(float(res.loc['[0.025']))
                df_ci2.append(float(res.loc['0.975]']))
        
        # create data frame
        data = zip(df_ses, df_dep, df_ind, df_coef, df_z, df_pval, df_ci1, df_ci2)
        cols = ['Session','dependent variable','independent variable',\
                'coefficient','z-value','p-value','95% CI (lower)','95% CI (upper)']
        df   = pd.DataFrame(data, columns=cols)
        
        # save data frame to CSV/XLS
        if '00' in gr[0]:
            df.to_csv('/data/pt_02495/EMPRISE7t_analysis/Figures/WP1_Table_2_sub-adults'+'_ses-visual.csv', index=False)
        else:
            df.to_csv('/data/pt_02495/emprise7t_2_analysis/Figures/WP1_Table_2_sub-childs'+'_ses-visual.csv', index=False)
        #df.to_excel('Figures_WP1/WP1_Table_2_sub-all_ses-both.xlsx', index=False)

# function: Work Package 1, Table 3
#-----------------------------------------------------------------------------#
def WP1_Tab3(sub_group, ses, model, hemi, space, cv=True, verts=[0,0], save= False):
    # Create a list of the maximum Rsq and its fwhm of subjects
    # here, we used model values before it cluster thresholded

    # prepare a list of sub, maximum rsq, its fwhm
    verts = input('Vertices: ')
    verts = [int(i) for i in verts.split(',')]
    
    if len(verts) ==1: # the maximum vertex value for 1<mu<5
        # column names
        col = ['Sub','Rsq','MeanRsq','FWHM']
    else: # the maximum vertices for 1<mu<2 and 4<mu<5
        col = ['Sub','Rsq_m12','MeanRsq_m12','FWHM_12', 'Rsq_m45','MeanRsq_m45','FWHM_45']
        
    lst = []
    
    for sub in sub_group:   
        # sublist of subject 
        sub_lst = [sub]
        
        # load session data
        mod            = EMPRISE.Model(sub, ses, model, space)
        Y, M           = mod.load_mask_data(hemi)

        # load model results
        res_file = mod.get_results_file(hemi)
        NpRF     = sp.io.loadmat(res_file)
        mu       = np.squeeze(NpRF['mu_est'])
        fwhm     = np.squeeze(NpRF['fwhm_est'])
        beta     = np.squeeze(NpRF['beta_est'])
        
        # load vertex-wise coordinates
        hemis     = {'L': 'left', 'R': 'right'}
        para_map  = res_file[:res_file.find('numprf.mat')] + 'mu.surf.gii'
        para_mask = nib.load(para_map).darrays[0].data != 0
        mesh_file = mod.get_mesh_files(space,'pial')[hemis[hemi]]
        mesh_gii  = nib.load(mesh_file)
        XYZ       = mesh_gii.darrays[0].data
        XYZ       = XYZ[para_mask,:]
        del para_map, mesh_file, mesh_gii
        
        # if CV, load cross-validated R^2
        if cv:
            Rsq_map = res_file[:res_file.find('numprf.mat')] + 'cvRsq.surf.gii'
            cvRsq   = nib.load(Rsq_map).darrays[0].data
            Rsq     = cvRsq[para_mask]
            del Rsq_map, para_mask, cvRsq
            
        # otherwise, calculate total R^2
        else:
            MLL1 = np.squeeze(NpRF['MLL_est'])
            MLL0 = np.squeeze(NpRF['MLL_const'])
            n    = np.prod(mod.calc_runs_scans())
            Rsq  = NumpRF.MLL2Rsq(MLL1, MLL0, n)
        
        # select vertices 
        vert_lst = []
        for k, vertex in enumerate(verts):
            if vertex == 0:
                if k == 0 and len(verts) == 1:
                    # select maximum R^2 in vertices with numerosity 1 < mu < 5
                    vertex = np.argmax(Rsq + np.logical_and(mu>1, mu<5)+ (beta>0))
                elif k == 0 and len(verts) == 2:
                    # select maximum R^2 in vertices with numerosity 1 < mu < 2
                    vertex = np.argmax(Rsq + np.logical_and(mu>1, mu<2)+ (beta>0))
                elif k == 1:
                    # select maximum R^2 in vertices with numerosity 4 < mu < 5 or 3 < mu < 5
                    if ses == 'visual': vertex = np.argmax(Rsq + np.logical_and(mu>4, mu<5) + (fwhm<EMPRISE.fwhm_thr[1]) + (beta>0))
                    #if ses == 'audio':  vertex = np.argmax(Rsq + np.logical_and(mu>3, mu<5) + (fwhm<EMPRISE.fwhm_thr[1]) + (beta>0))
                
                vert_lst.append(vertex)
                
            Rsq_m = np.mean(Rsq)
            sub_lst.extend([Rsq[vert_lst[k]],Rsq_m,fwhm[vert_lst[k]]])
        
        # add it to lst
        lst.append(sub_lst)
    
    if save:
        # save values as csv
        df = pd.DataFrame(lst, columns = col)
        filename = '/data/pt_02495/emprise7t_2_analysis/Figures/Tables/WP1_Tab3_Rsq_Fwhm'+'_sub-'+str(sub_group)+'_hemi-'+hemi+'.csv'
        df.to_csv(filename, index=False)
    
    return lst
    
    
# function: Work Package 1, Supplement Table 1
#-----------------------------------------------------------------------------#
def WP1_S_Tab1():
    # extract the mean of supra th cvRsq within individual, for the both hemispheres
    # here, we used AFNI clustered cvRsq value.
    
    sess = ['visual']
    space = 'fsnative' 
    model = 'True_False_iid_1_V2_new'
    mesh = 'pial'

    
    # specify analysis
    
    sub_group  =[EMPRISE.adults, EMPRISE.childs]
    
    # for all sub group
    for gr in sub_group:
        print(f"subjects {gr} is being processed.")
        for ses in sess:
            print(f"session {ses} is being processed.")
            # load results
            if '00' in gr[0]:
                filepath = res_dir+'sub-adults'+'_ses-'+ses+'_space-'+space+'_mesh-'+mesh+'_AFNI_'
            else:
                filepath = res_dir+'sub-childs'+'_ses-'+ses+'_space-'+space+'_mesh-'+mesh+'_AFNI_'
            verts    = sp.io.loadmat(filepath+'verts.mat')
            #areas    = sp.io.loadmat(filepath+'areas.mat')
            print(f'AFNI clusted group map loaded.')
            # preallocate data
            df_subs       = []
            df_hemis      = []
            df_rsq        = []
            df_both       = [] # mean of Rsq for the both hemi
            df_max        = []
            df_max_mu     = []
            df_tsnr       = []
            df_mean_tsnr  = []
            df_both_tmean = [] # mean of tSNR for the both hemi
            # for all subjects
            for sub in gr:
                print(f'subject {sub}:')
                # AFNI clustered file containing all subjects in the given group
                verts_both = []
                tsnr_both  = []
                
                # for both hemispheres
                for j, hemi in enumerate(hemis):
                    print(f'hemi {hemi}:')
                    # load the tSNR data of the subject sub
                    filepath2  = res_dir + 'sub-' + sub + '/ses-' + ses + \
                                     '/model-' + model+'/sub-' + sub + '_ses-' + ses + \
                                     '_model-' + model + '_hemi-' + hemi + '_space-' + space + '_'
                    
                    mu_map   = filepath2 + 'mu.surf.gii'
                    mask     = nib.load(mu_map).darrays[0].data != 0  
                    tSNR_map = filepath2 + 'tSNR.surf.gii'
                    tSNR_map = nib.load(tSNR_map).darrays[0].data
                    tSNR_map = tSNR_map[mask]
                    
                    print('tSNR loaded ans masked:')
                    
                    # get supra-threshold vertices
                    verts_sub = verts[sub][hemi][0,0]
                    mu_vert   = verts_sub[:,2]
                    rsq_vert  = verts_sub[:,5]
                    ind_vert  = verts_sub[:,1]
                    
                    # if supra-threshold vertices exist
                    if verts_sub.shape[0] > 0:
                        verts_both.append(rsq_vert)
                        tsnr_both.append(tSNR_map)
                        rsq_m       = np.mean(rsq_vert)
                        rsq_max     = np.max(rsq_vert)
                        ind         = np.argmax(rsq_vert)
                        max_mu      = mu_vert[ind]
                        tSNR        = tSNR_map[int(ind_vert[ind])]
                        tSNR_mean   = np.mean(tSNR_map)
                        
                        # store into data frame
                        if not np.isnan(rsq_m): # area_s > 0:
                            df_subs.append(sub)
                            df_hemis.append(hemi)
                            df_rsq.append(rsq_m)
                            df_max.append(rsq_max)
                            df_max_mu.append(max_mu)
                            df_tsnr.append(tSNR)
                            df_mean_tsnr.append(tSNR_mean)
                print('first list is done.')            
                # the mean cv Rsq of the both hemis
                if len(verts_both) == 2:                                            
                    both_rsq   = np.concatenate((verts_both[0],verts_both[1]))
                    both_m     = np.mean(both_rsq)
                    df_both.extend([both_m, None])
                    
                    both_tsnr   = np.concatenate((tsnr_both[0],tsnr_both[1]))
                    both_tsnr_m = np.mean(both_tsnr) 
                    df_both_tmean.extend([both_tsnr_m, None])
                else:
                    df_both.extend([None, None])
                print('second list is done.')
            
            
            # create data frame
            data = zip(df_subs, df_hemis, df_rsq, df_both, df_max, df_max_mu, df_tsnr,df_mean_tsnr,df_both_tmean )
            cols = ['sub', 'hemi', 'sep_meanrsq', 'both_meanrsq','maxrsq','maxrsq_mu', 'mvert_tSNR', 'mean_tSNR', 'mean_both_tSNR']
            df    = pd.DataFrame(data, columns=cols)
        
        # save data frame to CSV/XLS
        if '00' in gr[0]:
            df.to_csv('/data/pt_02495/emprise7t_2_analysis/Figures/WP1_Tab_S1_sub-adults'+'_ses-visual.csv', index=False)
            print('Adult group table is saved.')
        else:
            df.to_csv('/data/pt_02495/emprise7t_2_analysis/Figures/WP1_Tab_S1_sub-childs'+'_ses-visual.csv', index=False)
            print('Children group table is saved.')

# function: Work Package 1, Figure 1
#-----------------------------------------------------------------------------#
def WP1_Fig1(Figure):
    
    # sub-function: plot tuning functions & time courses
    #-------------------------------------------------------------------------#
    def plot_tuning_function_time_course(sub, ses, model, hemi, space, cv=True, verts=[0,0], col='b'):
        
        # load session data
        mod            = EMPRISE.Model(sub, ses, model, space)
        Y, M           = mod.load_mask_data(hemi)
        labels         = EMPRISE.covs
        X_c            = mod.get_confounds(labels)
        X_c            = EMPRISE.standardize_confounds(X_c)
        ons, dur, stim = mod.get_onsets()
        ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
        ons0,dur0,stim0= ons[0], dur[0], stim[0]
        ons, dur, stim = EMPRISE.correct_onsets(ons0, dur0, stim0)
        
        # load model results
        res_file = mod.get_results_file(hemi)
        NpRF     = sp.io.loadmat(res_file)
        mu       = np.squeeze(NpRF['mu_est'])
        fwhm     = np.squeeze(NpRF['fwhm_est'])
        beta     = np.squeeze(NpRF['beta_est'])
        
        # determine tuning function
        try:
            ver = NpRF['version'][0]
        except KeyError:
            ver = 'V2'
        linear_model = 'lin' in ver
        
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
        
        # if CV, load cross-validated R^2
        if cv:
            Rsq_map = res_file[:res_file.find('numprf.mat')] + 'cvRsq.surf.gii'
            cvRsq   = nib.load(Rsq_map).darrays[0].data
            Rsq     = cvRsq[para_mask]
            del Rsq_map, para_mask, cvRsq
            
        # otherwise, calculate total R^2
        else:
            MLL1 = np.squeeze(NpRF['MLL_est'])
            MLL0 = np.squeeze(NpRF['MLL_const'])
            n    = np.prod(mod.calc_runs_scans())
            Rsq  = NumpRF.MLL2Rsq(MLL1, MLL0, n)
        
        # select vertices for plotting
        for k, vertex in enumerate(verts):
            if vertex == 0:
                if k == 0 and len(verts) == 1:
                    # select maximum R^2 in vertices with numerosity 1 < mu < 5
                    vertex = np.argmax(Rsq + np.logical_and(mu>1, mu<5)+ (beta>0))
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
                verts = vertex
        
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
            hdr  = 'sub-{}, ses-{}, hemi-{}'.format(sub, ses, hemi)
            txt1 = 'vertex {} \n(XYZ = [{:.0f}, {:.0f}, {:.0f}])'. \
                    format(vertex, XYZ[vertex,0], XYZ[vertex,1], XYZ[vertex,2])
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
            for j in range(y.shape[2]):
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
            txt = 'beta = {:.2f}\ncvR² = {:.2f}, {}\n '. \
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
    
    # define globals
    #sub_visual = '003'
    #sub_audio  = '009'
    subs_N    = EMPRISE.control
    subs_D    = EMPRISE.dyscal
    model      = model_def
    space      = 'fsnative'
    filepath   = '/data/pt_02495/emprise7t_2_analysis/Figures/Fig2/'
    
    # Figure 1
    if Figure == '1':
        
        # Figure 1, Part 1: visual child data
        #fig = plot_tuning_function_time_course('113', 'visual', model, 'L', space, verts=[0,0], col='b')
        #fig.savefig(filepath + 'WP1_Figure_1_ses-visual_children.png', dpi=150, transparent=True)
        
        # Figure 1, Part 2: visual data for the given subject and model (ex: log Gaussian, Linear )
        
        sub = input('Subject: ')
        model = input('Model: ')
        ver_ind= input('vertices: ')                  # [0,0] for rsq maximum for small numerosity and that of big numerosity, 
        verts = [int(i) for i in ver_ind.split('.') ] # [0] for maximum in full numerosity
        color = input('Color r or b: ')
        fig = plot_tuning_function_time_course(sub, 'visual', model, 'L', space, verts=verts, col = color)
        fig.savefig(filepath + 'WP1_Figure_1_ses-visual_'+ 'sub-'+ sub + '_model-'+ model +'.png', dpi=150, transparent=True)
        
        # Figure 1, Part 1: visual data (fsaverage)
        #fig = plot_tuning_function_time_course(sub_visual, 'visual', model, 'L', 'fsaverage', verts=[0,0], col='b')
        #fig.savefig('Figures_WP1/WP1_Figure_1_ses-visual_space-fsaverage.png', dpi=150, transparent=True)
        
        # Figure 1, Part 2: auditory data (fsaverage)
        #fig = plot_tuning_function_time_course(sub_audio, 'audio', model, 'R', 'fsaverage', verts=[0,0], col='r')
        #fig.savefig('Figures_WP1/WP1_Figure_1_ses-audio_space-fsaverage.png', dpi=150, transparent=True)
    
    # Figure S1
    if Figure == 'S1':
    
        # Figure S1: children subjects, visual
        mod = input('Type individual sub ID, otherwhise n: ')
        if mod=='n':
            for sub in subs_N:
                fig = plot_tuning_function_time_course(sub, 'visual', model, 'L', space, verts=[0], col='b')
                filename = filepath+'WP1_Figure_S1'+'_ses-'+'visual'+'_sub-'+sub+'.png'
                fig.savefig(filename, dpi=150, transparent=True)
        else:
            sub = mod
            fig = plot_tuning_function_time_course(sub, 'visual', model, 'L', space, verts=[0], col='b')
            filename = filepath+'WP1_Figure_S1'+'_ses-'+'visual'+'_sub-'+sub+'.png'
            fig.savefig(filename, dpi=150, transparent=True)
    
    # Figure S2
    if Figure == 'S2':
    
        # Figure S2: adult subjects, visual
        for sub in subs_D:
            fig = plot_tuning_function_time_course(sub, 'visual', model, 'L', space, verts=[0], col='r')
            filename = filepath+'WP1_Figure_S2'+'_ses-'+'visual'+'_sub-'+sub+'.png'
            fig.savefig(filename, dpi=150, transparent=True)
    
    # Figure S9
    if Figure == 'S9':
        
        # Figure S9: linear model adults and children visual session
        model = 'True_False_iid_1_V2-lin'
        fig   = plot_tuning_function_time_course('113', 'visual', model, 'L', space, verts=[0,0], col='b')
        fig.savefig(filepath+'WP1_Figure_S9_ses-visual'+'_sub-113'+'.png', dpi=150, transparent=True)
        
        fig   = plot_tuning_function_time_course('003', 'visual', model, 'L', space, verts=[0,0], col='r')
        fig.savefig(filepath+'WP1_Figure_S9_ses-visual'+'_sub-003'+'.png', dpi=150, transparent=True)
        
    # FIgures of the top 5 vertices
        # Figure S9
    if Figure == 'top5':
        
        # Figure S9: linear model adults and children visual session
        model = 'True_False_iid_1_V2'
        sub = input('subject: ')
        fig   = plot_tuning_function_time_course(sub, 'visual', model, 'L', space, verts=[1], col='b')
        fig.savefig(filepath+'WP1_Figure_S_top5_ses-visual'+'_sub-'+sub+'.png', dpi=150, transparent=True)
        


# function: Work Package 1, Figure 2
#-----------------------------------------------------------------------------#
def WP1_Fig2(Figure):
    
    # sub-function: plot surface parameter map
    #-------------------------------------------------------------------------#
    def plot_surface_para(sub, ses, model, space, para='mu', cv=False, Rsq_thr=None, alpha_thr=None, meth_str=''):
        
        # load analysis details
        mod = EMPRISE.Model(sub, ses, model, space)
        n   = np.prod(mod.calc_runs_scans())# effective number of observations in model
        p   = [4,2][int(cv)]                # number of explanatory variables used for R^2
                                            
        # specify thresholds
        if Rsq_thr is None:
            if alpha_thr is None:
                Rsq_thr = EMPRISE.Rsq_def
        # Explanation: If "Rsq_thr" and "alpha_thr" are not specified, then "Rsq_thr" is specified.
        elif Rsq_thr is not None:
            if alpha_thr is not None:
                Rsq_thr = None
        # Explanation: If "Rsq_thr" and "alpha_thr" are both specified, then "Rsq_thr" is disspecified.
        
        # configure thresholds              # for details, see
        mu_thr   = EMPRISE.mu_thr           # EMPRISE global variables
        fwhm_thr = EMPRISE.fwhm_thr
        beta_thr = EMPRISE.beta_thr
        if Rsq_thr is not None:
            crit = 'Rsqmb,'+ str(Rsq_thr)
        elif alpha_thr is not None:
            if alpha_thr < 0.001:
                crit = 'Rsqmb,p='+ '{:1.2e}'.format(alpha_thr) + meth_str
            else:
                crit = 'Rsqmb,p='+ str(alpha_thr) + meth_str
        
        # analyze hemispheres
        hemis = {'L': 'left', 'R': 'right'}
        maps  = {}
        for hemi in hemis.keys():
            
            # load numerosity map
            res_file = mod.get_results_file(hemi)
            filepath = res_file[:res_file.find('numprf.mat')]
            mu_map   = filepath + 'mu.surf.gii'
            if os.path.exists(res_file) and os.path.exists(mu_map):
                NpRF     = sp.io.loadmat(res_file)
                image    = nib.load(mu_map)
                mask     = image.darrays[0].data != 0
            else:
                print(f'file {filepath} does not exsist.')
                continue
            
            # load estimation results
            mu   = np.squeeze(NpRF['mu_est'])
            fwhm = np.squeeze(NpRF['fwhm_est'])
            beta = np.squeeze(NpRF['beta_est'])
            
            # if CV, load cross-validated R^2
            if cv:
                Rsq_map = res_file[:res_file.find('numprf.mat')] + 'cvRsq.surf.gii'
                if os.path.exists(Rsq_map):
                    
                    cvRsq   = nib.load(Rsq_map).darrays[0].data
                    Rsq     = cvRsq[mask]
                    
                else:
                    print(f"file {Rsq_map} doesn't exsit.")
                    continue
                
            # otherwise, calculate total R^2
            else:
                MLL1 = np.squeeze(NpRF['MLL_est'])
                MLL0 = np.squeeze(NpRF['MLL_const'])
                Rsq  = NumpRF.MLL2Rsq(MLL1, MLL0, n)
            
            # prepare quantities for thresholding
            ind_m = np.logical_or(mu<mu_thr[0], mu>mu_thr[1])
            ind_f = np.logical_or(fwhm<fwhm_thr[0], fwhm>fwhm_thr[1])
            ind_b = np.logical_or(beta<beta_thr[0], beta>beta_thr[1])
            
            # apply conditions for exclusion
            ind = mu > np.inf
            if 'Rsq' in crit:
                if not 'p=' in crit:
                    ind = np.logical_or(ind, Rsq < Rsq_thr)
                else:
                    ind = np.logical_or(ind, ~NumpRF.Rsqsig(Rsq, n, p, alpha_thr, meth_str))
            if 'm' in crit:
                ind = np.logical_or(ind, ind_m)
            if 'f' in crit:
                ind = np.logical_or(ind, ind_f)
            if 'b' in crit:
                ind = np.logical_or(ind, ind_b)
            
            # threshold parameter map
            para_est       = {'mu': mu, 'fwhm': fwhm, 'beta': beta, 'Rsq': Rsq}
            if para == 'cvRsq': para_est['cvRsq'] = para_est.pop('Rsq')
            para_map       = np.nan * np.ones(mask.size, dtype=np.float32)
            para_crit      = para_est[para]
            para_crit[ind] = np.nan
            para_map[mask] = para_crit
            filename       = filepath + para + '_thr-' + crit + '.surf.gii'
            para_img       = EMPRISE.save_surf(para_map, image, filename)
            maps[hemis[hemi]] = filename
            del para_crit, para_map, para_est, image, filename
        
        # specify threshold
        if Rsq_thr is None: Rsq_thr = 0.0
        # Explanation: If "alpha_thr" is used, set "Rsq_thr" to 0.0.
        
        # specify plotting
        if para == 'mu':
            caxis  = EMPRISE.mu_thr
            cmap   = 'gist_rainbow'
            clabel = 'preferred numerosity'
            cbar   ={'n_ticks': 5, 'decimals': 0}
        elif para == 'fwhm':
            caxis  = EMPRISE.fwhm_thr
            cmap   = 'rainbow'
            clabel = 'tuning width (FWHM)'
            cbar   = {'n_ticks': 7, 'decimals': 0}
        elif para == 'beta':
            caxis  = [0,5]
            cmap   = 'hot'
            clabel = 'scaling factor'
            cbar   = {'n_ticks': 6, 'decimals': 0}
        elif para == 'Rsq' or para == 'cvRsq':
            caxis  = [Rsq_thr,1]
            cmap   = 'hot'
            if para == 'Rsq':
                clabel = 'variance explained (R²)'
            elif para == 'cvRsq':
                clabel = 'variance explained (cvR²)'
            if Rsq_thr == 0.0:              # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
                cbar   = {'n_ticks': 6, 'decimals': 1}
            elif Rsq_thr == 0.1:            # 0.1, 0.4, 0.7, 1.0
                cbar   = {'n_ticks': 4, 'decimals': 1}
            elif Rsq_thr == 0.15:           # 0.15, 0.32, 0.49, 0.66, 0.83, 1.00
                cbar   = {'n_ticks': 6, 'decimals': 2}
            elif Rsq_thr == 0.2:            # 0.2, 0.4, 0.6, 0.8, 1.0
                cbar   = {'n_ticks': 5, 'decimals': 1}
            elif Rsq_thr == 0.25:           # 0.25, 0.50, 0.75, 1.00
                cbar   = {'n_ticks': 4, 'decimals': 2}
            elif Rsq_thr == 0.3:            # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
                cbar   = {'n_ticks': 8, 'decimals': 1}
            else:                           # otherwise, five ticks and two decimals
                cbar   = {'n_ticks': 5, 'decimals': 2}
        cbar['fontsize'] = 24
        
        # specify mesh files
        mesh_files = mod.get_mesh_files(space)
        sulc_files = mod.get_sulc_files(space)
        
        # load surface images
        surf_imgs = {}
        sulc_data = {}
        for hemi in maps.keys():
            surf_imgs[hemi] = surface.load_surf_data(maps[hemi])
            sulc_data[hemi] = surface.load_surf_data(sulc_files[hemi])
            sulc_data[hemi] = np.where(sulc_data[hemi]>0.0, 0.6, 0.2)
        
        # specify surface plot
        plot = Plot(mesh_files['left'], mesh_files['right'],
                    layout='row', views='lateral', size=(1600,600), zoom=1.5)
        plot.add_layer(sulc_data, color_range=(0,1),
                       cmap='Greys', cbar=False)
        plot.add_layer(surf_imgs, color_range=(caxis[0],caxis[1]),
                       cmap=cmap, cbar_label=clabel)
        
        # display surface plot
        fig = plot.build(colorbar=True, cbar_kws=cbar)
        fig.tight_layout()
        return fig
    
    # sub-function: plot participant count map
    #-------------------------------------------------------------------------#
    def plot_participant_count(subs, ses, model, cv=True, Rsq_thr=None, alpha_thr=None, meth_str=''):
        
        # specify thresholds
        if Rsq_thr is None:
            if alpha_thr is None:
                Rsq_thr = EMPRISE.Rsq_def
        # Explanation: If "Rsq_thr" and "alpha_thr" are not specified, then "Rsq_thr" is specified.
        elif Rsq_thr is not None:
            if alpha_thr is not None:
                Rsq_thr = None
        # Explanation: If "Rsq_thr" and "alpha_thr" are both specified, then "Rsq_thr" is disspecified.
        
        # configure thresholds              # for details, see
        mu_thr   = EMPRISE.mu_thr           # EMPRISE global variables
        fwhm_thr = EMPRISE.fwhm_thr
        beta_thr = EMPRISE.beta_thr
        if Rsq_thr is not None:
            crit = 'Rsqmb,'+ str(Rsq_thr)
        elif alpha_thr is not None:
            if alpha_thr < 0.001:
                crit = 'Rsqmb,p='+ '{:1.2e}'.format(alpha_thr) + meth_str
            else:
                crit = 'Rsqmb,p='+ str(alpha_thr) + meth_str
        
        # prepare loading
        N     = len(subs)
        maps  = [{} for i in range(N)]
        hemis = {'L': 'left', 'R': 'right'}
        space =  'fsaverage'
        
        # load all subjects
        for i, sub in enumerate(subs):
            
            # load analysis details
            mod = EMPRISE.Model(sub, ses, model, space)
            n   = np.prod(mod.calc_runs_scans())# effective number of observations in model
            p   = [4,2][int(cv)]                # number of explanatory variables used for R^2
            
            # analyze hemispheres
            for hemi in hemis.keys():
                
                # load numerosity map
                res_file = mod.get_results_file(hemi)
                filepath = res_file[:res_file.find('numprf.mat')]
                mu_map   = filepath + 'mu.surf.gii'
                NpRF     = sp.io.loadmat(res_file)
                image    = nib.load(mu_map)
                mask     = image.darrays[0].data != 0
                
                # load estimation results
                mu   = np.squeeze(NpRF['mu_est'])
                fwhm = np.squeeze(NpRF['fwhm_est'])
                beta = np.squeeze(NpRF['beta_est'])
                
                # if CV, load cross-validated R^2
                if cv:
                    Rsq_map = res_file[:res_file.find('numprf.mat')] + 'cvRsq.surf.gii'
                    cvRsq   = nib.load(Rsq_map).darrays[0].data
                    Rsq     = cvRsq[mask]
                    
                # otherwise, calculate total R^2
                else:
                    MLL1 = np.squeeze(NpRF['MLL_est'])
                    MLL0 = np.squeeze(NpRF['MLL_const'])
                    Rsq  = NumpRF.MLL2Rsq(MLL1, MLL0, n)
                
                # prepare quantities for thresholding
                ind_m = np.logical_or(mu<mu_thr[0], mu>mu_thr[1])
                ind_f = np.logical_or(fwhm<fwhm_thr[0], fwhm>fwhm_thr[1])
                ind_b = np.logical_or(beta<beta_thr[0], beta>beta_thr[1])
                
                # apply conditions for exclusion
                ind = mu > np.inf
                if 'Rsq' in crit:
                    if not 'p=' in crit:
                        ind = np.logical_or(ind, Rsq < Rsq_thr)
                    else:
                        ind = np.logical_or(ind, ~NumpRF.Rsqsig(Rsq, n, p, alpha_thr, meth_str))
                if 'm' in crit:
                    ind = np.logical_or(ind, ind_m)
                if 'f' in crit:
                    ind = np.logical_or(ind, ind_f)
                if 'b' in crit:
                    ind = np.logical_or(ind, ind_b)
                
                # threshold R-squared map
                para_map       = np.nan * np.ones(mask.size, dtype=np.float32)
                para_crit      = Rsq
                para_crit[ind] = np.nan
                para_map[mask] = para_crit
                maps[i][hemis[hemi]] = para_map
                del para_crit, para_map
        
        # specify target directory
        mod      = EMPRISE.Model('all', ses, model, space)
        targ_dir = mod.get_model_dir()
        if not os.path.isdir(targ_dir): os.makedirs(targ_dir)
        
        # calculate participant count maps
        cnt_maps = {}
        for hemi in hemis.keys():
            Y = np.array([y[hemis[hemi]] for y in maps])
            C = np.sum(~np.isnan(Y), axis=0).astype(np.int32)
            res_file = mod.get_results_file(hemi)
            filepath = res_file[:res_file.find('numprf.mat')]
            filename = filepath + 'cnt' + '_thr-' + crit + '.surf.gii'
            cnt_img  = EMPRISE.save_surf(C, image, filename)
            cnt_maps[hemis[hemi]] = filename
            del Y, C, res_file, filepath, filename
        
        # specify plotting
        caxis  = [1,N]
        cmap   = 'gist_rainbow'
        clabel = 'participant count'
        cbar   ={'n_ticks': N, 'decimals': 0, 'fontsize': 24}
        
        # specify mesh files
        mesh_files = mod.get_mesh_files(space)
        sulc_files = mod.get_sulc_files(space)
        
        # load surface images
        surf_imgs = {}
        sulc_data = {}
        for hemi in cnt_maps.keys():
            surf_imgs[hemi] = surface.load_surf_data(cnt_maps[hemi])
            sulc_data[hemi] = surface.load_surf_data(sulc_files[hemi])
            sulc_data[hemi] = np.where(sulc_data[hemi]>0.0, 0.6, 0.2)
        
        # specify surface plot
        plot = Plot(mesh_files['left'], mesh_files['right'],
                    layout='row', views='lateral', size=(1600,600), zoom=1.5)
        plot.add_layer(sulc_data, color_range=(0,1),
                       cmap='Greys', cbar=False)
        plot.add_layer(surf_imgs, color_range=(caxis[0],caxis[1]),
                       cmap=cmap, cbar_label=clabel)
        
        # display surface plot
        fig = plot.build(colorbar=True, cbar_kws=cbar)
        fig.tight_layout()
        return fig
    
 # sub function: plot multiple subjects within a table (2x2)   
 #------------------------------------------------------------------------------
    def plot_surface_para_multi(subjects, ses, model, space, para='mu', cv=True, Rsq_thr=None, alpha_thr=None, meth_str=''):
        """
        Create a table of surface parameter figures for multiple subjects.
        
        Parameters:
        - subjects: List of subject IDs
        - ses, model, space: Same as in original plot_surface_para function
        - para: Parameter to plot (default 'mu')
        - cv, Rsq_thr, alpha_thr, meth_str: Same as in original function
        
        Returns:
        - matplotlib figure with subplot grid of surface plots
        """        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 2, figsize=(40, 30))
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        axes = axes.flatten()
        
        # Store individual plot parameters for consistent colorbar
        plot_data = []
        
        # Generate surface plots for each subject
        for i, sub in enumerate(subjects):
            # Generate surface plot for this subject
            sub_fig = plot_surface_para(sub, ses, model, space, para, cv, Rsq_thr, alpha_thr, meth_str)
            
            # Extract the surface layer from the subplot
            for layer in sub_fig.axes:
                if hasattr(layer, 'images') and len(layer.images) > 0:
                    # Get the image data and colormap
                    img = layer.images[-1]
                    plot_data.append({
                        'data': img.get_array(),
                        'cmap': img.cmap,
                        'norm': img.norm
                    })
                    
                    # Copy the plot to the corresponding subplot
                    axes[i].imshow(img.get_array(), cmap=img.cmap, norm=img.norm, aspect='auto')
                    axes[i].axis('off')
            
            # Close the temporary figure to free up memory
            plt.close(sub_fig)
        
        # If we have consistent color mappings, create a single colorbar
        if plot_data:
            # Use the last plot's colormap and normalization
            sm = plt.cm.ScalarMappable(cmap=plot_data[-1]['cmap'], norm=plot_data[-1]['norm'])
            sm.set_array([])
            
            # Determine colorbar label based on parameter
            if para == 'mu':
                label = 'preferred numerosity'
            elif para == 'fwhm':
                label = 'tuning width (FWHM)'
            elif para == 'beta':
                label = 'scaling factor'
            elif para in ['Rsq', 'cvRsq']:
                label = 'variance explained (R²)' if para == 'Rsq' else 'variance explained (cvR²)'
            else:
                label = para
            
            # Add colorbar at the bottom of the figure
            #fig.subplots_adjust(bottom=0.15)
            #cbar = fig.colorbar(sm, ax=axes, shrink=0.8, aspect=30, pad=0.08)
            #cbar.set_label(label, fontsize=16)
            #cbar.ax.tick_params(labelsize=14)
        
        # Adjust layout and return figure
        plt.tight_layout()
        return fig
    
    # define globals
    sub_visual = 'N006'
    #sub_audio  = '009'
    subs_children = EMPRISE.childs
    subs_adults   = EMPRISE.adults
    subs_N        = EMPRISE.control
    subs_D        = EMPRISE.dyscal
    sess       =['audio1'] #, 'audio']
    model      = input('Model?: ')#model_def
    model_ab = model.split('_')[-1]

    space      = 'fsnative'
    Rsq        = 0.2            # R² > 0.2
    alpha      = 0.05           # p < 0.05
    meth       = 'B'            # Bonferroni
    filepath   = '/data/pt_02495/emprise7t_2_analysis/Figures/Fig3/'
    
    # Figure 2A
    if Figure == '2A':
        
        # Figure 2A, Part 1: visual data children example 
        fig = plot_surface_para('113', 'visual', model, space, para='cvRsq', alpha_thr=alpha, meth_str=meth)
        fig.savefig(filepath+'WP1_Figure_3a_sub-'+'113'+'_ses-visual_model-'+model_ab+'.png', dpi=300, transparent=True)
        
        # Figure 2A, Part 2: visual data adults example
        fig = plot_surface_para('003', 'visual', model, space, para='cvRsq', alpha_thr=alpha, meth_str=meth)
        fig.savefig(filepath+'WP1_Figure_3b_sub-'+'003'+'_ses-visual_model-'+model_ab+'.png', dpi=300, transparent=True)
    
    # Figure 2B
    if Figure == '2B':
        
        # Figure 2B, Part 1: visual data children
        #fig = plot_participant_count(subs_N, 'visual', model, alpha_thr=alpha, meth_str=meth)
        #fig.savefig(filepath+'WP1_Figure_3c_sub-control_ses-visual_model-'+model_ab+'.png', dpi=300, transparent=True)
        
        # Figure 2B, Part 2: audio data, N007 hsa no data so exclude it
        subs_N.pop(5)
        fig = plot_participant_count(subs_N, 'audio1', model, alpha_thr=alpha, meth_str=meth)
        fig.savefig(filepath+'WP1_Figure_3d_sub-control_ses-audio1_model-'+model_ab+'.png', dpi=300, transparent=True)
    
    # Figure 2B (different threshold)
    if Figure == '2Bp':
        
        # specify significance level
        alpha = 0.001
        
        # Figure 2B, Part 1: visual data
        fig = plot_participant_count(subs_children, 'visual', model, alpha_thr=alpha, meth_str='')
        fig.savefig(filepath+'WP1_Figure_3c_sub-children_ses-visual_p-'+str(alpha)+'.png', dpi=150, transparent=True)
        
        # Figure 2B, Part 2: audio data
        #fig = plot_participant_count(subs_all, 'audio', model, alpha_thr=alpha, meth_str='')
        #fig.savefig('Figures_WP1/WP1_Figure_2B_ses-audio_p-'+str(alpha)+'.png', dpi=150, transparent=True)
    
    # Figure 2C
    if Figure == '2C':
        
        # Figure 2C, Part 1: visual data children
        fig = plot_surface_para('113', 'visual', model, space, para='mu', alpha_thr=alpha, meth_str=meth)
        fig.savefig(filepath + 'WP1_Figure_3e_sub-'+'113'+'_ses-visual_model-'+model_ab+'.png', dpi=300, transparent=True)
        
        # Figure 2C, Part 2: visual data adults
        fig = plot_surface_para('003', 'visual', model, space, para='mu', alpha_thr=alpha, meth_str=meth)
        fig.savefig(filepath+'WP1_Figure_3f_sub-'+'003'+'_ses-visual_model-'+model_ab+'.png', dpi=300, transparent=True)
    
    # Figure S3
    if Figure == 'S3':
    
        # Figure S3: all subjects, cvRsq, significant test with alpha 0.05
        for sub in subs_N:
            for ses in sess:
                fig = plot_surface_para(sub, ses, model, space, para='Rsq', cv = False, alpha_thr=alpha, meth_str=meth)
                filename = filepath + 'WP1_Figure_S3'+'_ses-'+ses+'_sub-'+sub+ '_model-'+model_ab+'.png'
                fig.savefig(filename, dpi=300, transparent=True)
    
    # Figure S4
    if Figure == 'S4':
    
        # Figure S4: all subjects, mu after significant test
        for sub in subs_N:
            for ses in sess:
                fig = plot_surface_para(sub, ses, model, space, para='mu', cv = False, alpha_thr=alpha, meth_str=meth)
                filename = filepath + 'WP1_Figure_S4'+'_ses-'+ses+'_sub-'+sub+'_model-'+model_ab+'.png'
                fig.savefig(filename, dpi=300, transparent=True)

    if Figure == 'S3_Rsq':
    
        # Figure S3: all subjects, Rsq thresholded without significant test
        for sub in subs_N:
            for ses in sess:
                fig = plot_surface_para(sub, ses, model, space, para='Rsq',cv = False, Rsq_thr = 0.2, alpha_thr=None, meth_str='')
                filename = filepath + 'WP1_Figure_S3_Rsq'+'_ses-'+ses+'_sub-'+sub+ '_model-'+model_ab+'.png'
                fig.savefig(filename, dpi=300, transparent=True)

    # Multi subject figures
    if Figure == 'Multi-S3':
    
        for i in range (0,9,4):
            sub_list = [subs_children[i],subs_adults[i],subs_children[i+1],subs_adults[i+1],
                        subs_children[i+2],subs_adults[i+2],subs_children[i+3],subs_adults[i+3]]
            fig = plot_surface_para_multi(sub_list, 'visual', model, space, para='cvRsq', alpha_thr=alpha, meth_str=meth)
            filename = filepath + 'fig_S3_'+str(i+1) + '_model-'+ model_ab + '.png'
            fig.savefig(filename, dpi=300, transparent = True)

    if Figure == 'Multi-S4':
    
        for i in range (0,9,4):
            sub_list = [subs_children[i],subs_adults[i],subs_children[i+1],subs_adults[i+1],
                        subs_children[i+2],subs_adults[i+2],subs_children[i+3],subs_adults[i+3]]
            fig = plot_surface_para_multi(sub_list, 'visual', model, space, para='mu', alpha_thr=alpha, meth_str=meth)
            filename = filepath + 'fig_S4_'+str(i+1) + '_model-' + model_ab + '.png'
            fig.savefig(filename, dpi=300, transparent = True)
            
    if Figure == '2':
        
        # Specify the sub, ses, model, space, parameter
        
        sub   = input('Subject: ')
        ses   = input('Session: ')
        model = input('Model: ')
        space = input('Space: ')
        param  = input('Parameter - eg_Rsq,cvRsq, mu?: ')
        fig = plot_surface_para(sub, ses, model, space, para=param, alpha_thr=alpha, meth_str=meth)
        if param == 'cvRsq':
            fig.savefig(filepath+'WP1_Figure_S3_ses-'+ ses+ '_sub-' + sub + '_model-'+model.split('_')[-2] + '.png', dpi=300, transparent=True)
        elif param == 'mu':
            fig.savefig(filepath+'WP1_Figure_S4_ses-'+ ses+ '_sub-' + sub + '_model-'+model.split('_')[-2] +'.png', dpi=300, transparent=True)
        else:
            fig.savefig(filepath+'WP1_Figure_'+param + '_ses-'+ ses+ '_sub-' + sub + '_model-'+model.split('_')[-2] +'.png', dpi=300, transparent=True)
            
# function: Work Package 1, Figure 3
#-----------------------------------------------------------------------------#
def WP1_Fig3(Figure):
    
    # class: Figure 3 object
    #-------------------------------------------------------------------------#
    class Fig3_Obj():
        
        # function: initialize Figure 3 object
        #---------------------------------------------------------------------#
        def __init__(self, sub, res_dir, ses, model, space, mesh, AFNI=False):
            
            # store session, space, mesh
            self.sub     = sub
            self.res_dir = res_dir
            self.ses     = ses
            self.model     = model
            self.space   = space
            self.mesh    = mesh
            self.AFNI    = AFNI
            self.pref    = ['','AFNI_'][int(AFNI)]
        
        # sub-function: extract surface clusters
        #---------------------------------------------------------------------#
        def extract_surface_clusters(self, crit='Rsqmb', ctype='coords', d=3, k=100):
            
            # specify analysis
            if self.sub == 'adults':
                subs = EMPRISE.adults
            else:
                subs = EMPRISE.childs
            
            #model = model_def
            # if self.ses == 'visual': model = model_def
            # if self.ses == 'audio':  model = 'True_False_iid_1_V2-lin_new'
            
            # add R^2 threshold
            Rsq_thr = 0.2
            if ',' not in crit:
                crit = crit + ',' + str(Rsq_thr)
            
            # analyze subjects
            verts = {}
            trias = {}
            areas = {}
            for sub in subs:
                
                # check if results exist
                mod      = EMPRISE.Model(sub, self.ses, self.model, self.space)
                res_file = mod.get_results_file('L')
                if os.path.isfile(res_file):
                    
                    # if AFNI edge clustering
                    if self.AFNI:
                        # extract vertices and triangles
                        verts[sub], trias[sub] = \
                            mod.threshold_AFNI_cluster(crit, self.mesh, cv=True)
                    else:
                        verts[sub] = {}
                        trias[sub] = {}
                    
                    # analyze hemispheres
                    areas[sub] = {}
                    for hemi in hemis:
                        
                        # if XYZ distance clustering
                        if not self.AFNI:
                            # extract vertices and triangles
                            verts[sub][hemi], trias[sub][hemi] = \
                                mod.threshold_and_cluster(hemi, crit, self.mesh, ctype, d, k)
                        
                        # compute triangle surface areas
                        areas[sub][hemi] = \
                            calc_surface_area(verts[sub][hemi], trias[sub][hemi])
            
            # save results
            model_ab  = self.model.split('_')[-2]
            filepath1 = self.res_dir+'sub-'+ self.sub +'_space-'+self.space+ '_model-' + model_ab + '_'
            filepath2 = self.res_dir+'sub-'+ self.sub +'_ses-'+self.ses+'_model-'+ model_ab +'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
            sp.io.savemat(filepath1+'trias.mat', trias)
            sp.io.savemat(filepath2+'verts.mat', verts)
            sp.io.savemat(filepath2+'areas.mat', areas)
    
    # if input is '3', extract Figure data
    if Figure == '3':
    
        # specify extraction
        subjects = input('Subjects? childs or adults: ') #['adults','childs']
        subjects = subjects.split(',')
        sess   = ['visual']
        model  = input('Model?: ')
        spaces = input('Space? fsnative, fsaverage or both: ')
        spaces = spaces.split(',')#['fsaverage','fsnative'] 
        meshs  = ['pial'] #EMPRISE.meshs
        crit   = EMPRISE.crit_def           # "Rsqmb" = R-squared, mu and beta
        crit   = crit + ',p=0.05B'          # "p=0.05B" = p < 0.05, Bonferroni-corrected
        AFNI   = True           # edge clustering
        # ctype  = 'coords'     # distance clustering
        # d_mm   = 1.7          # maximum distance to cluster ~ voxel resolution
        # k_min  = 50           # minimum number of vertices in cluster = 50
        
        # extract clusters
        for sub in subjects:
            for ses in sess:
                for space in spaces:
                    for mesh in meshs:
                        f3 = Fig3_Obj(sub, res_dir, ses, model, space, mesh, AFNI)
                        if AFNI: f3.extract_surface_clusters(crit)
    
    # otherwise, create Figure panels
    else:
        
        # define globals
        sub_visual = '003'
        sub_audio  = '009'
        subs_all   = EMPRISE.adults
        sess       =['visual', 'audio']
        space      = 'fsnative'
        mesh       = 'pial'
        AFNI       = True
        
        # specify visualization
        d_mu  = 0.5
        cols  = {'visual': ['dodgerblue', 'darkblue'], \
                 'audio':  ['orangered',  'darkred']}
        f3    = {'visual': Fig3_Obj(res_dir, 'visual', space, mesh, AFNI), \
                 'audio':  Fig3_Obj(res_dir, 'audio',  space, mesh, AFNI)}

# function: Work Package 1, Figure 4
#-----------------------------------------------------------------------------#
def WP1_Fig4(Figure):
    
    # class: Figure 4 object
    #-------------------------------------------------------------------------#
    class Fig4_Obj():
        
        # function: initialize Figure 4 object
        #---------------------------------------------------------------------#
        def __init__(self, res_dir, ses, space, mesh, AFNI=True):
            
            # store session, space, mesh
            self.res_dir = res_dir
            self.ses     = ses
            self.space   = space
            self.mesh    = mesh
            self.AFNI    = AFNI
            self.pref    = ['','AFNI_'][int(AFNI)]
        
        # sub-function: plot area vs. map
        #---------------------------------------------------------------------#
        def plot_area_vs_map(self, subs, A_min, d_max, cols=['b','r']):
            
            # load results
            if subs == EMPRISE.adults:
                filepath = self.res_dir+'sub-adults'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
            elif subs == EMPRISE.childs:
                filepath = self.res_dir+'sub-childs'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
            else:
                print(f"Sub {subs} is not in the given subject groups")
                
            verts    = sp.io.loadmat(filepath+'verts.mat')
            areas    = sp.io.loadmat(filepath+'areas.mat')
            nmaps    = maps[self.ses]['labels']
            
            # calculate surface areas
            N = len(subs)
            A = np.zeros((N,len(nmaps),len(hemis)))
            for i, sub in enumerate(subs):
                for j, hemi in enumerate(hemis):
                    verts_sub = verts[sub][hemi][0,0]
                    areas_sub = areas[sub][hemi][0,0]
                    for k, nmap in enumerate(nmaps):
                        verts_map, areas_map = \
                            filter_clusters(verts_sub, areas_sub, A_min, d_max, nmap, hemi)
                        A[i,k,j] = np.sum(areas_map[:,6])
            
            # report existing maps
            print('\n-> ses-{}:'.format(self.ses))
            for j, hemi in enumerate(hemis):
                print('   - hemi-{}:'.format(hemi))
                for i, sub in enumerate(subs):
                    print('     - sub-{}: '.format(sub), end='')
                    for k, nmap in enumerate(nmaps):
                        if A[i,k,j] != 0:
                            print('{}, '.format(nmap), end='')
                    print()
                if hemi =='R': print()
            print()
            
            # average surface areas
            A_n    = np.zeros((len(hemis),len(nmaps)))
            A_mean = np.zeros((len(hemis),len(nmaps)))
            A_se   = np.zeros((len(hemis),len(nmaps)))
            for j, hemi in enumerate(hemis):
                for k, nmap in enumerate(nmaps):
                    A_jk        = A[A[:,k,j]!=0,k,j]
                    A_n[j,k]    = np.sum(A[:,k,j]!=0)
                    A_mean[j,k] = np.mean(A_jk)
                    A_se[j,k]   = np.std(A_jk)/math.sqrt(A_n[j,k])
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            ax  = fig.add_subplot(111)
            
            # plot area vs. map
            hdr = 'sub-all, ses-{}'.format(self.ses)
            for j, hemi in enumerate(hemis):
                lab_j = 'hemi-'+hemi
                x_j   = np.arange(len(nmaps))-0.1+j*0.2
                y_ls  = [1250, 1350]
                y_j   = y_ls[j] #((12-j)/10)*np.max(A_mean+A_se)
                ax.plot(x_j, A_mean[j,:], '-o', label=lab_j, \
                        color=cols[j], markerfacecolor=cols[j], markersize=10)
                ax.errorbar(x_j, A_mean[j,:], yerr=A_se[j,:], \
                            fmt='none', ecolor=cols[j], elinewidth=2)
                for k, nmap in enumerate(nmaps):
                    ax.text(k, y_j, 'n={}'.format(int(A_n[j,k])), color=cols[j], fontsize=20, \
                            horizontalalignment='center', verticalalignment='center')
            ax.axis([(0-1), len(nmaps), 0, 1400]) #(13/10)*np.max(A_mean+A_se)])
            ax.legend(loc='right', fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xticks(np.arange(len(nmaps)), labels=nmaps)
            ax.set_xlabel('numerotopic map', fontsize=28)
            ax.set_ylabel('cortical surface area [mm²]', fontsize=28)
            ax.set_title('', fontweight='bold', fontsize=20)
            
            # display figure
            fig.show()
            return fig
        
        # sub-function: plot area vs. mu
        #---------------------------------------------------------------------#
        def plot_area_vs_mu(self, sub, A_min=np.inf, d_mu=0.5, cols=['b','r']):
            
            # load results
            if sub in EMPRISE.adults:
                filepath = self.res_dir+'sub-adults'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
            elif sub in EMPRISE.childs:
                filepath = self.res_dir+'sub-childs'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
            else:
                print(f"Sub {sub} is not in the given groups.")
                
            verts    = sp.io.loadmat(filepath+'verts.mat')
            areas    = sp.io.loadmat(filepath+'areas.mat')
            
            # specify mu grid
            mu_min = EMPRISE.mu_thr[0]
            mu_max = EMPRISE.mu_thr[1]
            mu_b   = np.arange(mu_min, mu_max+d_mu, d_mu)
            mu_c   = np.arange(mu_min+d_mu/2, mu_max+d_mu/2, d_mu)
            mu_x   = np.arange(mu_min, mu_max+0.1, 0.1)
            
            # preallocate results
            pds    = [1,2]      # polynomial degrees
            pdstr  = ['linear','quadratic']
            area_s = np.zeros((len(hemis),mu_c.size))
            area_p = np.zeros((len(hemis),mu_x.size,len(pds)))
            R2_a   = np.zeros((len(hemis),len(pds)))
            R2p_a  = np.zeros((len(hemis),len(pds)))
            r_a    = np.zeros(len(hemis))
            p_a    = np.zeros(len(hemis))
            n_a    = np.zeros(len(hemis), dtype=np.int32)
            
            # for both hemispheres
            for j, hemi in enumerate(hemis):
                
                # get supra-threshold vertices/triangles
                verts_sub = verts[sub][hemi][0,0]
                areas_sub = areas[sub][hemi][0,0]
                mu_sub    = areas_sub[:,2]
                area_sub  = areas_sub[:,6]
                
                # if supra-threshold vertices exist
                if verts_sub.shape[0] > 0:
                    
                    # go through numerosity bins
                    for k in range(mu_c.size):
                        
                        # if this numerosity exists
                        ind_k = np.logical_and(mu_sub>mu_b[k],mu_sub<mu_b[k+1])
                        if np.sum(ind_k) > 0:
                            area_s[j,k] = np.sum(area_sub[ind_k])
                        else:
                            area_s[j,k] = 0 # np.nan

                    # extract x and y data
                    ind_j  = ~np.isnan(area_s[j,:])
                    n_a[j] = np.sum(ind_j)
                    y = area_s[j,ind_j]
                    x = mu_c[ind_j]
                    r_a[j], p_a[j], b0_a, b1_a = simplinreg(y, x)
                    y = np.array([y]).T
                    
                    # explore polynomial curves
                    for k, d in enumerate(pds):
                    
                        # fit polynomial curve
                        X = np.zeros((n_a[j],d+1))
                        for m in range(0,d+1):
                            X[:,m] = np.power(x, m)
                        bh = PySPM.GLM(y, X).OLS()
                        yp = X @ bh
                        R2_a[j,k]  = NumpRF.yp2Rsq(y, yp)
                        R2p_a[j,k] = NumpRF.Rsq2pval(R2_a[j,k], n_a[j], p=d+1)
                        
                        # predict polynomial curve
                        X = np.zeros((mu_x.size,d+1))
                        for m in range(0,d+1):
                            X[:,m] = np.power(mu_x, m)
                        area_p[j,:,k] = np.squeeze(X @ bh)
                        del X, bh, yp
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            ax  = fig.add_subplot(111)
    
            # plot area vs. mu
            hdr = 'sub-{}, ses-{}'.format(sub, self.ses)
            for j, hemi in enumerate(hemis):
                if np.any(area_s[j,:]):
                    ax.plot(mu_c, area_s[j,:], 'o', \
                            color=cols[j], markerfacecolor=cols[j], markersize=10)
                    for k, d in enumerate(pds):
                        if d == 1:
                            lab_j ='hemi-'+hemi+', fit-'+pdstr[k]+' (r = {:.2f}, {}, n = {})'. \
                                    format(r_a[j], pvalstr(p_a[j]), n_a[j])
                        else:
                            lab_j ='hemi-'+hemi+', fit-'+pdstr[k]+' (R² = {:.2f}, {}, n = {})'. \
                                    format(R2_a[j,k], pvalstr(R2p_a[j,k]), n_a[j])
                        ax.plot(mu_x, area_p[j,:,k], '-', \
                                color=cols[j], alpha=1/d, label=lab_j)
            ax.set_xlim(mu_min-d_mu, mu_max+d_mu)
            ax.set_ylim(-(1/10)*np.nanmax(area_s), +(11/10)*np.nanmax(area_s))
            ax.legend(loc='upper right', fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xlabel('preferred numerosity', fontsize=28)
            ax.set_ylabel('cortical surface area [mm²]', fontsize=28)
            ax.set_title('', fontweight='bold', fontsize=28)
            
            # display figure
            fig.show()
            return fig
        
        # sub-function: plot fwhm vs. mu
        #---------------------------------------------------------------------#
        def plot_fwhm_vs_mu(self, sub, A_min=np.inf, d_mu=0.5, cols=['b','r']):
            
            # load results
            if sub in EMPRISE.adults:
                filepath = self.res_dir+'sub-adults'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
            elif sub in EMPRISE.childs:
                filepath = self.res_dir+'sub-childs'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
                
            verts    = sp.io.loadmat(filepath+'verts.mat')
            areas    = sp.io.loadmat(filepath+'areas.mat')
            
            # specify mu grid
            mu_min = EMPRISE.mu_thr[0]
            mu_max = EMPRISE.mu_thr[1]
            mu_b   = np.arange(mu_min, mu_max+d_mu, d_mu)
            mu_c   = np.arange(mu_min+d_mu/2, mu_max+d_mu/2, d_mu)
            
            # preallocate results
            fwhm_m  = np.zeros((len(hemis),mu_c.size))
            fwhm_se = np.zeros((len(hemis),mu_c.size))
            r_f     = np.zeros(len(hemis))
            p_f     = np.zeros(len(hemis))
            b_f     = np.zeros((2,len(hemis)))
            n_f     = np.zeros(len(hemis), dtype=np.int32)
            
            # for both hemispheres
            for j, hemi in enumerate(hemis):
                
                # get supra-threshold vertices
                verts_sub = verts[sub][hemi][0,0]
                areas_sub = areas[sub][hemi][0,0]
                mu_sub    = verts_sub[:,2]
                fwhm_sub  = verts_sub[:,3]
                
                # if supra-threshold vertices exist
                if verts_sub.shape[0] > 0:
                
                    # go through numerosity bins
                    for k in range(mu_c.size):
                        
                        # if this numerosity exists
                        ind_k = np.logical_and(mu_sub>mu_b[k],mu_sub<mu_b[k+1])
                        if np.sum(ind_k) > 0:
                            fwhm_m[j,k]  = np.mean(fwhm_sub[ind_k])
                            fwhm_se[j,k] = np.std(fwhm_sub[ind_k])/math.sqrt(np.sum(ind_k))
                        else:
                            fwhm_m[j,k]  = np.nan
                            fwhm_se[j,k] = np.nan
                    
                    # calculate regression lines
                    ind_j  =~np.isnan(fwhm_m[j,:])
                    n_f[j] = np.sum(ind_j)
                    r_f[j], p_f[j], b_f[0,j], b_f[1,j] = \
                        simplinreg(fwhm_m[j,ind_j], mu_c[ind_j])
                    del ind_k, ind_j
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            ax  = fig.add_subplot(111)
    
            # plot fwhm vs. mu
            hdr = 'sub-{}, ses-{}'.format(sub, self.ses)
            for j, hemi in enumerate(hemis):
                if np.any(fwhm_m[j,:]):
                    lab_j = 'hemi-'+hemi+' (r = {:.2f}, {}, n = {})'. \
                             format(r_f[j], pvalstr(p_f[j]), n_f[j])
                    ax.plot(mu_c, fwhm_m[j,:], 'o', \
                            color=cols[j], markerfacecolor=cols[j], markersize=10)
                    ax.errorbar(mu_c, fwhm_m[j,:], yerr=fwhm_se[j,:], \
                                fmt='none', ecolor=cols[j], elinewidth=2)
                    ax.plot([mu_min,mu_max], np.array([mu_min,mu_max])*b_f[0,j]+b_f[1,j], '-', \
                            color=cols[j], label=lab_j)
            ax.set_xlim(mu_min-d_mu, mu_max+d_mu)
            ax.set_ylim(0, (11/10)*np.nanmax(fwhm_m+fwhm_se))
            ax.legend(loc='upper left', fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xlabel('preferred numerosity', fontsize=28)
            ax.set_ylabel('FWHM tuning width', fontsize=28)
            ax.set_title('', fontweight='bold', fontsize=28)
            
            # display figure
            fig.show()
            return fig
        
        # sub-function: plot topography
        #---------------------------------------------------------------------#
        def plot_topography(self, sub, hemi, A_min, d_max, cols='rgbcmy'):
            
            # load results
            if sub in EMPRISE.adults:
                filepath = self.res_dir+'sub-adults'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
            elif sub in EMPRISE.childs:
                filepath = self.res_dir+'sub-childs'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref

            verts    = sp.io.loadmat(filepath+'verts.mat')
            areas    = sp.io.loadmat(filepath+'areas.mat')
            nmaps    = maps[self.ses]['labels']
            clusts   = {}
            
            # get supra-threshold vertices/triangles
            verts_sub = verts[sub][hemi][0,0]
            areas_sub = areas[sub][hemi][0,0]
            
            # for all maps
            clusts = {}
            for k, nmap in enumerate(nmaps):
            
                # filter vertices
                verts_map, areas_map = \
                    filter_clusters(verts_sub, areas_sub, A_min, d_max, nmap, hemi)
                verts_cls = verts_map[:,1].astype(np.int32)
                areas_cls = areas_map[:,1].astype(np.int32)
                
                # calculate cluster areas
                areas_uni = np.unique(areas_cls)
                areas_all = [np.sum(areas_map[areas_cls==l,6]) for l in areas_uni]
                
                # for largest cluster
                for l in range(areas_uni.size):
                    if areas_all[l] == np.max(areas_all):
                        
                        # calculate cluster center and surface area
                        cls_i  = areas_uni[l]
                        XYZ_m  = np.mean(verts_map[verts_cls==cls_i,6:9], axis=0)
                        area_s = np.sum(areas_map[areas_cls==cls_i,6])
                        
                        # calculate distance to numerotopic map
                        if self.ses == 'visual':
                            c = maps['visual']['mean'][hemi][k,:]
                            e = XYZ_m - c
                            dist_m = np.sqrt(np.sum(np.power(e, 2)))
                        if self.ses == 'audio':
                            c = maps['audio']['mean'][hemi][k,:]
                            e = XYZ_m[1:3] - c[1:3]
                            e = np.concatenate((np.array([np.mean(e)]), e), axis=0)
                            dist_m = np.sqrt(np.sum(np.power(e, 2)))
                        
                        # retrieve numerosity and surface coordinates
                        y = verts_map[verts_cls==cls_i,2:3]
                        X = verts_map[verts_cls==cls_i,6:9]
                        X0= X - np.tile(XYZ_m, (X.shape[0],1))
                        X = np.zeros((X.shape[0],0))
                        d = 5 # degree of polynomial expansion
                        for m in range(1,d+1):
                            X = np.c_[X, np.power(X0,m)]
                        X = np.c_[X, np.ones((X.shape[0],1))]
                        
                        # prepare cross-validation matrix
                        n  = X.shape[0]
                        k  = 10 # number of cross-validation folds
                        CV = np.zeros((n,k))
                        nf = np.ceil(n/k)
                        ia = np.arange(0,n)
                        for g in range(k):
                            i2 = np.arange(g*nf, np.min([(g+1)*nf, n]), dtype=int)
                            i1 = [i for i in ia if i not in i2]
                            CV[i1,g] = 1
                            CV[i2,g] = 2
                        
                        # cross-validated linear regression
                        yp = np.zeros(y.shape)
                        for g in range(k):
                            i1 = np.nonzero(CV[:,g]==1)
                            i2 = np.nonzero(CV[:,g]==2)
                            i1 = np.array(i1[0], dtype=int)
                            i2 = np.array(i2[0], dtype=int)
                            b_hat    = PySPM.GLM(y[i1,:], X[i1,:]).OLS()
                            yp[i2,:] = X[i2,:] @ b_hat
                        Y = np.c_[y, yp]
                        del y, X, yp
                        
                        # store numerosity map
                        clusts[nmap] = {'cluster': cls_i, 'distance': dist_m, \
                                        'center': XYZ_m, 'area': area_s, \
                                        'Y': Y, 'removed': False}
                        del cls_i, dist_m, XYZ_m, area_s, Y
        
            # for all maps
            for nmap1 in nmaps:
                if nmap1 in clusts.keys():
                    cls1 = clusts[nmap1]
                    
                    # if other maps use the same cluster
                    if cls1['cluster'] in [clusts[nmap2]['cluster'] for nmap2 in clusts.keys() if nmap2 != nmap1]:
                        dists = [clusts[nmap2]['distance'] for nmap2 in clusts.keys()
                                  if clusts[nmap2]['cluster'] == cls1['cluster']]
                        
                        # if cluster does not have the minimum distance
                        if cls1['distance'] != np.min(dists):
                            clusts[nmap1]['removed'] = True
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            ax  = fig.add_subplot(111)
    
            # plot actual vs. predicted
            hdr    = 'sub-{}, ses-{}'.format(sub, self.ses)
            mu_thr = EMPRISE.mu_thr
            y_min  = mu_thr[0]
            y_max  = mu_thr[1]
            legend = False
            for k, nmap in enumerate(nmaps):
                if nmap in clusts.keys():
                    if not clusts[nmap]['removed']:
                        Y = clusts[nmap]['Y']
                        r, p, b1, b0 = simplinreg(Y[:,1], Y[:,0])
                        lab_k  = 'hemi-'+hemi+', map-'+nmap+' (cv-r = {:.2f}, {}, n = {})'.format(r, pvalstr(p), Y.shape[0])
                        legend = True
                        ax.plot(Y[:,0], Y[:,1], 'o', \
                                color=cols[k], markerfacecolor=cols[k], markersize=3)
                        ax.plot(mu_thr, np.array(mu_thr)*b1+b0, '-', \
                                color=cols[k], label=lab_k)
            y_rng = y_max - y_min
            ax.axis([y_min-(1/10)*y_rng, y_max+(1/10)*y_rng, \
                     y_min-(1/10)*y_rng, y_max+(1/10)*y_rng])
            if legend: ax.legend(loc='upper left', fontsize=20)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xlabel('actual preferred numerosity', fontsize=28)
            ax.set_ylabel('predicted preferred numerosity', fontsize=28)
            ax.set_title('', fontweight='bold', fontsize=28)
            
            # display figure
            fig.show()
            return fig
        
        # sub-function: plot range vs. map
        #---------------------------------------------------------------------#
        def plot_range_vs_map(self, sub, A_min, d_max, cols=['b','r']):
            
            # load results
            if sub in EMPRISE.adults:
                filepath = self.res_dir+'sub-adults'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
            elif sub in EMPRISE.childs:
                filepath = self.res_dir+'sub-childs'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
                
            verts    = sp.io.loadmat(filepath+'verts.mat')
            areas    = sp.io.loadmat(filepath+'areas.mat')
            nmaps    = maps[self.ses]['labels']
            clusts   = {}
            
            # for both hemispheres
            for j, hemi in enumerate(hemis):
                
                # get supra-threshold vertices/triangles
                verts_sub = verts[sub][hemi][0,0]
                areas_sub = areas[sub][hemi][0,0]
                
                # for all maps
                clusts[hemi] = {}
                for k, nmap in enumerate(nmaps):
                
                    # filter vertices
                    verts_map, areas_map = \
                        filter_clusters(verts_sub, areas_sub, A_min, d_max, nmap, hemi)
                    verts_cls = verts_map[:,1].astype(np.int32)
                    areas_cls = areas_map[:,1].astype(np.int32)
                    
                    # calculate cluster areas
                    areas_uni = np.unique(areas_cls)
                    areas_all = [np.sum(areas_map[areas_cls==l,6]) for l in areas_uni]
                    
                    # for largest cluster
                    for l in range(areas_uni.size):
                        if areas_all[l] == np.max(areas_all):
                            
                            # extract preferred numerosities
                            cls_i    = areas_uni[l]
                            XYZ_m    = np.mean(verts_map[verts_cls==cls_i,6:9], axis=0)
                            mu_clust = verts_map[verts_cls==cls_i,2]
                            
                            # calculate distance to numerotopic map
                            if self.ses == 'visual':
                                c = maps['visual']['mean'][hemi][k,:]
                                e = XYZ_m - c
                                dist_m = np.sqrt(np.sum(np.power(e, 2)))
                            if self.ses == 'audio':
                                c = maps['audio']['mean'][hemi][k,:]
                                e = XYZ_m[1:3] - c[1:3]
                                e = np.concatenate((np.array([np.mean(e)]), e), axis=0)
                                dist_m = np.sqrt(np.sum(np.power(e, 2)))
                            del c, e
                    
                            # store numerosity map
                            clusts[hemi][nmap] = \
                                {'cluster': cls_i, 'distance': dist_m, \
                                 'center': XYZ_m, 'mu': mu_clust, 'other': ''}
                            del cls_i, dist_m, XYZ_m, mu_clust
                
                # for all maps
                for nmap1 in nmaps:
                    if nmap1 in clusts[hemi].keys():
                        cls1 = clusts[hemi][nmap1]
                        
                        # if other maps use the same cluster
                        if cls1['cluster'] in [clusts[hemi][nmap2]['cluster'] for nmap2 in clusts[hemi].keys() if nmap2 != nmap1]:
                            dists = [clusts[hemi][nmap2]['distance'] for nmap2 in clusts[hemi].keys()
                                     if clusts[hemi][nmap2]['cluster'] == cls1['cluster']]
                            dmaps = [nmap2 for nmap2 in clusts[hemi].keys()
                                     if clusts[hemi][nmap2]['cluster'] == cls1['cluster']]
                            
                            # if cluster does not have the minimum distance
                            if cls1['distance'] != np.min(dists):
                                clusts[hemi][nmap1]['other'] = dmaps[np.argmin(dists)]
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            ax  = fig.add_subplot(111)
    
            # plot mu vs. cluster
            hdr = 'sub-{}, ses-{}'.format(sub, self.ses)
            lab = []; c = 0;
            for j, hemi in enumerate(hemis):
                for k, nmap in enumerate(nmaps):
                    c = c + 1
                    if nmap in clusts[hemi].keys():
                        if not clusts[hemi][nmap]['other']:
                            y  = clusts[hemi][nmap]['mu']
                            bp = ax.boxplot(y, positions=[c], widths=0.6, \
                                            sym='+k', notch=True, patch_artist=True)
                            bp['boxes'][0].set_facecolor(cols[j])
                            bp['medians'][0].set_color('k')
                        else:
                            ax.text(c, 1.5, 'same as '+hemi+'-'+clusts[hemi][nmap]['other'],
                                    fontsize=20, rotation=90, rotation_mode='default',
                                    horizontalalignment='center', verticalalignment='bottom')
                    lab.append(hemi+'-'+nmap)
            ax.axis([(1-0.5), (c+0.5), EMPRISE.mu_thr[0]-0.5, EMPRISE.mu_thr[1]+0.5])
            ax.set_xticks(np.arange(1,c+1), labels=lab)
            ax.tick_params(axis='both', labelsize=20)
            ax.set_xlabel('hemisphere and cluster', fontsize=28)
            ax.set_ylabel('preferred numerosities', fontsize=28)
            ax.set_title('', fontweight='bold', fontsize=28)
            
            # display figure
            fig.show()
            return fig
        
    # if input is '4', extract Figure data
    if Figure == '4':
    
        # perform extraction
        WP1_Fig3('3')
    
    # otherwise, create Figure panels
    else:
        
        # define globals
        #sub_visual = '003'
        #sub_audio  = '009'
        sub_adult  ='003'
        sub_child  ='113'
        subs_all   = EMPRISE.adults + EMPRISE.childs
        sess       =['visual'] #, 'audio']
        space      =['fsnative', 'fsaverage']
        mesh       = 'pial'
        AFNI       = True
        filepath   = '/data/pt_02495/emprise7t_2_analysis/Figures/'
        
        # specify visualization
        d_mu  = 0.5
        A_min = {'visual': 50, 'audio': 25}
        d_max = {'visual': 25, 'audio': 50}
        cols  = {'visual': ['dodgerblue', 'darkblue'], \
                 'audio':  ['orangered',  'darkred']}
        f4    = {'visual': {'fsnative':  Fig4_Obj(res_dir, 'visual', 'fsnative',  mesh, AFNI), \
                            'fsaverage': Fig4_Obj(res_dir, 'visual', 'fsaverage', mesh, AFNI)}, \
                 'audio':  {'fsnative':  Fig4_Obj(res_dir, 'audio',  'fsnative',  mesh, AFNI), \
                            'fsaverage': Fig4_Obj(res_dir, 'audio',  'fsaverage', mesh, AFNI)}}
        
        # Figure 4A
        if Figure == '4A':
        
            # Figure 4A, Part 1: children data
            fig = f4['visual']['fsaverage'].plot_area_vs_map(subs_all[12:], A_min['visual'], d_max['visual'], cols['visual'])
            fig.savefig(filepath + 'Fig4/WP1_Figure_4A_sub-children_ses-visual.png', dpi=150, transparent=True)
            
            # Figure 4A, Part 2: adults data
            fig = f4['visual']['fsaverage'].plot_area_vs_map(subs_all[:12], A_min['visual'], d_max['visual'], cols['audio'])
            fig.savefig(filepath + 'Fig4/WP1_Figure_4A_sub-adults_ses-visual.png', dpi=150, transparent=True)
        
        # Figure 4B
        if Figure == '4B':
        
            # Figure 4B, Part 1: children data
            fig = f4['visual']['fsnative'].plot_area_vs_mu(sub_child, A_min['visual'], d_mu, cols['visual'])
            fig.savefig(filepath + 'Fig5/WP1_Figure_4B_sub-child_ses-visual.png', dpi=150, transparent=True)
            
            # Figure 4B, Part 2: adults data
            fig = f4['visual']['fsnative'].plot_area_vs_mu(sub_adult, A_min['visual'], d_mu, cols['audio'])
            fig.savefig(filepath + 'Fig5/WP1_Figure_4B_sub-adult_ses-visual.png', dpi=150, transparent=True)
        
        # Figure 4C
        if Figure == '4C':
        
            # Figure 4C, Part 1: children data
            fig = f4['visual']['fsnative'].plot_fwhm_vs_mu(sub_child, A_min['visual'], d_mu, cols['visual'])
            fig.savefig(filepath + 'Fig5/WP1_Figure_4C_sub-child_ses-visual.png', dpi=150, transparent=True)
            
            # Figure 4C, Part 2: adults data
            fig = f4['visual']['fsnative'].plot_fwhm_vs_mu(sub_adult, A_min['visual'], d_mu, cols['audio'])
            fig.savefig(filepath + 'Fig5/WP1_Figure_4C_sub-adult_ses-visual.png', dpi=150, transparent=True)
        
        # Figure 4D
        if Figure == '4D':
        
            # Figure 4D, Part 1: children data
            fig = f4['visual']['fsaverage'].plot_topography(sub_child, 'L', A_min['visual'], d_max['visual'])
            fig.savefig(filepath + 'Fig6/WP1_Figure_4D_sub-child_ses-visual.png', dpi=150, transparent=True)
            
            # Figure 4D, Part 2: adults data
            fig = f4['visual']['fsaverage'].plot_topography(sub_adult, 'L', A_min['visual'], d_max['visual'])
            fig.savefig(filepath + 'Fig6/WP1_Figure_4D_sub-adult_ses-visual.png', dpi=150, transparent=True)
        
        # Figure 4E
        if Figure == '4E':
        
            # Figure 4E, Part 1: children data
            fig = f4['visual']['fsaverage'].plot_range_vs_map(sub_child, A_min['visual'], d_max['visual'], cols['visual'])
            fig.savefig(filepath + 'Fig6/WP1_Figure_4E_sub-child_ses-visual.png', dpi=150, transparent=True)
            
            # Figure 4E, Part 2: adults data
            fig = f4['visual']['fsaverage'].plot_range_vs_map(sub_adult, A_min['visual'], d_max['visual'], cols['audio'])
            fig.savefig(filepath + 'Fig6/WP1_Figure_4E_sub-adult_ses-visual.png', dpi=150, transparent=True)
            
        # Figure S5
        if Figure == 'S5':
        
            # Figure S5: all subjects, visual & audio
            for sub in subs_all[12:]:
                for ses in sess:
                    fig = f4[ses]['fsnative'].plot_area_vs_mu(sub, A_min[ses], d_mu, cols['visual'])
                    filename = filepath + 'Fig5/WP1_Figure_S5'+'_sub-'+sub+'_ses-'+ses+'.png'
                    fig.savefig(filename, dpi=150, transparent=True)
            
            for sub in subs_all[:12]:
                for ses in sess:
                    fig = f4[ses]['fsnative'].plot_area_vs_mu(sub, A_min[ses], d_mu, cols['audio'])
                    filename = filepath + 'Fig5/WP1_Figure_S5'+'_sub-'+sub+'_ses-'+ses+'.png'
                    fig.savefig(filename, dpi=150, transparent=True)
                
        
        # Figure S6
        if Figure == 'S6':
        
            # Figure S6: all subjects, visual & audio
            for sub in subs_all[12:]:
                for ses in sess:
                    fig = f4[ses]['fsnative'].plot_fwhm_vs_mu(sub, A_min[ses], d_mu, cols['visual'])
                    filename = filepath + 'Fig5/WP1_Figure_S6'+'_sub-'+sub+'_ses-'+ses+'.png'
                    fig.savefig(filename, dpi=150, transparent=True)
                    
            for sub in subs_all[:12]:
                for ses in sess:
                    fig = f4[ses]['fsnative'].plot_fwhm_vs_mu(sub, A_min[ses], d_mu, cols['audio'])
                    filename = filepath + 'Fig5/WP1_Figure_S6'+'_sub-'+sub+'_ses-'+ses+'.png'
                    fig.savefig(filename, dpi=150, transparent=True)
        
        # Figure S7
        if Figure == 'S7':
        
            # Figure S7: all subjects, visual & audio
            for sub in subs_all:
                for ses in sess:
                    if ses == 'visual':
                        fig = f4[ses]['fsaverage'].plot_topography(sub, 'L', A_min[ses], d_max[ses])
                    #elif ses == 'audio':
                    #    fig = f4[ses]['fsaverage'].plot_topography(sub, 'R', A_min[ses], d_max[ses])
                    filename = filepath + 'Fig6/WP1_Figure_S7'+'_sub-'+ sub +'_ses-'+ses+'.png'
                    fig.savefig(filename, dpi=150, transparent=True)
        
        # Figure S8
        if Figure == 'S8':
        
            # Figure S8: all subjects, visual & audio
            for sub in subs_all[12:]:
                for ses in sess:
                    fig = f4[ses]['fsaverage'].plot_range_vs_map(sub, A_min[ses], d_max[ses], cols['visual'])
                    filename = filepath + 'Fig6/WP1_Figure_S8'+'_sub-'+sub+'_ses-'+ses+'.png'
                    fig.savefig(filename, dpi=150, transparent=True)
                    
            for sub in subs_all[:12]:
                for ses in sess:
                    fig = f4[ses]['fsaverage'].plot_range_vs_map(sub, A_min[ses], d_max[ses], cols['audio'])
                    filename = filepath + 'Fig6/WP1_Figure_S8'+'_sub-'+sub+'_ses-'+ses+'.png'
                    fig.savefig(filename, dpi=150, transparent=True)

        # Figure of a specific subject
        if Figure == '5':
        
            # Figure S5
            sub = input('Subjects?: ')
            col = input('Colors?: visual sets blue, audio sets red ')
            for ses in sess:
                fig = f4[ses]['fsnative'].plot_area_vs_mu(sub, A_min[ses], d_mu, cols[col])
                filename = filepath + 'Fig5/WP1_Figure_S5'+'_sub-'+sub+'_ses-'+ses+'.png'
                fig.savefig(filename, dpi=150, transparent=True)

        
        if Figure == '6':
        
            # Figure S6
            sub = input('Subjects?: ')
            col = input('Colors?: visual sets blue, audio sets red ')
            for ses in sess:
                fig = f4[ses]['fsnative'].plot_fwhm_vs_mu(sub, A_min[ses], d_mu, cols[col])
                filename = filepath + 'Fig5/WP1_Figure_S6'+'_sub-'+sub+'_ses-'+ses+'.png'
                fig.savefig(filename, dpi=150, transparent=True)

        
        if Figure == '7':
        
            # Figure 7
            sub = input('Subjects?: ')
            for ses in sess:
                if ses == 'visual':
                    fig = f4[ses]['fsaverage'].plot_topography(sub, 'L', A_min[ses], d_max[ses])
                #elif ses == 'audio':
                #    fig = f4[ses]['fsaverage'].plot_topography(sub, 'R', A_min[ses], d_max[ses])
                filename = filepath + 'Fig6/WP1_Figure_S7'+'_sub-'+ sub +'_ses-'+ses+'.png'
                fig.savefig(filename, dpi=150, transparent=True)

        if Figure == '8':
        
            # Figure 8
            sub = input('Subjects?: ')
            col = input('Colors?: visual sets blue, audio sets red ')
            for ses in sess:
                fig = f4[ses]['fsaverage'].plot_range_vs_map(sub, A_min[ses], d_max[ses], cols[col])
                filename = filepath + 'Fig6/WP1_Figure_S8'+'_sub-'+sub+'_ses-'+ses+'.png'
                fig.savefig(filename, dpi=150, transparent=True)

# function: Work Package 1, Figure 5
#-----------------------------------------------------------------------------#
def WP1_Fig5(Step):
    
    # define analyses
    sub_id = input('Subject - adults or children or a list of subs? : ')
    if sub_id == 'adults':
        subs  = EMPRISE.adults
    elif sub_id == 'children':
        subs = EMPRISE.childs
    else:
        subs = sub_id.split(',')
        
    
    sess = input('Sessions: ')        
    sess  = sess.split(',')
    space = 'fsaverage'
    model = 'False_iid'
    
    # step 1: single-subject GLM analyses
    #-------------------------------------------------------------------------#
    if Step == 1:

        # define contrasts
        cons = [{'type': 'F', 'c': np.array([[1/5, 1/5, 1/5, 1/5, 1/5, -1]]).T},  \
                {'type': 't', 'c': np.array([+1/5, +1/5, +1/5, +1/5, +1/5, -1])}, \
                {'type': 't', 'c': np.array([-1/5, -1/5, -1/5, -1/5, -1/5, +1])}]
        
        # perform GLM analyses
        for sub in subs:
            for ses in sess:
                mod = EMPRISE.Model(sub, ses, model, space)
                mod.run_GLM_analysis(False, 'iid', cons)
    
    # step 2: group-level GLM analyses
    #-------------------------------------------------------------------------#
    if Step == 2:
    
        # specify model
        ana = 'False_iid_con_0002'
        y   = 'con_0002.surf.gii'
        X   = np.ones((len(subs),1))
        
        # define contrasts
        cons = [{'type': 'F', 'c': np.array([[1]])}, \
                {'type': 't', 'c': np.array([+1])},  \
                {'type': 't', 'c': np.array([-1])}]
    
        # perform GLM analyses
        for ses in sess:
            #grp = EMPRISE.Group('all', ses, model, ana, subs)
            grp = EMPRISE.Group(sub_id, ses, model, ana, subs)
            grp.run_GLM_analysis_group(y, X, cons)

    # step 3: threshold statistical maps
    #-------------------------------------------------------------------------#
    if Step == 3:
    
        # define inference
        ana   =   'False_iid_con_0002'
        cons  = [{'type': 'F', 'c': np.array([[1]])}, \
                 {'type': 't', 'c': np.array([+1])},  \
                 {'type': 't', 'c': np.array([-1])}]
        labs  = [ '1-5_neq_20', '1-5_gr_20', '20_gr_1-5' ]
        alpha = 0.001
        
        # threshold SPMs
        for ses in sess:
            grp = EMPRISE.Group(sub_id, ses, model, ana, subs)
            for k in range(1,len(cons)):
                fig = grp.threshold_SPM(k+1, alpha)
                filepath = '/data/pt_02495/emprise7t_2_analysis/Figures/FigS/'
                filename = filepath + 'WP1_Figure_S10'+ '_sub-'+ sub + '_ses-'+ses+'_con-'+labs[k]+'.png'
                fig.savefig(filename, dpi=150, transparent=True)

# function: Work Package 1, Figure 6
#-----------------------------------------------------------------------------#
def WP1_Fig6(Figure):
    def plot_simulation(mus = None, fwhms = None):
        # define numerosity signals
        if mus == None:
            mus   = [2, 4]
        elif mus == 'float':
            mus    = []
            mu_lst = input('preferred numerosity in float: ')
            mu_lst = mu_lst.split(',')
            for i in range(len(mu_lst)):
                a = float(mu_lst[i])
                mus.append(a)                
        elif mus == 'int':
            mus    = []
            mu_lst = input('preferred numerosity in float: ')
            mu_lst = mu_lst.split(',')
            for i in range(len(mu_lst)):
                a = int(mu_lst[i])
                mus.append(a)
            
        if fwhms == None:
            fwhms = [5, 10, 20, 40, 80]
        elif fwhms == 'float':
            fwhms    = []
            fwhm_lst = input('preferred numerosity in float: ')
            fwhm_lst = fwhm_lst.split(',')
            for i in range(len(fwhm_lst)):
                a = float(fwhm_lst[i])
                fwhms.append(a)                
        elif fwhms == 'int':
            fwhms    = []
            fwhm_lst = input('preferred numerosity in float: ')
            fwhm_lst = fwhm_lst.split(',')
            for i in range(len(fwhm_lst)):
                a = int(fwhm_lst[i])
                fwhms.append(a)
        
        # define numerosity experiment
        ons, dur, stim = EMPRISE.Session('108', 'visual').get_onsets()
        ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
        ons, dur, stim = EMPRISE.correct_onsets(ons[0], dur[0], stim[0])
        
        # open figure
        fig = plt.figure(figsize=(27,27))
        axs = fig.subplots(len(fwhms), len(mus))
        
        # simulate signals
        for k, mu in enumerate(mus):
            for j, fwhm in enumerate(fwhms):
                
                # create hemodynamic signals
                mu_log, sig_log = NumpRF.lin2log(mu, fwhm)
                # mu_lin, sig_lin = (mu, NumpRF.fwhm2sig(fwhm))
                Z, t = NumpRF.neuronal_signals(ons, dur, stim, EMPRISE.TR, EMPRISE.mtr, np.array([mu_log]), np.array([sig_log]), lin=False)
                # Z, t = NumpRF.neuronal_signals(ons, dur, stim, EMPRISE.TR, EMPRISE.mtr, np.array([mu_lin]), np.array([sig_lin]), lin=True)
                S, t = NumpRF.hemodynamic_signals(Z, t, EMPRISE.scans_per_epoch, EMPRISE.mtr, EMPRISE.mto, p=None, order=1)
                y    = S[:,:,0]
                
                # prepare axis limits
                y_min = np.min(y)
                y_max = np.max(y)
                y_rng = y_max-y_min
                t_max = np.max(t)
                
                # plot hemodynamic signals
                axs[j,k].plot(t, y, '-b', linewidth=2, label='expected hemodynamic signal')
                for i in range(len(ons)):
                    axs[j,k].plot(np.array([ons[i],ons[i]]), \
                                  np.array([y_max+(1/20)*y_rng, y_max+(3/20)*y_rng]), '-k')
                    axs[j,k].text(ons[i]+(1/2)*dur[i], y_max+(2/20)*y_rng, str(stim[i]), fontsize=18,
                                  horizontalalignment='center', verticalalignment='center')
                axs[j,k].plot(np.array([ons[-1]+dur[-1],ons[-1]+dur[-1]]), \
                              np.array([y_max+(1/20)*y_rng, y_max+(3/20)*y_rng]), '-k')
                axs[j,k].plot(np.array([ons[0],ons[-1]+dur[-1]]), \
                              np.array([y_max+(1/20)*y_rng, y_max+(1/20)*y_rng]), '-k')
                axs[j,k].axis([0, t_max, y_min-(1/20)*y_rng, y_max+(3/20)*y_rng])
                if j == 2 and k == 1:
                    axs[j,k].legend(loc='lower right', fontsize=27)
                if j == len(fwhms)-1:
                    axs[j,k].set_xlabel('within-cycle time [s]', fontsize=27)
                if k == 0:
                    axs[j,k].set_ylabel('fwhm = {:.0f}'.format(fwhm), fontsize=36)
                if j == 0:
                    axs[j,k].set_title('mu = {:.0f}'.format(mu), fontsize=36)
                axs[j,k].tick_params(axis='both', labelsize=18)
        
        fig.show()
                
        return fig

    # Figure S11
    if Figure == 'S11':
        
        # hemodynamic signals for different mu and fwhm
        filename = '/data/pt_02495/emprise7t_2_analysis/Figures/FigS/WP1_Figure_S11'+'_ses-'+'stimulation'+'.png'
        fig = plot_simulation(mus = 'float', fwhms = 'float')
        fig.savefig(filename, dpi=150, transparent=True)

# function: Work Package 1, Analysis 1
#-----------------------------------------------------------------------------#
def WP1_Ana1():
    
    # specify analyses
    subs  = EMPRISE.adults
    sess  =['visual', 'audio']
    spaces= EMPRISE.spaces
    mesh  = 'pial'
    cols  =['yellow', 'gold', 'orange', 'red', 'deeppink', 'purple', \
            'lime', 'darkgreen', 'cyan', 'blue', 'grey', 'sienna']
    A_min ={'visual': 50, 'audio': 25}
    d_max ={'visual': 51, 'audio': 51}
    
    # analyze sessions
    for ses in sess:
        
        # analyze spaces
        for space in spaces:
        
            # load results
            filepath = res_dir+'sub-adults'+'_ses-'+ses+'_space-'+space+'_mesh-'+mesh+'_'+'AFNI_'
            verts    = sp.io.loadmat(filepath+'verts.mat')
            areas    = sp.io.loadmat(filepath+'areas.mat')
            
            # open figure
            fig = plt.figure(figsize=(32,9))
            axs = fig.subplots(1,2)
            
            # analyze hemispheres
            for j, hemi in enumerate(hemis):
                
                # analyze clusters
                clusts = {}
                for sub in subs:
                    
                    # get clusters
                    verts_sub = verts[sub][hemi][0,0]
                    areas_sub = areas[sub][hemi][0,0]
                    
                    # filter clusters
                    if space == 'fsaverage':
                        verts_sub, areas_sub = filter_clusters(verts_sub, areas_sub, \
                                                               A_min[ses], d_max[ses], \
                                                               ses, hemi)
                    
                    # get cluster indices
                    verts_cls = verts_sub[:,1].astype(np.int32)
                    areas_cls = areas_sub[:,1].astype(np.int32)
                    
                    # if there are clusters
                    if verts_sub.shape[0] > 0:
                    
                        # get clusters
                        num_clust  = np.max(verts_cls)
                        clusts_sub = np.zeros((num_clust,5))
                        for k in range(num_clust):
                            clusts_sub[k,0]   = k+1
                            clusts_sub[k,1:4] = np.mean(verts_sub[verts_cls==k+1,6:9], axis=0)
                            clusts_sub[k,4]   = np.sum(areas_sub[areas_cls==k+1,6])
                        
                        # store clusters
                        clusts[sub] = clusts_sub
                        del clusts_sub
                        
                    # if there are no clusters
                    else:
                        
                        # store empty matrix
                        clusts[sub] = np.zeros((0,5))
                
                # visualize clusters
                hdr = 'ses-{}, space-{}, mesh-{}, hemi-{}'.format(ses, space, mesh, hemi)
                for i, sub in enumerate(subs):
                    axs[j].plot(-100, -100, 'o', linewidth=3, label='sub-'+sub, \
                                markeredgecolor=cols[i], markerfacecolor='none', markersize=15)
                    clusts_sub = clusts[sub]
                    num_clust  = clusts_sub.shape[0]
                    for k in range(num_clust):
                        x = clusts_sub[k,2]
                        y = clusts_sub[k,3]
                        A = clusts_sub[k,4]
                        if ses == 'visual': ms = round(A/10)
                        if ses == 'audio':  ms = round(A/2)
                        axs[j].plot(x, y, 'o', linewidth=3, \
                                    markeredgecolor=cols[i], markerfacecolor='none', markersize=ms)
                if space == 'fsaverage':
                    ses_maps = maps[ses]['labels']
                    ses_mean = maps[ses]['mean']
                    for k in range(len(ses_maps)):
                        x = ses_mean[hemi][k,1]
                        y = ses_mean[hemi][k,2]
                        axs[j].plot(x, y, '^sppph'[k], linewidth=3, label=ses_maps[k], \
                                    markeredgecolor='black', markerfacecolor='gray', markersize=15)
                axs[j].axis([-100, 30, -40, 80])
                if hemi == 'L':
                    axs[j].invert_xaxis()
                if ses == 'visual':
                    if hemi == 'L': axs[j].legend(loc='lower left',  fontsize=12)
                    if hemi == 'R': axs[j].legend(loc='lower right', fontsize=12)
                if ses == 'audio':
                    if hemi == 'L': axs[j].legend(loc='upper right', fontsize=12)
                    if hemi == 'R': axs[j].legend(loc='upper left',  fontsize=12)
                axs[j].tick_params(axis='both', labelsize=16)
                axs[j].set_xlabel('y-coordinate [mm]', fontsize=16)
                axs[j].set_ylabel('z-coordinate [mm]', fontsize=16)
                axs[j].set_title(hdr, fontweight='bold', fontsize=20)
            
            # save figure
            filename = 'Figures_WP1/WP1_Analysis_1'+'_ses-'+ses+'_space-'+space+'_mesh-'+mesh+'.png'
            fig.savefig(filename, dpi=150)

# function: Work Package 1, Analysis 2
#-----------------------------------------------------------------------------#
def WP1_Ana2():
    
    # Figures:
    # - Harvey et al. (2013), Fig. S5
    # - Harvey et al. (2015), Fig. 1E
    
    # specify analyses
    subs   = EMPRISE.adults
    sess   =['visual', 'audio']
    models ={'log': 'True_False_iid_1_V2_new', 
             'lin': 'True_False_iid_1_V2-lin_new'}
    space  = 'fsnative'
    cols   ={'visual': 'dodgerblue', 'audio': 'orangered'}
    select = 'loglin'
    cv     = True
    Rsq    = 0.2                # R² > 0.2
    alpha  = 0.001              # p < 0.001
    meth   = ''                 # uncorrected
    
    # analyze sessions
    for ses in sess:
        
        # analyze hemispheres
        for hemi in hemis:
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            axs = fig.subplots(3,4)
            
            # analyze subjects
            for i, sub in enumerate(subs):
                
                # analyze models
                Res = {}
                for fct in models.keys():
                    
                    # load model
                    mod = EMPRISE.Model(sub, ses, models[fct], space)
                    res_file = mod.get_results_file(hemi)
                    filepath = res_file[:res_file.find('numprf.mat')]
                    mu_map   = filepath + 'mu.surf.gii'
                    NpRF     = sp.io.loadmat(res_file)
                    mask     = nib.load(mu_map).darrays[0].data != 0
                    
                    # load parameter estimates
                    mu   = np.squeeze(NpRF['mu_est'])
                    fwhm = np.squeeze(NpRF['fwhm_est'])
                    beta = np.squeeze(NpRF['beta_est'])
                    n    = np.prod(mod.calc_runs_scans())# effective number of observations in model
                    p    = [4,2][int(cv)]                # number of explanatory variables used for R^2
                    
                    # load (cross-validated) R-squared
                    Res[fct] = {}
                    if cv:
                        Rsq_map = filepath + 'cvRsq.surf.gii'
                        cvRsq   = nib.load(Rsq_map).darrays[0].data
                        Res[fct]['Rsq'] = cvRsq[mask]
                    else:
                        MLL1 = np.squeeze(NpRF['MLL_est'])
                        MLL0 = np.squeeze(NpRF['MLL_const'])
                        Res[fct]['Rsq'] = NumpRF.MLL2Rsq(MLL1, MLL0, n)
                    
                    # select vertices
                    ind = NumpRF.Rsqsig(Res[fct]['Rsq'], n, p, alpha, meth)
                    # alternatively: ind = Res[fct]['Rsq'] > Rsq
                    ind = np.logical_and(ind, np.logical_and(mu   >= EMPRISE.mu_thr[0],
                                                             mu   <= EMPRISE.mu_thr[1]))
                    ind = np.logical_and(ind, np.logical_and(beta >= EMPRISE.beta_thr[0],
                                                             beta <= EMPRISE.beta_thr[1]))
                    Res[fct]['ind'] = ind
                    del mu, fwhm, beta, ind
                
                # compare R-squared
                if select == 'loglin':
                    ind_both  = np.logical_and(Res['log']['ind'], Res['lin']['ind'])
                elif select == 'log':
                    ind_both  = Res['log']['ind']
                elif select == 'lin':
                    ind_both  = Res['lin']['ind']
                print(np.sum(ind_both))
                delta_Rsq  = Res['log']['Rsq'] - Res['lin']['Rsq']
                delta_Rsq  = delta_Rsq[ind_both]
                log_better = np.sum(delta_Rsq>0)/delta_Rsq.size
                
                # prepare histogram
                d_Rsq        = 0.01
                x_Rsq        = np.arange(-1, +1+d_Rsq, d_Rsq)
                n_Rsq, x_Rsq = np.histogram(delta_Rsq, x_Rsq)
                x_Rsq        = x_Rsq[:-1] + d_Rsq/2
                
                # plot histogram
                j1    = i // 4
                j2    = i %  4
                hdr   = 'ses-{}, hemi-{}: sub-{}'.format(ses, hemi, sub)
                x_max = 0.1
                n_max = np.max(n_Rsq)
                axs[j1,j2].bar(x_Rsq, n_Rsq, width=(3/4)*d_Rsq, color=cols[ses])
                axs[j1,j2].plot([0,0], [0,(11/10)*n_max], '-k', linewidth=1)
                axs[j1,j2].axis([-x_max, +x_max, 0, (21/20)*n_max])
                axs[j1,j2].tick_params(axis='both', labelsize=10)
                if j1 == 2:
                    axs[j1,j2].set_xlabel('log cvR² minus lin cvR²', fontsize=12)
                if j2 == 0:
                    axs[j1,j2].set_ylabel('number of vertices', fontsize=12)
                if i == 0:
                    axs[j1,j2].set_title(hdr, fontweight='bold', fontsize=12)
                else:
                    axs[j1,j2].set_title('sub-{}'.format(sub), fontweight='bold', fontsize=12)
                axs[j1,j2].text((9/10)*x_max, (9/10)*n_max,
                                'log better:\n{:.2f}%'.format(log_better*100),
                                fontsize=10, horizontalalignment='right', verticalalignment='center')
                axs[j1,j2].text(-(9/10)*x_max, (9/10)*n_max,
                                'linear better:\n{:.2f}%'.format(100-log_better*100),
                                fontsize=10, horizontalalignment='left', verticalalignment='center')
            
            # save figure
            filename = 'Figures_WP1/WP1_Analysis_2'+'_ses-'+ses+'_hemi-'+hemi+'_select-'+select+'.png'
            fig.savefig(filename, dpi=150)

# test area / debugging section
#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    
    # select Figures
    Figures = ['S10']
    
    # available Figures
    # Figures = ['1', '2A', '2B', '2C', '2Bp', \
    #            '3', '4',  '4A', '4B', '4C', '4D', '4E', '5', \
    #            'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', \
    #            'T1', 'T2', 'A1', 'A2']
    
    # create Figures
    for Figure in Figures:
    
        # create Table 2
        if Figure == 'T1':
            WP1_Tab1()
        
        # create Table 2
        if Figure == 'T2':
            WP1_Tab2()
        
        # create Figure 1
        if Figure == '1' or Figure in ['S1','S2','S9']:
            WP1_Fig1(Figure)
        
        # create Figure 2
        if Figure.startswith('2') or Figure in ['S3','S4']:
            WP1_Fig2(Figure)
            
        # create Figure 3
        if Figure.startswith('3'):
            WP1_Fig3(Figure)
        
        # create Figure 4
        if Figure.startswith('4') or Figure in ['S5','S6','S7','S8']:
            WP1_Fig4(Figure)
        
        # create Figure 5
        if Figure == '5':
            WP1_Fig5(1)
            WP1_Fig5(2)
        if Figure == 'S10':
            WP1_Fig5(3)
        
        # create Figure 5
        if Figure == '6' or Figure == 'S11':
            WP1_Fig6(Figure)
        
        # run Analysis 1
        if Figure == 'A1':
            WP1_Ana1()
        
        # run Analysis 2
        if Figure == 'A2':
            WP1_Ana2()