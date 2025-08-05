"""
Figures - Figures within the EMPRISE project

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-10-23, 16:15: WP 1, Figure 1
2023-10-24, 18:26: WP 1, Figure 1
2023-10-26, 18:05: WP1_Fig1
2023-10-26, 21:47: WP 1, Figure 2
2023-10-30, 15:40: WP1_Fig2
2023-10-30, 16:22: WP1_Fig2
2023-10-31, 17:19: WP1_Fig2
2023-11-01, 17:55: WP 1, Figure 3
2023-11-02, 10:47: WP1_Fig1
2023-11-02, 12:55: WP1_Fig2
2023-11-02, 14:43: WP 1, Figure 3
2023-11-02, 18:07: WP 1, Figure 3
2023-11-06, 15:30: WP1_Fig3
2023-11-06, 20:34: WP1_Fig3
2023-11-07, 16:58: Talk, Figures
2023-11-08, 16:01: Talk, Figures
2023-11-08, 16:42: WP1_Fig3
2023-11-09, 11:30: refactoring
2023-11-09, 15:55: simplinreg
2023-11-09, 18:32: WP1_Fig3
2023-11-09, 19:26: WP1_Fig3
2023-11-09, 19:59: WP1_Fig1
2023-11-09, 20:05: WP1_Fig2
2023-11-09, 20:27: WP1_Fig3
2023-11-09, 21:50: Talk, Figures
2023-11-10, 11:09: Talk, Figures 1-4
2023-11-10, 11:25: Talk, Figures 5-6
2023-11-10, 16:05: Talk, Figure A
2023-11-10, 17:25: Talk, Figure B
2023-11-13, 13:16: Talk, Figures
2023-11-13, 13:25: WP1_Fig3
2023-11-16, 13:00: refactoring
2023-11-16, 15:00: refactoring
2023-11-16, 15:15: WP1_Ana1
2023-11-16, 15:39: WP1_Fig3
2023-11-16, 19:31: Talk_Figs
2023-11-16, 19:57: WP1_Fig3
2023-11-20, 14:29: WP1_Tab0
2023-11-20, 14:50: WP1_Fig2
2023-11-20, 15:16: WP1_Fig3
2023-11-22, 11:34: WP1_Ana1
2023-11-22, 12:41: WP1_Fig3
2023-11-23, 16:24: WP1_Ana2
2023-11-24, 09:48: WP1_Ana2
2023-11-27, 14:58: WP1_Fig1
2023-11-27, 15:55: WP1_Fig2/Fig3/Ana2
2023-11-30, 21:58: WP1_Fig3
2023-12-04, 14:57: filter_clusters
2023-12-07, 17:22: filter_clusters
2023-12-07, 18:37: WP1_Fig4
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
import imageio
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import PySPMs as PySPM
import NumpRF
import EMPRISE

# specify results directory
#-----------------------------------------------------------------------------#
if EMPRISE.at_MPI:
    res_dir = '/data/u_jeong_software/Figures_WP3/'
#else:
    #res_dir = r'C:\Users\sochj\ownCloud_MPI\MPI\EMPRISE\tools\Python\NumpRF_results/'

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
    observations y on predictors x and 
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
                      # E = np.mean(verts[verts_cls==i,6:9], axis=0) - c
                      # d = np.sqrt(np.sum(np.power(E, 2)))
                        E = verts[verts_cls==i,6:9] - np.tile(c, (np.sum(verts_cls==i),1))
                        d = np.sqrt(np.sum(np.power(E, 2), axis=1))
                        d = np.min(d)
                    if ses == 'audio':
                        c = maps['audio']['mean'][hemi][ind,:]
                      # E = np.mean(verts[verts_cls==i,7:9]) - c[1:3]
                      # E = np.concatenate((np.array([np.mean(E)]), E), axis=0)
                      # d = np.sqrt(np.sum(np.power(E, 2)))
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
    
# function: Institute Colloquium Talk
#-----------------------------------------------------------------------------#
def Talk_Figs():
    
    # Figure 1: distinct pattern
    L   = 10
    dy  = 0.05
    xd  = np.arange(1,L+1)
    y   = np.array([5, 3, 8, 10, 4, 9, 7, 2, 4, 6])/10
    fig = plt.figure(figsize=(16,9))
    ax  = fig.add_subplot(111)
    ax.bar(xd, y, color='b', edgecolor='k')
    ax.axis([0, L+1, 0-dy, 1+dy])
    ax.set_xticks(xd)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('numerosity', fontsize=24)
    ax.set_ylabel('activity', fontsize=24)
    ax.set_title('Distinct Numerosity Pattern', fontsize=24)
    # fig.savefig('Figures_Talk/Talk_Figure_1.png', dpi=150)
    
    # Figure 2: smooth pattern
    mu  = 4
    fwhm= 8
    sig = NumpRF.fwhm2sig(fwhm)
    y   = NumpRF.f_lin(xd, mu, sig)
    fig = plt.figure(figsize=(16,9))
    ax  = fig.add_subplot(111)
    ax.bar(xd, y, color='b', edgecolor='k')
    ax.axis([0, L+1, 0-dy, 1+dy])
    ax.set_xticks(xd)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('numerosity', fontsize=24)
    ax.set_ylabel('activity', fontsize=24)
    ax.set_title('Smooth Numerosity Pattern', fontsize=24)
    # fig.savefig('Figures_Talk/Talk_Figure_2.png', dpi=150)
    
    # Figure 3: linear Gaussian tuning function
    dx  = 0.1
    xc  = np.arange(dx, L+1, dx)
    y   = NumpRF.f_lin(xc, mu, sig)
    fig = plt.figure(figsize=(16,9))
    ax  = fig.add_subplot(111)
    ax.plot(xc, y, '-b', linewidth=3)
    ax.axis([0, L+1, 0-dy, 1+dy])
    ax.set_xticks(xd)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('numerosity', fontsize=24)
    ax.set_ylabel('activity', fontsize=24)
    ax.set_title('Linear Gaussian Tuning', fontsize=24)
    # fig.savefig('Figures_Talk/Talk_Figure_3.png', dpi=150)
    
    # Figure 4: logarithmic Gaussian tuning function
    y   = NumpRF.f_log(xc, np.log(mu), NumpRF.lin2log(mu,fwhm)[1])
    fig = plt.figure(figsize=(16,9))
    ax  = fig.add_subplot(111)
    ax.plot(xc, y, '-b', linewidth=2)
    ax.axis([0, L+1, 0-dy, 1+dy])
    ax.set_xticks(xd)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('numerosity', fontsize=24)
    ax.set_ylabel('activity', fontsize=24)
    ax.set_title('Logarithmic Gaussian Tuning', fontsize=24)
    # fig.savefig('Figures_Talk/Talk_Figure_4.png', dpi=150)
    
    # Figure 4: logarithmic Gaussian tuning function (with mu and fwhm)
    (mu_log, sig_log) = NumpRF.lin2log(mu, fwhm)
    y   = NumpRF.f_log(xc, mu_log, sig_log)
    fig = plt.figure(figsize=(16,9))
    ax  = fig.add_subplot(111)
    x1  = np.exp(mu_log - math.sqrt(2*math.log(2))*sig_log)
    x2  = np.exp(mu_log + math.sqrt(2*math.log(2))*sig_log)
    ax.plot(xc, y, '-b', linewidth=2)
    ax.plot([mu, mu], [0,1], '-', color='gray', linewidth=2)
    ax.plot([x1, x2], [0.5,0.5], '-', color='gray', linewidth=2)
    ax.text(mu, 0+0.01, ' mu = preferred numerosity', fontsize=16, \
            horizontalalignment='left', verticalalignment='bottom')
    ax.text((x1+x2)/2, 0.5-0.01, 'fwhm = tuning width', fontsize=16, \
            horizontalalignment='center', verticalalignment='top')
    ax.axis([0, L+1, 0-dy, 1+dy])
    ax.set_xticks(xd)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('numerosity', fontsize=24)
    ax.set_ylabel('activity', fontsize=24)
    ax.set_title('Logarithmic Gaussian Tuning', fontsize=24)
    # fig.savefig('Figures_Talk/Talk_Figure_4a.png', dpi=150)
    
    # Figure A: linear to logarithmic tuning
    xcl = np.arange(-5, 10, dx/2)
    ycl = NumpRF.f_lin(xcl, mu_log, sig_log)
    
    # Figure A: function
    def Figure_A(exp):
        if exp == 2.7: exp = np.exp(1)
        fig = plt.figure(figsize=(16,9))
        ax  = fig.add_subplot(111)
        ax.plot(np.power(exp, xcl), ycl, '-b', linewidth=2)
        ax.axis([0, L+1, 0-dy, 1+dy])
        ax.set_xticks(xd)
        ax.tick_params(axis='both', labelsize=20)
        if exp > 2:
            ax.set_xlabel('numerosity', fontsize=24)
        else:
            ax.set_xlabel('logarithmized numerosity', fontsize=24)
        ax.set_ylabel('activity', fontsize=24)
        ax.set_title('Logarithmic Gaussian Tuning', fontsize=24)
        return fig
    
    # Figure A: create GIF
    images = []
    exps   = np.concatenate((np.arange(2.7, 1.2, -0.1), np.arange(1.3, 2.8, 0.1)))
    for exp in exps:
        fig = Figure_A(exp)
        filename = 'Figures_Talk/Talk_Figure_A_temp.png'
        fig.savefig(filename, dpi=150)
        images.append(imageio.imread(filename))
    os.remove(filename)
    imageio.mimsave('Figures_Talk/Talk_Figure_A.gif', images, fps=10, loop=0)
    del fig, images, exps
    
    # Figure 5: different tuning functions
    labels = ['low-numerosity neuron', 'high-numerosity neuron', 'no numerosity response']
    colors = ['g','b','r']
    (mu1, fwhm1)        = (1.5, 3)
    (mu2, fwhm2)        = (mu, fwhm)
    (mu_log1, sig_log1) = NumpRF.lin2log(mu1, fwhm1)
    (mu_log2, sig_log2) = NumpRF.lin2log(mu2, fwhm2)
    ys  =[NumpRF.f_log(xc, mu_log1, sig_log1), \
          NumpRF.f_log(xc, mu_log2, sig_log2), \
          np.zeros(xc.shape)]
    fig = plt.figure(figsize=(16,9))
    ax  = fig.add_subplot(111)
    for k in range(3):
        ax.plot(xc, ys[k], '-'+colors[k], linewidth=2, label=labels[k])
    ax.axis([0, L+1, 0-dy, 1+dy])
    ax.set_xticks(xd)
    ax.legend(loc='upper right', fontsize=16)
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('numerosity', fontsize=24)
    ax.set_ylabel('activity', fontsize=24)
    ax.set_title('Differently Tuned Neurons', fontsize=24)
    # fig.savefig('Figures_Talk/Talk_Figure_5.png', dpi=150)
    
    # Figure 6: hemodynamic response function
    dy  = 0.1
    dt  = 0.1
    tH  = np.arange(0, 32, dt)
    y   = PySPM.spm_hrf(dt)
    fig = plt.figure(figsize=(16,9))
    ax  = fig.add_subplot(111)
    ax.plot(tH, y, '-m', linewidth=2)
    ax.axis([0, np.max(tH)+dt, 0-dy, 1+dy])
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('post-stimulus time [s]', fontsize=24)
    ax.set_ylabel('hemodynamic response', fontsize=24)
    ax.set_title('Hemodynamic Response Function', fontsize=24)
    # fig.savefig('Figures_Talk/Talk_Figure_6.png', dpi=150)
    
    # Figure B: stimuli, neuronal and hemodynamic signals
    np.random.seed(1)
    ons, dur, stim = EMPRISE.Session('001', 'visual').get_onsets()
    ons, dur, stim = EMPRISE.onsets_trials2blocks(ons, dur, stim, 'closed')
    ons, dur, stim = EMPRISE.correct_onsets(ons[0], dur[0], stim[0])
    dur[-1] = 15.6
    # (ons,dur,stim) = (ons[0], dur[0], stim[0])
    T   = math.ceil(np.max(ons+dur))
    dt  = EMPRISE.TR/EMPRISE.mtr
    ons0= np.round(np.array(ons)/dt)
    dur0= np.round(np.array(dur)/dt)
    X   = np.zeros(math.ceil(T/dt))
    tX  = np.arange(0,T,dt)
    for o,d,s in zip(ons0,dur0,stim):
        X[int(o):(int(o)+int(d))] = s
    for i in range(1,X.size-1):
        if X[i] == 0 and X[i-1] != 0 and X[i+1] != 0: X[i] = X[i-1]
    Z1, tZ = NumpRF.neuronal_signals(ons, dur, stim, EMPRISE.TR, EMPRISE.mtr, np.array([mu_log1]), np.array([sig_log1]))
    Z2, tZ = NumpRF.neuronal_signals(ons, dur, stim, EMPRISE.TR, EMPRISE.mtr, np.array([mu_log2]), np.array([sig_log2]))
    Z3     = np.zeros(Z1.shape)
    nZ     = EMPRISE.scans_per_epoch # EMPRISE.n
    Y1, tY = NumpRF.hemodynamic_signals(Z1, tZ, nZ, EMPRISE.mtr, EMPRISE.mto)
    Y2, tY = NumpRF.hemodynamic_signals(Z2, tZ, nZ, EMPRISE.mtr, EMPRISE.mto)
    Y3     = np.zeros(Y1.shape)
    Z      = np.c_[Z1, Z2, Z3]
    Y      = np.c_[Y1[:,:,0], Y2[:,:,0], Y3[:,:,0]]
    Yn     = Y + np.random.normal(0, 0.2, Y.shape)
    del T, ons0, dur0, o, d, s, Z1, Z2, Z3, Y1, Y2, Y3
    
    # Figure B: function
    def Figure_B(t):
        
        # select current stimulus
        x = X[np.argmin(abs(tX-t))]
        
        # Panel A: stimuli
        fig = plt.figure(figsize=(16,9))
        axs = fig.subplots(3, 2, width_ratios=[1,2])
        axs[0,0].remove()
        axs[0,1].plot(tX[tX<=t], X[tX<=t], '-k', linewidth=2)
        axs[0,1].axis([0-10*dt, np.max(tY)+10*dt, 0-5*dy, 7+5*dy])
        axs[0,1].set_xticks([])
        axs[0,1].set_yticks(list(range(1,6)))
        axs[0,1].tick_params(axis='both', labelsize=16)
        axs[0,1].set_ylabel('numerosity', fontsize=24)
        axs[0,1].set_title('Presented Stimuli', fontsize=28)
        for i in range(len(ons)):
            axs[0,1].plot(np.array([ons[i],ons[i]]), np.array([6+5*dy, 7+5*dy]), '-k')
            axs[0,1].text(ons[i]+(1/2)*dur[i], np.mean([6+5*dy, 7+5*dy]), str(stim[i]), fontsize=16,
                          horizontalalignment='center', verticalalignment='center')
        axs[0,1].plot(np.array([ons[-1]+dur[-1],ons[-1]+dur[-1]]), np.array([6+5*dy, 7+5*dy]), '-k')
        axs[0,1].plot(np.array([ons[0],ons[-1]+dur[-1]]), np.array([6+5*dy, 6+5*dy]), '-k')
        
        # Panel B: neuronal signals
        cols_alt = ['lightgreen', 'lightskyblue']
        for k in range(3):
            axs[1,1].plot(tZ[tZ<=t], Z[tZ<=t,k], '-'+colors[k], linewidth=2)
            if k < 2:
                axs[1,1].plot([0-10*dt,t], [Z[np.argmin(abs(tZ-t)),k], Z[np.argmin(abs(tZ-t)),k]], '-', color=cols_alt[k], linewidth=2)
                axs[1,1].plot(t, Z[np.argmin(abs(tZ-t)),k], 'o'+colors[k], markerfacecolor=colors[k], markersize=10, linewidth=2)
        axs[1,1].axis([0-10*dt, np.max(tY)+10*dt, 0-dy, 1+dy])
        axs[1,1].set_xticks([])
        axs[1,1].set_yticks([0,1])
        axs[1,1].tick_params(axis='both', labelsize=16)
        axs[1,1].set_ylabel('activity', fontsize=24)
        axs[1,1].set_title('Neuronal Signals', fontsize=28)
        
        # Panel C: hemodynamic signals
        for k in range(3):
            if t <= np.round(np.max(tX)):
                axs[2,1].plot(tY[tY<=t], Y[tY<=t,k], '-'+colors[k], linewidth=2)
            else:
                axs[2,1].plot(tY[tY<=t], Yn[tY<=t,k], '-'+colors[k], linewidth=2)
        axs[2,1].axis([0-10*dt, np.max(tY)+10*dt, 0-dy, 1+dy])
        axs[2,1].set_yticks([0,1])
        axs[2,1].tick_params(axis='both', labelsize=16)
        axs[2,1].set_xlabel('time [s]', fontsize=24)
        axs[2,1].set_ylabel('activity', fontsize=24)
        if t <= np.round(np.max(tX)):
            axs[2,1].set_title('Hemodynamic Signals', fontsize=28)
        else:
            axs[2,1].set_title('Hemodynamic Signals + Noise', fontsize=28)
    
        # Panel D: logarithmic tuning functions
        labels = ['low', 'high', 'null']
        for k in range(3):
            axs[1,0].plot(xc, ys[k], '-'+colors[k], linewidth=2, label=labels[k])
            if k < 2:
                axs[1,0].plot([x,L+1], [ys[k][np.argmin(abs(xc-x))], ys[k][np.argmin(abs(xc-x))]], '-', color=cols_alt[k], linewidth=2)
                axs[1,0].plot(x, ys[k][np.argmin(abs(xc-x))], 'o'+colors[k], markerfacecolor=colors[k], markersize=10, linewidth=2)
        axs[1,0].axis([0, L+1, 0-dy, 1+dy])
        axs[1,0].set_xticks(xd)
        axs[1,0].legend(loc='upper right', fontsize=12)
        axs[1,0].tick_params(axis='both', labelsize=16)
        axs[1,0].set_xlabel('numerosity', fontsize=24)
        axs[1,0].set_ylabel('activity', fontsize=24)
        axs[1,0].set_title('Tuning Functions', fontsize=28)
        
        # Panel E: hemodynamic response function
        axs[2,0].plot(tH, y, '-m', linewidth=2, label='HRF')
        axs[2,0].axis([0, np.max(tH)+dt, 0-dy, 1+dy])
        axs[2,0].legend(loc='upper right', fontsize=12)
        axs[2,0].tick_params(axis='both', labelsize=16)
        axs[2,0].set_xlabel('post-stimulus time [s]', fontsize=24)
        axs[2,0].set_ylabel('response', fontsize=24)
        return fig
    
    # Figure B: create GIF
    images = []
    ts     = range(76)
    for t in ts:
        fig = Figure_B(t)
        filename = 'Figures_Talk/Talk_Figure_B_temp.png'
        fig.savefig(filename, dpi=150)
        images.append(imageio.imread(filename))
    os.remove(filename)
    imageio.mimsave('Figures_Talk/Talk_Figure_B.gif', images, fps=10, loop=0)
    del fig, images, ts
    
    # Figure B: with noise
    fig = Figure_B(75)
    # fig.savefig('Figures_Talk/Talk_Figure_7.png', dpi=150)
    fig = Figure_B(76)
    # fig.savefig('Figures_Talk/Talk_Figure_8.png', dpi=150)

# function: Work Package 1, Table 0
#-----------------------------------------------------------------------------#
def WP1_Tab0():

    # read participant table
    filename  = EMPRISE.data_dir+'participants.tsv'
    part_info = pd.read_table(filename)
    subs      = EMPRISE.adults + EMPRISE.childs
    
    # preallocate table columns
    group     = []
    gender    = []
    age       = []
    num_runs  = []
    func_runs = []
    anat_dir  = []
    surf_imgs = []
    
    # collect subject information
    for sub in subs:
        
        # get subject info
        sub_info = part_info[part_info['participant_id']=='sub-'+sub]
        if sub in EMPRISE.adults:
            group.append('adult')
        elif sub in EMPRISE.childs:
            group.append('child')
        else:
            group.append('unknown')
        gender.append(sub_info.iloc[0,2])
        age.append(np.mean(np.unique(sub_info['age'])))
        
        # check functional runs
        num_str  = []
        runs_str = []
        for ses in EMPRISE.sess:
            sess = EMPRISE.Session(sub,ses)
            runs = ''
            for run in EMPRISE.runs:
                if os.path.isfile(sess.get_bold_gii(run)):
                    runs = runs + str(run)
            num_str.append(len(runs))
            runs_str.append(runs)
        num_runs.append(num_str)
        func_runs.append(runs_str)
        del num_str, runs_str
        
        # check anatomical directory
        sess       = EMPRISE.Session(sub,'visual')
        mesh_files = sess.get_mesh_files()
        mesh_file  = mesh_files['left']
        anat_fold  = mesh_file[mesh_file.find('sub-')+len('sub-000'): \
                               mesh_file.find('anat/')]
        anat_dir.append(anat_fold)
        del mesh_file, mesh_files
        
        # check surface images
        surfs    = ['inflated','pial','white','midthickness']
        surf_str = []
        for surf in surfs:
            mesh_files = sess.get_mesh_files('fsnative',surf)
            if mesh_files['left'] == 'n/a':
                surf_str.append('no')
            else:
                surf_str.append('yes')
        surf_imgs.append(surf_str)
        del surf_str, mesh_files
        
    # create subject columns
    num_visual_runs = [x[0] for x in num_runs]
    num_audio_runs  = [x[1] for x in num_runs]
    num_digit_runs  = [x[2] for x in num_runs]
    num_spoken_runs = [x[3] for x in num_runs]
    visual_runs     = [s[0] for s in func_runs]
    audio_runs      = [s[1] for s in func_runs]
    digit_runs      = [s[2] for s in func_runs]
    spoken_runs     = [s[3] for s in func_runs]
    infl_mesh       = [b[0] for b in surf_imgs]
    pial_mesh       = [b[1] for b in surf_imgs]
    white_mesh      = [b[2] for b in surf_imgs]
    midthick_mesh   = [b[3] for b in surf_imgs]
    
    # create data frame
    data = zip(subs, group, gender, age, \
               num_visual_runs, num_audio_runs, num_digit_runs, num_spoken_runs, \
               visual_runs, audio_runs, digit_runs, spoken_runs, \
               anat_dir, infl_mesh, pial_mesh, white_mesh, midthick_mesh)
    cols = ['Subject_ID', 'group', 'gender', 'age', \
            'num_visual_runs', 'num_audio_runs', 'num_digit_runs', 'num_spoken_runs', \
            'visual_runs', 'audio_runs', 'digit_runs', 'spoken_runs', \
            'anat_dir', 'inflated_mesh', 'pial_mesh', 'white_mesh', 'midthickness_mesh']
    df = pd.DataFrame(data, columns=cols)
    
    # save data frame to CSV/XLS
    filename = 'subjects.csv'
    df.to_csv(filename, index=False)
    filename = 'subjects.xlsx'
    df.to_excel(filename, index=False)

# function: Work Package 1, Figure 1
#-----------------------------------------------------------------------------#
def WP1_Fig1(Figure):
    
    # sub-function: plot tuning functions & time courses
    #-------------------------------------------------------------------------#
    def plot_tuning_function_time_course(sub, ses, model, hemi, space, verts=[0,0], col='b'):
        
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
        
        # load vertex-wise coordinates
        hemis     = {'L': 'left', 'R': 'right'}
        para_map  = res_file[:res_file.find('numprf.mat')] + 'mu.surf.gii'
        para_mask = nib.load(para_map).darrays[0].data != 0
        mesh_file = mod.get_mesh_files(space,'pial')[hemis[hemi]]
        mesh_gii  = nib.load(mesh_file)
        XYZ       = mesh_gii.darrays[0].data
        XYZ       = XYZ[para_mask,:]
        del para_map, para_mask, mesh_file, mesh_gii
        
        # calculate vertex-wise R-squared
        MLL1 = np.squeeze(NpRF['MLL_est'])
        MLL0 = np.squeeze(NpRF['MLL_const'])
        n    = np.prod(mod.calc_runs_scans())
        Rsq  = NumpRF.MLL2Rsq(MLL1, MLL0, n)
        
        # select vertices for plotting
        for k, vertex in enumerate(verts):
            if vertex == 0:
                if k == 0 and len(verts) == 1:
                    # select maximum R^2 in vertices with numerosity 1 < mu < 5
                    vertex = np.argmax(Rsq + np.logical_and(mu>1, mu<5))
                elif k == 0 and len(verts) == 2:
                    # select maximum R^2 in vertices with numerosity 1 < mu < 2
                    vertex = np.argmax(Rsq + np.logical_and(mu>1, mu<2))
                elif k == 1:
                    # select maximum R^2 in vertices with numerosity 4 < mu < 5 or 3 < mu < 5
                    if ses == 'visual': vertex = np.argmax(Rsq + np.logical_and(mu>4, mu<5))
                    if ses == 'audio':  vertex = np.argmax(Rsq + np.logical_and(mu>3, mu<5))
                verts[k] = vertex
        
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
            mu_log, sig_log = NumpRF.lin2log(mu[vertex], fwhm[vertex])
            x  = np.arange(dx, xm+dx, dx)
            y  = NumpRF.f_log(x, mu_log, sig_log)
            xM = mu[vertex]
            x1 = np.exp(mu_log - math.sqrt(2*math.log(2))*sig_log)
            x2 = np.exp(mu_log + math.sqrt(2*math.log(2))*sig_log)
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
                axs[k,0].text(xM, 0+0.01, ' mu = preferred numerosity', fontsize=16, \
                              horizontalalignment='left', verticalalignment='bottom')
                axs[k,0].text((x1+x2)/2, 0.5-0.01, 'fwhm = tuning width', fontsize=16, \
                              horizontalalignment='center', verticalalignment='top')
            axs[k,0].axis([0, xm, 0, 1+(1/20)])
            if k == len(verts)-1:
                axs[k,0].set_xlabel('presented numerosity', fontsize=20)
            axs[k,0].set_ylabel('normalized neuronal response', fontsize=20)
            if k == 0:
                axs[k,0].set_title(hdr, fontweight='bold', fontsize=20)
            axs[k,0].tick_params(axis='both', labelsize=16)
            axs[k,0].text(xm-(1/20)*xm, 0.85, txt1, fontsize=16,
                          horizontalalignment='right', verticalalignment='top')
            axs[k,0].text(xm-(1/20)*xm, 0.05, txt2, fontsize=16,
                          horizontalalignment='right', verticalalignment='bottom')
            del mu_log, sig_log, x, y, xM, x1, x2
        
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
            mu_log, sig_log = NumpRF.lin2log(mu[vertex], fwhm[vertex])
            
            # compute predicted signal (run)
            y_run, t = EMPRISE.average_signals(y_reg, None, [True, False])
            z, t = NumpRF.neuronal_signals(ons0, dur0, stim0, EMPRISE.TR, EMPRISE.mtr, np.array([mu_log]), np.array([sig_log]))
            s, t = NumpRF.hemodynamic_signals(z, t, y.shape[0], EMPRISE.mtr)
            glm  = PySPM.GLM(y_run, np.c_[s[:,:,0], np.ones((y.shape[0], 1))])
            b_run= glm.OLS()
            s_run= glm.X @ b_run
            
            # compute predicted signal (epoch)
            y_avg, t = EMPRISE.average_signals(y_reg, None, [True, True])
            # Note: For visualization purposes, we here apply "avg = [True, True]".
            z, t = NumpRF.neuronal_signals(ons, dur, stim, EMPRISE.TR, EMPRISE.mtr, np.array([mu_log]), np.array([sig_log]))

            if y_avg.shape[0] == EMPRISE.scans_per_epoch:
                s, t = NumpRF.hemodynamic_signals(z, t, EMPRISE.scans_per_epoch, EMPRISE.mtr)
                glm  = PySPM.GLM(y_avg, np.c_[s[:,:,0], np.ones((EMPRISE.scans_per_epoch, 1))])
                b_avg= glm.OLS()
                s_avg= glm.X @ b_avg
            else:
                short_scan_num = y_avg.shape[0]
                s, t = NumpRF.hemodynamic_signals(z, t, short_scan_num, EMPRISE.mtr)
                glm  = PySPM.GLM(y_avg, np.c_[s[:,:,0], np.ones((short_scan_num, 1))])
                b_avg= glm.OLS()
                s_avg= glm.X @ b_avg                
            
            # assess statistical significance
            Rsq_run = Rsq[vertex]
            Rsq_avg = NumpRF.yp2Rsq(y_avg, s_avg)[0]
            h, p_run, stats = NumpRF.Rsqtest(y_run, s_run, p=4)
            h, p_avg, stats = NumpRF.Rsqtest(y_avg, s_avg, p=4)
            
            # prepare axis limits
            y_min = np.min(y_avg)
            y_max = np.max(y_avg)
            y_rng = y_max-y_min
            t_max = np.max(t)
            xM    = t[np.argmax(s_avg)]
            yM    = np.max(s_avg)
            y0    = b_avg[1,0]
            
            # plot hemodynamic signals signals
            txt = 'beta = {:.2f}\nR² = {:.2f}, {}\n '. \
                   format(beta[vertex], Rsq_run, pvalstr(p_run))

            # Note: For visualization purposes, we here apply "avg = [True, True]".
            axs[k,1].plot(t, y_avg[:,0], ':ok', markerfacecolor='k', markersize=8, linewidth=1, label='measured signal')
            axs[k,1].plot(t, s_avg[:,0], '-'+col, linewidth=2, label='predicted signal')
            for i in range(len(ons)):
                axs[k,1].plot(np.array([ons[i],ons[i]]), \
                              np.array([y_max+(1/20)*y_rng, y_max+(3/20)*y_rng]), '-k')
                axs[k,1].text(ons[i]+(1/2)*dur[i], y_max+(2/20)*y_rng, str(stim[i]), fontsize=16,
                              horizontalalignment='center', verticalalignment='center')
            axs[k,1].plot(np.array([ons[-1]+dur[-1],ons[-1]+dur[-1]]), \
                          np.array([y_max+(1/20)*y_rng, y_max+(3/20)*y_rng]), '-k')
            axs[k,1].plot(np.array([ons[0],ons[-1]+dur[-1]]), \
                          np.array([y_max+(1/20)*y_rng, y_max+(1/20)*y_rng]), '-k')
            if k == 0:
                axs[k,1].plot([xM,xM], [y0,yM], '-', color='gray', linewidth=2)
                axs[k,1].text(xM, (y0+yM)/2, 'beta = scaling factor', fontsize=16, \
                              horizontalalignment='right', verticalalignment='center', rotation=90)
                axs[k,1].legend(loc='lower right', fontsize=16)
            axs[k,1].axis([0, t_max, y_min-(1/20)*y_rng, y_max+(3/20)*y_rng])
            if k == len(verts)-1:
                axs[k,1].set_xlabel('within-epoch time [s]', fontsize=20)
            axs[k,1].set_ylabel('measured hemodynamic signal [%]', fontsize=20)
            if k == 0:
                axs[k,1].set_title('presented numerosity', fontsize=20)
            axs[k,1].tick_params(axis='both', labelsize=16)
            axs[k,1].text((1/20)*t_max, y_min-(1/20)*y_rng, txt, fontsize=16, \
                          horizontalalignment='left', verticalalignment='bottom')
            del y, y_reg, y_avg, z, s, s_avg, t, b_reg, b_avg, mu_log, sig_log, y_min, y_max, y_rng, txt
            
        # return figure
        return fig
    
    # define globals
    sub_visual = '105'
    sub_audio  = '113'
    subs_all   = EMPRISE.childs
    model      = 'NumSig' # Log grid, [True,False] i.d.d, hemo order 1, Rsq
    space      = 'fsnative'
    fig_path = '/data/u_jeong_software/EMPRISE/code/Python/Figures_WP3/'
    
    # Figure 1
    if Figure == '1':
        
        # Figure 1, Part 1: visual data
        fig = plot_tuning_function_time_course(sub_visual, 'visual', model, 'L', space, verts=[0,0], col='b')
        fig.savefig(fig_path + 'WP3_Figure_1A_ses-visual.png', dpi=150)
        
        # Figure 1, Part 2: auditory data
        fig = plot_tuning_function_time_course(sub_audio, 'audio', model, 'L', space, verts=[0,0], col='r')
        fig.savefig(fig_path + 'WP3_Figure_1_ses-audio.png', dpi=150)
    
    # Figure S1
    if Figure == 'S1':
    
        # Figure S1: all subjects, visual, Left hemi
        for sub in subs_all:
            fig = plot_tuning_function_time_course(sub, 'visual', model, 'L', space, verts=[0,0], col='b')
            filename = fig_path + 'Fig1_visual/WP3_Figure_S1'+'_ses-'+'visual'+'_sub-'+sub +'_hemi-L_' + model +'.png'
            fig.savefig(filename, dpi=150)
    
    # Figure S1_R
    if Figure == 'S1R':
    
        # Figure S1: all subjects, visual, Right hemi
        for sub in subs_all:
            fig = plot_tuning_function_time_course(sub, 'visual', model, 'R', space, verts=[0,0], col='b')
            filename = fig_path + 'Fig1_visual/WP3_Figure_S1'+'_ses-'+'visual'+'_sub-'+sub+ '_hemi-R_'+ model +'.png'
            fig.savefig(filename, dpi=150)
    
    # Figure S2
    if Figure == 'S2':
    
        # Figure S2: all subjects, audio, Left hemi
        for sub in subs_all:
            fig = plot_tuning_function_time_course(sub, 'audio', model, 'L', space, verts=[0,0], col='r')
            filename = fig_path + 'Fig1_audio/WP3_Figure_S2'+'_ses-'+'audio'+'_sub-'+sub +'_hemi-L_' + model +'.png'
            fig.savefig(filename, dpi=150)
    
    # Figure S2
    if Figure == 'S2R':
    
        # Figure S2: all subjects, audio, Right hemi
        for sub in subs_all:
            fig = plot_tuning_function_time_course(sub, 'audio', model, 'R', space, verts=[0,0], col='r')
            filename = fig_path + 'Fig1_audio/WP3_Figure_S2'+'_ses-'+'audio'+'_sub-'+sub +'_hemi-R_' + model +'.png'
            fig.savefig(filename, dpi=150)


    # Creat figure of a specific condition stated in a list [sub, ses, model, hemi]
    # e.g: Figures.WP1_Fig1(['101','visual','NumSig','L']) = A figure of sub '101', ses 'visual', model 'NumSig', hemi 'L'
    if type(Figure) is list:
    
        # Use the given condition
        
        fig = plot_tuning_function_time_course(Figure[0], Figure[1], Figure[2], Figure[3], space, verts=[0,0], col='b')
        filename = fig_path + 'Fig1_visual/WP3_Figure_S1'+'_ses-'+Figure[1]+'_sub-'+ Figure[0] +'_hemi-'+ Figure[3] + '_' + Figure[2] +'.png'
        fig.savefig(filename, dpi=150)    

# function: Work Package 1, Figure 2
#-----------------------------------------------------------------------------#
def WP1_Fig2(Figure):
    
    # sub-function: plot surface parameter map
    #-------------------------------------------------------------------------#
    def plot_surface_para(sub, ses, model, space, para='mu', Rsq_thr=None):
        
        # load analysis details
        mod = EMPRISE.Model(sub, ses, model, space)
        n   = np.prod(mod.calc_runs_scans())# effective number of
                                            # observations in model
        # specify thresholds
        if Rsq_thr is None:                 # for details, see
            Rsq_thr = EMPRISE.Rsq_def       # EMPRISE global variables
        mu_thr   = EMPRISE.mu_thr
        fwhm_thr = EMPRISE.fwhm_thr
        beta_thr = EMPRISE.beta_thr
        crit     = 'Rsqmb,'+str(Rsq_thr)
        
        # analyze hemispheres
        hemis = {'L': 'left', 'R': 'right'}
        maps  = {}
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
            MLL1 = np.squeeze(NpRF['MLL_est'])
            MLL0 = np.squeeze(NpRF['MLL_const'])
            
            # compute quantities for thresholding
            Rsq   = NumpRF.MLL2Rsq(MLL1, MLL0, n)
            ind_m = np.logical_or(mu<mu_thr[0], mu>mu_thr[1])
            ind_f = np.logical_or(fwhm<fwhm_thr[0], fwhm>fwhm_thr[1])
            ind_b = np.logical_or(beta<beta_thr[0], beta>beta_thr[1])
            
            # apply conditions for exclusion
            ind = mu > np.inf
            if 'Rsq' in crit:
                ind = np.logical_or(ind, Rsq<Rsq_thr)
            if 'm' in crit:
                ind = np.logical_or(ind, ind_m)
            if 'f' in crit:
                ind = np.logical_or(ind, ind_f)
            if 'b' in crit:
                ind = np.logical_or(ind, ind_b)
            
            # threshold parameter map
            para_est       = {'mu': mu, 'fwhm': fwhm, 'beta': beta, 'Rsq': Rsq}
            para_map       = np.nan * np.ones(mask.size, dtype=np.float32)
            para_crit      = para_est[para]
            para_crit[ind] = np.nan
            para_map[mask] = para_crit
            filename       = filepath + para + '_thr-' + crit + '.surf.gii'
            para_img       = EMPRISE.save_surf(para_map, image, filename)
            maps[hemis[hemi]] = filename
            del para_crit, para_map, para_est, image, filename
        
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
        elif para == 'Rsq':
            caxis  = [Rsq_thr,1]
            cmap   = 'hot'
            clabel = 'variance explained (R²)'
            if Rsq_thr == 0.1:              # 0.1, 0.4, 0.7, 1.0
                cbar   = {'n_ticks': 4, 'decimals': 1}
            elif Rsq_thr == 0.15:           # 0.15, 0.32, 0.49, 0.66, 0.83, 1.00
                cbar   = {'n_ticks': 6, 'decimals': 2}
            elif Rsq_thr == 0.2:            # 0.2, 0.4, 0.6, 0.8, 1.0
                cbar   = {'n_ticks': 5, 'decimals': 1}
            elif Rsq_thr == 0.25:           # 0.25, 0.50, 0.75, 1.00
                cbar   = {'n_ticks': 4, 'decimals': 2}
            elif Rsq_thr == 0.3:            # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
                cbar   = {'n_ticks': 8, 'decimals': 1}
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
                     size=(1600,600), zoom=1.5)
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
    def plot_participant_count(subs, ses, model, Rsq_thr=None):
        
        # specify thresholds
        if Rsq_thr is None:                 # for details, see
            Rsq_thr = EMPRISE.Rsq_def       # EMPRISE global variables
        mu_thr   = EMPRISE.mu_thr
        fwhm_thr = EMPRISE.fwhm_thr
        beta_thr = EMPRISE.beta_thr
        crit     = 'Rsqmb,'+str(Rsq_thr)
        
        # prepare loading
        N     = len(subs)
        maps  = [{} for i in range(N)]
        hemis = {'L': 'left', 'R': 'right'}
        space = 'fsaverage'
        para  = 'Rsq'
        
        # load all subjects
        for i, sub in enumerate(subs):
            
            # load analysis details
            mod = EMPRISE.Model(sub, ses, model, space)
            n   = np.prod(mod.calc_runs_scans())# effective number of
                                                # observations in model
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
                MLL1 = np.squeeze(NpRF['MLL_est'])
                MLL0 = np.squeeze(NpRF['MLL_const'])
                
                # compute quantities for thresholding
                Rsq   = NumpRF.MLL2Rsq(MLL1, MLL0, n)
                ind_m = np.logical_or(mu<mu_thr[0], mu>mu_thr[1])
                ind_f = np.logical_or(fwhm<fwhm_thr[0], fwhm>fwhm_thr[1])
                ind_b = np.logical_or(beta<beta_thr[0], beta>beta_thr[1])
                
                # apply conditions for exclusion
                ind = mu > np.inf
                if 'Rsq' in crit:
                    ind = np.logical_or(ind, Rsq<Rsq_thr)
                if 'm' in crit:
                    ind = np.logical_or(ind, ind_m)
                if 'f' in crit:
                    ind = np.logical_or(ind, ind_f)
                if 'b' in crit:
                    ind = np.logical_or(ind, ind_b)
                
                # threshold R-squared map
                para_est       = {'mu': mu, 'fwhm': fwhm, 'beta': beta, 'Rsq': Rsq}
                para_map       = np.nan * np.ones(mask.size, dtype=np.float32)
                para_crit      = para_est[para]
                para_crit[ind] = np.nan
                para_map[mask] = para_crit
                maps[i][hemis[hemi]] = para_map
                del para_crit, para_map, para_est
        
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
                    size=(1600,600), zoom=1.5)
        plot.add_layer(sulc_data, color_range=(0,1),
                       cmap='Greys', cbar=False)
        plot.add_layer(surf_imgs, color_range=(caxis[0],caxis[1]),
                       cmap=cmap, cbar_label=clabel)
        
        # display surface plot
        fig = plot.build(colorbar=True, cbar_kws=cbar)
        fig.tight_layout()
        return fig
    
    # define globals
    sub_visual = '105'
    sub_audio  = '113'
    subs_all   = EMPRISE.childs
    sess       =['visual', 'audio']
    model      = 'NumSig' #'True_False_i.i.d_1', with the log fwhm grid
    space      = 'fsnative'
    Rsq_thr    = 0.2
    fig_path = '/data/u_jeong_software/EMPRISE/code/Python/Figures_WP3/'
    
    # Figure 2A
    if Figure == '2A':
        
        # Figure 2A, Part 1: visual data
        fig = plot_surface_para(sub_visual, 'visual', model, space, 'Rsq', Rsq_thr)
        fig.savefig(fig_path + 'Fig2/WP3_Figure_2A_ses-visual'+ '_' + model+'.png', dpi=150)
        
        # Figure 2A, Part 2: auditory data
        fig = plot_surface_para(sub_audio, 'audio', model, space, 'Rsq', Rsq_thr)
        fig.savefig(fig_path + 'Fig2/WP3_Figure_2A_ses-audio'+ '_' + model+'.png', dpi=150)
    
    # Figure 2B
    if Figure == '2B':
        
        # Figure 2B, Part 1: visual data
        fig  = plot_participant_count(subs_all, 'visual', model, Rsq_thr)
        fig.savefig(fig_path + 'Fig2/WP3_Figure_2B_ses-visual'+ '_' + model+'.png', dpi=150)
        
        # Figure 2B, Part 2: audio data
        fig  = plot_participant_count(subs_all, 'audio', model, Rsq_thr)
        fig.savefig(fig_path + 'Fig2/WP3_Figure_2B_ses-audio'+ '_' + model+'.png', dpi=150)
    
    # Figure 2C
    if Figure == '2C':
        
        # Figure 2C, Part 1: visual data
        fig = plot_surface_para(sub_visual, 'visual', model, space, 'mu', Rsq_thr)
        fig.savefig(fig_path + 'Fig2/WP3_Figure_2C_ses-visual'+ '_' + model+'.png', dpi=150)
        
        # Figure 2C, Part 2: auditory data
        fig = plot_surface_para(sub_audio, 'audio', model, space, 'mu', Rsq_thr)
        fig.savefig(fig_path + 'Fig2/WP3_Figure_2C_ses-audio'+ '_' + model+'.png', dpi=150)
    
    # Figure S3
    if Figure == 'S3':
    
        # Figure S3: all subjects, visual & audio
        for sub in subs_all:
            for ses in sess:
                fig = plot_surface_para(sub, ses, model, space, 'Rsq', Rsq_thr)
                filename = fig_path + 'Fig2/WP3_Figure_S3'+'_ses-'+ses+'_sub-'+sub+ '_'+ model+'.png'
                fig.savefig(filename, dpi=150)
    
    # Figure S4
    if Figure == 'S4':
    
        # Figure S4: all subjects, visual & audio
        for sub in subs_all:
            for ses in sess:
                fig = plot_surface_para(sub, ses, model, space, 'mu', Rsq_thr)
                filename = fig_path + 'Fig2/WP3_Figure_S4'+'_ses-'+ses+'_sub-'+sub+ '_' + model+'.png'
                fig.savefig(filename, dpi=150)

# function: Work Package 1, Figure 3
#-----------------------------------------------------------------------------#
def WP1_Fig3(Figure):
    
    # class: Figure 3 object
    #-------------------------------------------------------------------------#
    class Fig3_Obj():
        
        # function: initialize Figure 3 object
        #---------------------------------------------------------------------#
        def __init__(self, res_dir, ses, space, mesh, AFNI=False):
            
            # store session, space, mesh
            self.res_dir = res_dir
            self.ses     = ses
            self.space   = space
            self.mesh    = mesh
            self.AFNI    = AFNI
            self.pref    = ['','AFNI_'][int(AFNI)]
        
        # sub-function: extract surface clusters
        #---------------------------------------------------------------------#
        def extract_surface_clusters(self, crit='Rsqmb', ctype='coords', d=3, k=100):
            
            # specify analysis
            subs  = EMPRISE.childs
            model = 'NumSig'
            
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
                mod      = EMPRISE.Model(sub, self.ses, model, self.space)
                res_file = mod.get_results_file('L')
                if os.path.isfile(res_file):
                    
                    # if AFNI edge clustering
                    if self.AFNI:
                        # extract vertices and triangles
                        verts[sub], trias[sub] = \
                            mod.threshold_AFNI_cluster(crit, self.mesh)
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
            filepath1 = self.res_dir+'sub-children'+'_space-'+self.space+'_'
            filepath2 = self.res_dir+'sub-children'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
            sp.io.savemat(filepath1+'trias.mat', trias)
            sp.io.savemat(filepath2+'verts.mat', verts)
            sp.io.savemat(filepath2+'areas.mat', areas)
        
        # sub-function: plot area vs. map
        #---------------------------------------------------------------------#
        def plot_area_vs_map(self, subs, cols=['b','r']):
            
            # load results
            filepath = self.res_dir+'sub-children'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'
            areas    = sp.io.loadmat(filepath +'AFNI_areas.mat')
    
            # get surface areas
            N = len(subs)
            A = np.zeros((N,len(hemis)))
            for i, sub in enumerate(subs):
                for j, hemi in enumerate(hemis):
                    areas_sub = areas[sub][hemi][0,0]
                    A[i,j]    = np.sum(areas_sub[:,6])
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            ax  = fig.add_subplot(111)
            
            # plot area vs. map
            hdr = 'ses-{}'.format(self.ses)
            ax.barh(np.arange(N), -A[:,0], color=cols[0], edgecolor='k', label='hemi-'+hemis[0])
            ax.barh(np.arange(N), +A[:,1], color=cols[1], edgecolor='k', label='hemi-'+hemis[1])
            ax.legend(loc='upper right', fontsize=16)
            xt  = ax.get_xticks()
            xtl = []
            for x in xt:
                xtl.append(str(int(abs(x))))
            ax.set_xticks(xt, labels=xtl)
            ax.set_yticks(np.arange(N), labels=subs)
            ax.tick_params(axis='both', labelsize=16)
            ax.axis([-(11/10)*np.max(A), +(11/10)*np.max(A), 0-1, N])
            ax.invert_yaxis()
            ax.set_xlabel('surface area [mm²]', fontsize=16)
            ax.set_ylabel('subject ID', fontsize=16)
            ax.set_title(hdr, fontweight='bold', fontsize=20)
            
            # display figure
            fig.show()
            return fig
        
        # sub-function: plot area vs. mu
        #---------------------------------------------------------------------#
        def plot_area_vs_mu(self, sub, d_mu=0.5, cols=['b','r']):
            
            # load results
            filepath = self.res_dir+'sub-children'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+'AFNI_'
            verts    = sp.io.loadmat(filepath+'verts.mat')
            areas    = sp.io.loadmat(filepath+'areas.mat')
            
            # specify mu grid
            mu_min = EMPRISE.mu_thr[0]
            mu_max = EMPRISE.mu_thr[1]
            mu_b   = np.arange(mu_min, mu_max+d_mu, d_mu)
            mu_c   = np.arange(mu_min+d_mu/2, mu_max+d_mu/2, d_mu)
            
            # preallocate results
            area_s = np.zeros((len(hemis),mu_c.size))
            r_a    = np.zeros(len(hemis))
            p_a    = np.zeros(len(hemis))
            b_a    = np.zeros((2,len(hemis)))
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
                            area_s[j,k] = np.nan
                    
                    # calculate regression lines
                    ind_j  =~np.isnan(area_s[j,:])
                    n_a[j] = np.sum(ind_j)
                    r_a[j], p_a[j], b_a[0,j], b_a[1,j] = \
                        simplinreg(area_s[j,ind_j], mu_c[ind_j])
                    del ind_k, ind_j
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            ax  = fig.add_subplot(111)
    
            # plot area vs. mu
            hdr = 'sub-{}, ses-{}'.format(sub, self.ses)
            for j, hemi in enumerate(hemis):
                if np.any(area_s[j,:]):
                    lab_j = 'hemi-'+hemi+' (r = {:.2f}, {}, n = {})'. \
                             format(r_a[j], pvalstr(p_a[j]), n_a[j])
                    ax.plot(mu_c, area_s[j,:], 'o', \
                            color=cols[j], markerfacecolor=cols[j], markersize=10)
                    ax.plot([mu_min,mu_max], np.array([mu_min,mu_max])*b_a[0,j]+b_a[1,j], '-', \
                            color=cols[j], label=lab_j)
            if (sub == '003' and self.ses == 'visual') or (sub == '009' and self.ses == 'audio'):
                if self.ses == 'visual': y_max = 600
                if self.ses == 'audio':  y_max = 20
                ax.axis([mu_min-d_mu, mu_max+d_mu, 0-(1/20)*y_max, y_max+(1/20)*y_max])
            else:
                ax.set_xlim(mu_min-d_mu, mu_max+d_mu)
            ax.legend(loc='upper right', fontsize=16)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('preferred numerosity', fontsize=16)
            ax.set_ylabel('cortical surface area [mm²]', fontsize=16)
            ax.set_title(hdr, fontweight='bold', fontsize=20)
            
            # display figure
            fig.show()
            return fig
        
        # sub-function: plot fwhm vs. mu
        #---------------------------------------------------------------------#
        def plot_fwhm_vs_mu(self, sub, d_mu=0.5, cols=['b','r']):
            
            # load results
            filepath = self.res_dir+'sub-children'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+ 'AFNI_'
            verts    = sp.io.loadmat(filepath+'verts.mat')
            
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
            if (sub == '003' and self.ses == 'visual') or (sub == '009' and self.ses == 'audio'):
                if self.ses == 'visual': y_min =  0; y_max =  20;
                if self.ses == 'audio':  y_min = 25; y_max = 125;
                ax.axis([mu_min-d_mu, mu_max+d_mu, y_min, y_max])
            else:
                ax.set_xlim(mu_min-d_mu, mu_max+d_mu)
            ax.legend(loc='upper left', fontsize=16)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('preferred numerosity', fontsize=16)
            ax.set_ylabel('FWHM tuning width', fontsize=16)
            ax.set_title(hdr, fontweight='bold', fontsize=20)
            
            # display figure
            fig.show()
            return fig
        
        # sub-function: plot topography
        #---------------------------------------------------------------------#
        def plot_topography(self, sub, cols=['b','r']):
            
            # load results
            filepath = self.res_dir+'sub-children'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+'AFNI_'
            verts    = sp.io.loadmat(filepath+'verts.mat')
            areas    = sp.io.loadmat(filepath+'areas.mat')
            clusts   = {}
            
            # for both hemispheres
            for j, hemi in enumerate(hemis):
                
                # get supra-threshold vertices/triangles
                verts_sub = verts[sub][hemi][0,0]
                areas_sub = areas[sub][hemi][0,0]
                verts_cls = verts_sub[:,1].astype(np.int32)
                areas_cls = areas_sub[:,1].astype(np.int32)
                
                # if supra-threshold vertices exist
                if verts_sub.shape[0] > 0:
                    
                    # preallocate statistics
                    num_clust = np.max(verts_cls)
                    XYZ_m     = np.zeros((num_clust,3))
                    area_s    = np.zeros(num_clust)
                    Rsq_XYZ   = np.zeros(num_clust)
                    p_XYZ     = np.zeros(num_clust)
                    
                    # go through surface clusters
                    for k in range(num_clust):
                        
                        # calculate cluster center and surface area
                        XYZ_m[k,:] = np.mean(verts_sub[verts_cls==k+1,6:9], axis=0)
                        area_s[k]  = np.sum(areas_sub[areas_cls==k+1,6])
                        
                        # determine cortical progression of numerosity
                        try:
                            y = verts_sub[verts_cls==k+1,2:3]
                            X = verts_sub[verts_cls==k+1,6:9]
                            X = X - np.tile(XYZ_m[k,:], (X.shape[0],1))
                            X = np.c_[X, np.ones((X.shape[0],1))]
                          # X = np.c_[X, np.power(X,2), np.ones((X.shape[0],1))]
                            MLL1 = PySPM.GLM(y, X).MLL()
                            MLL0 = PySPM.GLM(y, X[:,-1:]).MLL()
                            n, p               = X.shape
                            Rsq_XYZ[k]         = NumpRF.MLL2Rsq(MLL1, MLL0, n)
                            h, p_XYZ[k], stats = NumpRF.Rsqtest(y, X @ PySPM.GLM(y,X).OLS(), p=X.shape[1])
                            del y, X, MLL1, MLL0, n, p, h, stats
                        except:
                            print('-> Subject "{}", Cluster "{}-{}": not enough vertices!'. \
                                  format(sub, hemi, k+1))
                        
                    # store hemisphere results
                    clusts[hemi] = {'center': XYZ_m, 'area': area_s, \
                                    'Rsq': Rsq_XYZ, 'p': p_XYZ}
                    
                    # analyze most significant cluster
                    try:
                        k = np.argmax(Rsq_XYZ)
                        y = verts_sub[verts_cls==k+1,2:3]
                        X = verts_sub[verts_cls==k+1,6:9]
                        X = X - np.tile(XYZ_m[k,:], (X.shape[0],1))
                        X = np.c_[X, np.ones((X.shape[0],1))]
                      # X = np.c_[X, np.power(X,2), np.ones((X.shape[0],1))]
                        yp= y - PySPM.GLM(y, X).regress()
                        r, p, b1, b0         = simplinreg(y[:,0], yp[:,0])
                        clusts[hemi]['max']  = (k, r, p, b1, b0, X.shape[0])
                        clusts[hemi]['Ymax'] = np.c_[yp, y]
                        del y, X, yp, r, p, b1, b0
                    except:
                        print('-> Subject "{}", Cluster "{}-{}": could not analyze!'. \
                              format(sub, hemi, k+1))
                
                # if no supra-threshold vertices exist
                else:
                    
                    # store empty dictionary
                    clusts[hemi] = {}
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            ax  = fig.add_subplot(111)
    
            # plot actual vs. predicted
            hdr    = 'sub-{}, ses-{}'.format(sub, self.ses)
            mu_thr = EMPRISE.mu_thr
            y_min  = mu_thr[0]
            y_max  = mu_thr[1]
            for j, hemi in enumerate(hemis):
                if bool(clusts[hemi]):
                    k, r, p, b1, b0, n = clusts[hemi]['max']
                    Y                  = clusts[hemi]['Ymax']
                    lab_j = 'hemi-'+hemi+', cls-'+str(k+1)+ \
                            ' (r = {:.2f}, {}, n = {})'.format(r, pvalstr(p), n)
                    ax.plot(Y[:,0], Y[:,1], 'o', \
                            color=cols[j], markerfacecolor=cols[j], markersize=3)
                    ax.plot(mu_thr, np.array(mu_thr)*b1+b0, '-', \
                            color=cols[j], label=lab_j)
            y_rng = y_max - y_min
            ax.axis([y_min-(1/10)*y_rng, y_max+(1/10)*y_rng, \
                     y_min-(1/10)*y_rng, y_max+(1/10)*y_rng])
            ax.legend(loc='upper left', fontsize=16)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('fitted preferred numerosity, based on cortical surface coordinates', fontsize=16)
            ax.set_ylabel('actual preferred numerosity', fontsize=16)
            ax.set_title(hdr, fontweight='bold', fontsize=20)
            
            # display figure
            fig.show()
            return fig
            
            # plot R-squared vs. area
            # fig = plt.figure(figsize=(16,18))
            # ax  = fig.add_subplot(111)
            # hdr = 'sub-{}, ses-{}'.format(sub, self.ses)
            # for j, hemi in enumerate(hemis):
            #     if bool(clusts[hemi]):
            #         k     = clusts[hemi]['max'][0]
            #         XYZ_m = clusts[hemi]['center']
            #         lab_j = 'hemi-'+hemi+' (k = {})'.format(clusts[hemi]['area'].size)
            #         txt_j = '  cls-'+str(k+1)+' (XYZ = [{:.1f}, {:.1f}, {:.1f}])'. \
            #                  format(XYZ_m[k,0], XYZ_m[k,1], XYZ_m[k,2])
            #         ax.plot(clusts[hemi]['area'], clusts[hemi]['Rsq'], 'o', \
            #                 color=cols[j], markerfacecolor=cols[j], markersize=10, label=lab_j)
            #         ax.text(clusts[hemi]['area'][k], clusts[hemi]['Rsq'][k], txt_j, \
            #                 horizontalalignment='left', verticalalignment='bottom', fontsize=16)
            # ax.axis([(10/11)*np.min(clusts[hemi]['area']), (11/10)*np.max(clusts[hemi]['area']), 0, 1])
            # ax.set_xscale('log')
            # ax.legend(loc='upper right', fontsize=16)
            # ax.tick_params(axis='both', labelsize=16)
            # ax.set_xlabel('cortical surface area [mm²]', fontsize=16)
            # ax.set_ylabel('expression of topography [R²]', fontsize=16)
            # ax.set_title(hdr, fontweight='bold', fontsize=20)
        
        # sub-function: plot range vs. map
        #---------------------------------------------------------------------#
        def plot_range_vs_map(self, sub, cols=['b','r']):
            
            # load results
            filepath = self.res_dir+'sub-children'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+'AFNI_'
            verts    = sp.io.loadmat(filepath+'verts.mat')
            clusts   = {}
            
            # for both hemispheres
            for j, hemi in enumerate(hemis):
                
                # get supra-threshold vertices/triangles
                verts_sub = verts[sub][hemi][0,0]
                verts_cls = verts_sub[:,1].astype(np.int32)
                
                # if supra-threshold vertices exist
                if verts_sub.shape[0] > 0:
                    
                    # preallocate statistics
                    num_clust = np.max(verts_cls)
                    mu_clust  = [[] for k in range(num_clust)]
                    XYZ_m     = np.zeros((num_clust,3))
                    
                    # go through surface clusters
                    for k in range(num_clust):
                        
                        # calculate cluster center
                        XYZ_m[k,:] = np.mean(verts_sub[verts_cls==k+1,6:9], axis=0)
                        
                        # extract preferred numerosities
                        mu_clust[k] = verts_sub[verts_cls==k+1,2]
                        
                    # store hemisphere results
                    clusts[hemi] = {'center': XYZ_m, 'mu': mu_clust}
                    
                # if no supra-threshold vertices exist
                else:
                    
                    # store empty dictionary
                    clusts[hemi] = {}
            
            # open figure
            fig = plt.figure(figsize=(16,9))
            ax  = fig.add_subplot(111)
    
            # plot mu vs. cluster
            hdr = 'sub-{}, ses-{}'.format(sub, self.ses)
            lab = []; c = 0;
            for j, hemi in enumerate(hemis):
                if bool(clusts[hemi]):
                    num_clust = len(clusts[hemi]['mu'])
                    y  = clusts[hemi]['mu']
                    x  = range(c,c+num_clust)
                    c  = c+num_clust
                    bp = ax.boxplot(y, positions=x, widths=0.6, \
                                    sym='+k', notch=True, patch_artist=True)
                    for k in range(len(bp['boxes'])):
                        bp['boxes'][k].set_facecolor(cols[j])
                        bp['medians'][k].set_color('k')
                        lab.append(hemi+str(k+1))
            ax.axis([(0-1), c, EMPRISE.mu_thr[0]-0.5, EMPRISE.mu_thr[1]+0.5])
            ax.set_xticks(np.arange(c), labels=lab)
            ax.legend(loc='upper right', fontsize=16)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('hemisphere and cluster', fontsize=16)
            ax.set_ylabel('preferred numerosities', fontsize=16)
            ax.set_title(hdr, fontweight='bold', fontsize=20)
            
            # display figure
            fig.show()
            return fig
    
    # if input is '3', extract Figure data
    if Figure == '3':
    
        # specify extraction
        sess   = ['visual', 'audio']
        spaces = EMPRISE.spaces
        meshs  = EMPRISE.meshs
        crit   = EMPRISE.crit_def
        ctype  = 'coords'       # distance clustering
        d_mm   = 1.7            # maximum distance to cluster ~ voxel resolution
        k_min  = 50             # minimum number of vertices in cluster = 50
        AFNI   = True           # edge clustering
        
        # extract clusters
        for ses in sess:
            for space in spaces:
                for mesh in meshs:
                    # if not (space == 'fsaverage' and mesh == 'midthickness'):
                    f3 = Fig3_Obj(res_dir, ses, space, mesh, AFNI)
                    if AFNI:
                        f3.extract_surface_clusters(crit)
                    else:
                        f3.extract_surface_clusters(crit, ctype, d_mm, k_min)
    
    # otherwise, create Figure panels
    else:
        
        # define globals
        sub_visual = '113' # Based on the surface area calculation (The largest area)
        sub_audio  = '107' # Based on the surface area calculation (The largest area)
        subs_all   = EMPRISE.childs
        sess       =['visual', 'audio']
        space      = 'fsnative'
        mesh       = 'pial'
        fig_path = '/data/u_jeong_software/EMPRISE/code/Python/Figures_WP3/'
        
        # specify visualization
        d_mu = 0.5
        cols = {'visual': ['dodgerblue', 'darkblue'], \
                'audio':  ['orangered',  'darkred']}
        f3   = {'visual': Fig3_Obj(res_dir, 'visual', space, mesh), \
                'audio':  Fig3_Obj(res_dir, 'audio',  space, mesh)}
        
        # Figure 3A
        if Figure == '3A':
        
            # Figure 3A, Part 1: visual data
            fig = f3['visual'].plot_area_vs_map(subs_all, cols['visual'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3A_ses-visual.png', dpi=150)
            
            # Figure 3A, Part 2: audio data
            fig = f3['audio'].plot_area_vs_map(subs_all, cols['audio'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3A_ses-audio.png', dpi=150)
        
        # Figure 3B
        if Figure == '3B':
        
            # Figure 3B, Part 1: visual data
            fig = f3['visual'].plot_area_vs_mu(sub_visual, d_mu, cols['visual'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3B_ses-visual.png', dpi=150)
            
            # Figure 3B, Part 2: audio data
            fig = f3['audio'].plot_area_vs_mu(sub_audio, d_mu, cols['audio'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3B_ses-audio.png', dpi=150)
        
        # Figure 3C
        if Figure == '3C':
        
            # Figure 3C, Part 1: visual data
            fig = f3['visual'].plot_fwhm_vs_mu(sub_visual, d_mu, cols['visual'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3C_ses-visual.png', dpi=150)
            
            # Figure 3C, Part 2: audio data
            fig = f3['audio'].plot_fwhm_vs_mu(sub_audio, d_mu, cols['audio'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3C_ses-audio.png', dpi=150)
        
        # Figure 3D
        if Figure == '3D':
        
            # Figure 3D, Part 1: visual data
            fig = f3['visual'].plot_topography(sub_visual, cols['visual'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3D_ses-visual.png', dpi=150)
            
            # Figure 3D, Part 2: audio data
            fig = f3['audio'].plot_topography(sub_audio, cols['audio'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3D_ses-audio.png', dpi=150)
        
        # Figure 3E
        if Figure == '3E':
        
            # Figure 3E, Part 1: visual data
            fig = f3['visual'].plot_range_vs_map(sub_visual, cols['visual'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3E_ses-visual.png', dpi=150)
            
            # Figure 3E, Part 2: audio data
            fig = f3['audio'].plot_range_vs_map(sub_audio, cols['audio'])
            fig.savefig(fig_path + 'Fig3/WP3_Figure_3E_ses-audio.png', dpi=150)
        
        # Figure S5
        if Figure == 'S5':
        
            # Figure S5: all subjects, visual & audio
            for sub in subs_all:
                for ses in sess:
                    fig = f3[ses].plot_area_vs_mu(sub, d_mu, cols[ses])
                    filename = fig_path + 'Fig3/WP3_Figure_S5'+'_ses-'+ses+'_sub-'+sub+'.png'
                    fig.savefig(filename, dpi=150)
        
        # Figure S6
        if Figure == 'S6':
        
            # Figure S6: all subjects, visual & audio
            for sub in subs_all:
                for ses in sess:
                    fig = f3[ses].plot_fwhm_vs_mu(sub, d_mu, cols[ses])
                    filename = fig_path + 'Fig3/WP3_Figure_S6'+'_ses-'+ses+'_sub-'+sub+'.png'
                    fig.savefig(filename, dpi=150)
        
        # Figure S7
        if Figure == 'S7':
        
            # Figure S7: all subjects, visual & audio
            for sub in subs_all:
                for ses in sess:
                    fig = f3[ses].plot_topography(sub, cols[ses])
                    filename = fig_path + 'Fig3/WP3_Figure_S7'+'_ses-'+ses+'_sub-'+sub+'.png'
                    fig.savefig(filename, dpi=150)
        
        # Figure S8
        if Figure == 'S8':
        
            # Figure S8: all subjects, visual & audio
            for sub in subs_all:
                for ses in sess:
                    fig = f3[ses].plot_range_vs_map(sub, cols[ses])
                    filename = fig_path + 'Fig3/WP3_Figure_S8'+'_ses-'+ses+'_sub-'+sub+'.png'
                    fig.savefig(filename, dpi=150)

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
        def plot_area_vs_map(self, subs, A_min, d_max):
            
            # load results
            filepath = self.res_dir+'sub-adults'+'_ses-'+self.ses+'_space-'+self.space+'_mesh-'+self.mesh+'_'+self.pref
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
            
            # open figure
            fig = plt.figure(figsize=(12,9))
            axs = fig.subplots(1,2)
            
            # plot area vs. map
            for j, hemi in enumerate(hemis):
                hdr = 'ses-{}, hemi-{}'.format(self.ses, hemi)
                pcm = axs[j].pcolor(A[:,:,j], cmap='hot', \
                                    vmin=0, vmax=(11/10)*np.max(A))
                axs[j].invert_yaxis()
                axs[j].set_xticks(np.arange(len(nmaps))+1/2, labels=nmaps)
                axs[j].set_yticks(np.arange(N)+1/2, labels=subs)
                axs[j].set_xlabel('numerotopic map', fontsize=16)
                axs[j].set_ylabel('subject ID', fontsize=16)
                axs[j].set_title(hdr, fontweight='bold', fontsize=20)
                fig.colorbar(pcm, ax=axs[j])
                for k, nmap in enumerate(nmaps):
                    axs[j].text(k+1/2, 6+1/2, str(np.sum(A[:,k,j]>0)), color='w', fontsize=20, \
                                horizontalalignment='center', verticalalignment='center')
            
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
        # sub_visual = '003'
        # sub_audio  = '009'
        subs_all   = EMPRISE.adults
        sess       =['visual', 'audio']
        space      = 'fsaverage'
        mesh       = 'pial'
        
        # specify visualization
        # d_mu  = 0.5
        A_min = {'visual': 50, 'audio': 25}
        d_max = {'visual': 17, 'audio': 17}
        # cols  = {'visual': ['dodgerblue', 'darkblue'], \
        #          'audio':  ['orangered',  'darkred']}
        f4    = {'visual': Fig4_Obj(res_dir, 'visual', space, mesh), \
                 'audio':  Fig4_Obj(res_dir, 'audio',  space, mesh)}
        
        # Figure 4A
        if Figure == '4A':
        
            # Figure 4A, Part 1: visual data
            fig = f4['visual'].plot_area_vs_map(subs_all, A_min['visual'], d_max['visual'])
            fig.savefig('Figures_WP1/WP1_Figure_4A_ses-visual.png', dpi=150)
            
            # Figure 4A, Part 2: audio data
            fig = f4['audio'].plot_area_vs_map(subs_all, A_min['audio'], d_max['audio'])
            fig.savefig('Figures_WP1/WP1_Figure_4A_ses-audio.png', dpi=150)

# function: Work Package 1, Analysis 1
#-----------------------------------------------------------------------------#
def WP1_Ana1():
    
    # specify analyses
    subs  = EMPRISE.childs
    sess  =['visual', 'audio']
    spaces= EMPRISE.spaces
    mesh  = 'pial'
    cols  =['yellow', 'gold', 'orange', 'red', 'deeppink', 'purple', \
            'lime', 'darkgreen', 'cyan', 'blue', 'grey', 'sienna']
    
    # analyze sessions
    for ses in sess:
        
        # analyze spaces
        for space in spaces:
        
            # load results
            filepath = res_dir+'sub-adults'+'_ses-'+ses+'_space-'+space+'_mesh-'+mesh+'_'
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
            filename = 'Figures_WP3/WP1_Analysis_1'+'_ses-'+ses+'_space-'+space+'_mesh-'+mesh+'.png'
            fig.savefig(filename, dpi=150)

# function: Work Package 1, Analysis 2
#-----------------------------------------------------------------------------#
def WP1_Ana2():
    
    # specify analysis
    subs  = EMPRISE.childs
    sess  =['visual', 'audio']
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
    
    # for all sessions
    for ses in sess:
    
        # load results
        filepath = res_dir+'sub-children'+'_ses-'+ses+'_space-'+space+'_mesh-'+mesh+'_'+'AFNI_'
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
        for sub in subs:
            
            # for both hemispheres
            for j, hemi in enumerate(hemis):
                
                # get supra-threshold vertices/triangles
                verts_sub = verts[sub][hemi][0,0]
                areas_sub = areas[sub][hemi][0,0]
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
    S    = pd.DataFrame(data, columns=cols)
    S.to_csv('Figures_WP3/WP3_Analysis_2_sub-all_ses-both.csv', index=False)
    S.to_excel('Figures_WP3/WP3_Analysis_2_sub-all_ses-both.xlsx', index=False)
    
# test area / debugging section
#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    
    # select Figure
    Figure = '4A'
    
    # create talk Figures
    if Figure == 'Talk':
        Talk_Figs()    
    
    # create Table 0
    if Figure == 'T0':
        WP1_Tab0()
    
    # create Figure 1
    if Figure == '1' or Figure in ['S1','S2']:
        WP1_Fig1(Figure)
    
    # create Figure 2
    if Figure.startswith('2') or Figure in ['S3','S4']:
        WP1_Fig2(Figure)
        
    # create Figure 3
    if Figure.startswith('3') or Figure in ['S5','S6','S7','S8']:
        WP1_Fig3(Figure)
    
    # create Figure 4
    if Figure.startswith('4'):
        WP1_Fig4(Figure)
    
    # run Analysis 1
    if Figure == 'A1':
        WP1_Ana1()
    
    # run Analysis 2
    if Figure == 'A2':
        WP1_Ana2()