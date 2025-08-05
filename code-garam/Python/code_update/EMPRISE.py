"""
EMPRISE - EMergence of PRecISE numerosity representations in the human brain

Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
2023-06-26, 15:31: get_onsets
2023-06-26, 16:39: get_confounds
2023-06-26, 18:03: get_mask_nii, get_bold_nii, get_events_tsv, get_confounds_tsv
2023-06-29, 11:21: load_mask, load_data
2023-06-29, 12:18: onsets_trials2blocks
2023-07-03, 09:56: load_data_all, average_signals, correct_onsets
2023-07-13, 10:11: average_signals
2023-08-10, 13:59: global variables
2023-08-21, 15:42: rewriting to OOP
2023-08-24, 16:36: standardize_confounds
2023-09-07, 16:20: standardize_signals
2023-09-12, 19:23: get_bold_gii, load_surf_data, load_surf_data_all
2023-09-14, 12:46: save_vol, save_surf
2023-09-21, 15:11: plot_surf
2023-09-26, 16:19: analyze_numerosity
2023-09-28, 11:18: threshold_maps
2023-09-28, 12:58: visualize_maps
2023-10-05, 19:12: rewriting to OOP
2023-10-05, 19:34: global variables
2023-10-05, 21:24: rewriting for MPI
2023-10-12, 12:02: threshold_maps, visualize_maps
2023-10-16, 10:56: threshold_maps, testing
2023-10-16, 14:41: load_data_all, load_surf_data_all, get_onsets, get_confounds
2023-10-26, 17:56: get_model_dir, get_results_file, load_mask_data, calc_runs_scans
2023-10-26, 21:06: get_mesh_files, get_sulc_files
2023-11-01, 14:50: get_sulc_files
2023-11-01, 17:53: threshold_and_cluster
2023-11-09, 11:30: refactoring
2023-11-17, 10:34: refactoring
2023-11-20, 12:56: get_mesh_files
2023-11-20, 16:02: threshold_and_cluster
2023-11-23, 12:57: analyze_numerosity
2023-11-28, 14:05: create_fsaverage_midthick, get_mesh_files
2023-11-30, 19:31: threshold_AFNI_cluster
2024-01-22, 17:08: run_GLM_analysis
2024-01-29, 11:50: run_GLM_analysis
2024-01-29, 15:49: threshold_SPM
2024-03-11, 11:08: get_onsets
2024-03-11, 15:34: run_GLM_analysis_group
2024-03-11, 16:44: threshold_SPM_group
2024-03-11, 17:37: refactoring
2024-04-04, 10:22: get_onsets, get_confounds
2024-05-14, 15:03: analyze_numerosity
2024-05-21, 10:35: calculate_Rsq
2024-06-25, 15:18: threshold_maps, threshold_and_cluster
2024-06-27, 13:39: threshold_maps, threshold_and_cluster
2024-07-01, 18:11: analyze_numerosity
"""


# import packages
#-----------------------------------------------------------------------------#
import os
import glob
import time
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import surface
from surfplot import Plot
import NumpRF

# determine location
#-----------------------------------------------------------------------------#
at_MPI = os.getcwd().startswith('/data/')

# define directories
#-----------------------------------------------------------------------------#
if at_MPI:
    stud_dir = r'/data/pt_02495/emprise7t/'
    data_dir = stud_dir
    deri_out = r'/data/pt_02495/emprise7t/derivatives/'
else:
    stud_dir = r'C:/Joram/projects/MPI/EMPRISE/'
    data_dir = stud_dir + 'data/'
    deri_out = data_dir + 'derivatives/'
deri_dir = data_dir + 'derivatives/'
tool_dir = os.getcwd() + '/'

# define identifiers
#-----------------------------------------------------------------------------#
sub   = '001'                   # pilot subject
ses   = 'visual'                # pilot session
sess  =['visual', 'audio']#'audio1', 'audio2', 'digits', 'spoken', 'congruent', 'incongruent']
task  = 'harvey'
acq   =['mprageised', 'fMRI1p75TE24TR2100iPAT3FS'] # EMPRISE2-['bfcmprageised', '1p5'] 
runs  =[1,2,3,4,5,6,7,8]
spaces=['fsnative', 'fsaverage']
meshs =['inflated', 'pial', 'white', 'midthickness']
desc  =['brain', 'preproc', 'confounds']

# define subject groups
#-----------------------------------------------------------------------------#
adults = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012']
childs = ['101', '102', '103', '104', '105', '106', '108', '111', '112', '113', '114', '116'] # 107,109,110 excluded for visual
control = ['N001', 'N002', 'N003', 'N004', 'N005', 'N006', 'N007', 'N008', 'N009', 'N010']
dyscal = ['D001', 'D002', 'D003', 'D004', 'D005', 'D006', 'D007', 'D008', 'D009', 'D010']

# specify scanning parameters
#-----------------------------------------------------------------------------#
TR               = 2.1          # fMRI repetition time
mtr              = 41 #43           # microtime resolution (= bins per TR)
mto              = 21 #22           # microtime onset (= reference slice)
n                = 145          # number of scans per run (EMPRISE-145, EMPRISE2-209/185 for audio2)
b                = 4*2*6        # number of blocks per run (epoch*(increase+decrease)*conditions, EMPRISE-4*2*6, EMPRISE2-4*2*10)
num_epochs       = 4            # number of epochs within run
num_scan_disc    = 1            # number of scans to discard before first epoch
scans_per_epoch  = int((n-num_scan_disc)/num_epochs)
blocks_per_epoch = int(b/num_epochs)

# specify thresholding parameters
#-----------------------------------------------------------------------------#
dAIC_thr  = 0                   # AIC diff must be larger than this
dBIC_thr  = 0                   # BIC diff must be larger than this
Rsq_def   = 0.3                 # R-squared must be larger than this
alpha_def = 0.05                # p-value must be smaller than this
mu_thr    =[1, 5]               # numerosity must be inside this range (EMPRISE-[1,5],EMPRISE2-[1,9])
fwhm_thr  =[0, 24]              # tuning width must be inside this range (EMPRISE-[0,24], EMPRISE2-[0,40])
beta_thr  =[0, np.inf]          # scaling parameter must be inside this range
crit_def  = 'Rsqmb'             # default thresholding option (see "threshold_maps")

# specify default covariates
#-----------------------------------------------------------------------------#
covs = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
        'white_matter', 'csf', 'global_signal', \
        'cosine00', 'cosine01', 'cosine02']

# class: subject/session
#-----------------------------------------------------------------------------#
class Session:
    """
    A Session object is initialized by a subject ID and a session ID and then
    allows for multiple operations performed on the data from this session.
    """
    
    # function: initialize subject/session
    #-------------------------------------------------------------------------#
    def __init__(self, subj_id, sess_id):
        """
        Initialize a Session from a Subject
        sess = EMPRISE.Session(subj_id, sess_id)
        
            subj_id - string; subject identifier (e.g. "001")
            sess_id - string; session identifier (e.g. "visual")
            
            sess    - a Session object
            o sub   - the subject ID
            o ses   - the session ID
        """
        
        # store subject ID and session name
        self.sub = subj_id
        self.ses = sess_id

    # function: get "mask.nii" filenames
    #-------------------------------------------------------------------------#
    def get_mask_nii(self, run_no, space):
        """
        Get Filename for Brain Mask NIfTI File
        filename = sess.get_mask_nii(run_no, space)
        
            run_no   - int; run number (e.g. 1)
            space    - string; image space (e.g. "T1w")
            
            filename - string; filename of "mask.nii.gz"
        
        filename = sess.get_mask_nii(run_no, space) returns the filename of the
        gzipped brain mask belonging to session sess, run run_no and in the
        selected image space.
        """
        
        # create filename
        filename = deri_dir + 'fmriprep' + \
                   '/sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                   '_acq-' + acq[1] + '_run-' + str(run_no) + '_space-' + space + '_desc-' + desc[0] + '_mask.nii.gz'
        return filename

    # function: get "bold.nii" filenames
    #-------------------------------------------------------------------------#
    def get_bold_nii(self, run_no, space=''):
        """
        Get Filename for BOLD NIfTI Files
        filename = sess.get_bold_nii(run_no, space)
        
            run_no   - int; run number (e.g. 1)
            space    - string; image space (e.g. "T1w")
            
            filename - string; filename of "bold.nii.gz"
        
        filename = sess.get_bold_nii(run_no, space) returns the filename of the
        gzipped 4D NIfTI belonging to session sess and run run_no. If space is
        non-empty, then the preprocessed images from the selected image space
        will be returned. By default, space is empty.
        """
        
        # create filename
        if not space:               # raw images in native space
            filename = data_dir + 'sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                       '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                       '_acq-' + acq[1] + '_run-' + str(run_no) + '_bold.nii.gz'
        else:                       # preprocessed images in space
            filename = deri_dir + 'fmriprep' + '/sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                       '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                       '_acq-' + acq[1] + '_run-' + str(run_no) + '_space-' + space + '_desc-' + desc[1] + '_bold.nii.gz'
        return filename
    
    # function: get "bold.gii" filenames
    #-------------------------------------------------------------------------#
    def get_bold_gii(self, run_no, hemi='L', space='fsnative'):
        """
        Get Filename for BOLD GIfTI Files
        filename = sess.get_bold_gii(run_no, hemi, space)
        
            run_no   - int; run number (e.g. 1)
            hemi     - string; brain hemisphere (e.g. "L")
            space    - string; image space (e.g. "fsnative")
            
            filename - string; filename of "bold.func.gii"
        
        filename = sess.get_bold_gii(run_no, hemi, space) returns the filename
        of the 4D GIfTI belonging to session sess, run run_no and brain hemi-
        sphere hemi.
        """
        
        # create filename
        filename = deri_dir + 'fmriprep' + \
                   '/sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                   '_acq-' + acq[1] + '_run-' + str(run_no) + '_hemi-' + hemi + '_space-' + space + '_bold.func.gii'
        return filename

    # function: get "events.tsv" filenames
    #-------------------------------------------------------------------------#
    def get_events_tsv(self, run_no):
        """
        Get Filename for Events TSV File
        filename = sess.get_events_tsv(run_no)
        
            run_no   - int; run number (e.g. 1)
            
            filename - string; filename of "events.tsv"
        
        filename = sess.get_events_tsv(run_no) returns the filename of the
        tab-separated events file belonging to session sess and run run_no.
        """
        
        # create filename
        filename = data_dir + 'sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                   '_acq-' + acq[1] + '_run-' + str(run_no) + '_events.tsv'
        return filename

    # function: get "timeseries.tsv" filenames
    #-------------------------------------------------------------------------#
    def get_confounds_tsv(self, run_no):
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
                   '/sub-' + self.sub + '/ses-' + self.ses + '/func' + \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_task-' + task + \
                   '_acq-' + acq[1] + '_run-' + str(run_no) + '_desc-' + desc[2] + '_timeseries.tsv'
        return filename
    
    # function: get mesh files
    #-------------------------------------------------------------------------#
    def get_mesh_files(self, space='fsnative', surface='inflated'):
        """
        Get Filenames for GIfTI Inflated Mesh Files
        mesh_files = sess.get_mesh_files(space, surface)
        
            space      - string; image space (e.g. "fsnative")
            surface    - string; surface image (e.g. "inflated")
            
            mesh_files - dict; filenames of inflated mesh files
            o left     - string; left hemisphere mesh file
            o right    - string; left hemisphere mesh file
        
        mesh_files = sess.get_mesh_files(space, surface) returns filenames for
        mesh files from specified image space and cortical surface to be used
        for surface plotting.
        """
        
        # if native image space
        if space == 'fsnative':
            
            # specify mesh files
            prep_dir  = deri_dir + 'fmriprep'
            mesh_path = prep_dir + '/sub-' + self.sub + '/anat' + \
                                   '/sub-' + self.sub + '*' + '_hemi-'
            mesh_file = mesh_path + 'L' + '_' + surface + '.surf.gii'
            if not glob.glob(mesh_file):
                for ses in sess:
                    mesh_path = prep_dir + '/sub-' + self.sub + '/ses-' + ses + '/anat' + \
                                           '/sub-' + self.sub + '_ses-' + ses + '*' + '_hemi-'
                    mesh_file = mesh_path + 'L' + '_' + surface + '.surf.gii'
                    if glob.glob(mesh_file):
                        break
            if not glob.glob(mesh_file):
                mesh_files = {'left' : 'n/a', \
                              'right': 'n/a'}
            else:
                mesh_files = {'left' : glob.glob(mesh_path+'L'+'_'+surface +'.surf.gii')[0], \
                              'right': glob.glob(mesh_path+'R'+'_'+surface +'.surf.gii')[0]}
        
        # if average image space
        elif space == 'fsaverage':
            
            # specify mesh dictionary
            mesh_dict = {'inflated':     'infl',  \
                         'pial':         'pial',  \
                         'white':        'white', \
                         'midthickness': 'midthick'}
            
            # specify mesh files
            if surface not in mesh_dict.keys():
                mesh_files = {'left' : 'n/a', \
                              'right': 'n/a'}
            else:
                free_dir   = deri_out + 'freesurfer'
                mesh_path  = free_dir + '/fsaverage/' + mesh_dict[surface]
                mesh_files = {'left' : mesh_path + '_left.gii', \
                              'right': mesh_path + '_right.gii'}
        
        # return mesh files
        return mesh_files
    
    # function: get sulci files
    #-------------------------------------------------------------------------#
    def get_sulc_files(self, space='fsnative'):
        """
        Get Filenames for FreeSurfer Sulci Files
        sulc_files = sess.get_sulc_files(space)
        
            space      - string; image space (e.g. "fsnative")
            
            sulc_files - dict; filenames of FreeSurfer sulci files
            o left     - string; left hemisphere sulci file
            o right    - string; left hemisphere sulci file
        
        sulc_files = sess.get_sulc_files(space) returns filenames for FreeSurfer
        sulci files from specified image space to be used for surface plotting.
        """
        
        # if native image space
        if space == 'fsnative':
            
            # specify sulci files
            free_dir   = deri_dir + 'freesurfer'
            sulc_path  = free_dir + '/sub-' + self.sub + '/surf'
            sulc_files = {'left' : sulc_path + '/lh.curv', \
                          'right': sulc_path + '/rh.curv'}
        
        # if average image space
        elif space == 'fsaverage':
            
            # specify mesh files
            free_dir   = deri_out + 'freesurfer'
            sulc_path  = free_dir + '/fsaverage/curv'
            sulc_files = {'left' : sulc_path + '_left.gii', \
                          'right': sulc_path + '_right.gii'}
        
        # return mesh files
        return sulc_files
    
    # function: load brain mask
    #-------------------------------------------------------------------------#
    def load_mask(self, run_no, space=''):
        """
        Load Brain Mask NIfTI File
        M = sess.load_mask(run_no, space)
        
            run_no - int; run number (e.g. 1)
            space  - string; image space (e.g. "T1w")
            
            M      - 1 x V vector; values of the mask image
        """
        
        # load image file
        filename = self.get_mask_nii(run_no, space)
        mask_nii = nib.load(filename)
        
        # extract mask image
        M = mask_nii.get_fdata()
        M = M.reshape((np.prod(M.shape),), order='C')
        return M
    
    # function: load fMRI data
    #-------------------------------------------------------------------------#
    def load_data(self, run_no, space=''):
        """
        Load Functional MRI NIfTI Files
        Y = sess.load_data(run_no, space)
            
            run_no - int; run number (e.g. 1)
            space  - string; image space (e.g. "T1w")
            
            Y      - n x V matrix; scan-by-voxel fMRI data
        """
        
        # load image file
        filename = self.get_bold_nii(run_no, space)
        bold_nii = nib.load(filename)

        # Keep only the first 209 time points (time is the last dimension)
        #Y = Y[:, :, :, :n]
        
        # extract fMRI data
        Y = bold_nii.get_fdata()
        Y = Y.reshape((np.prod(Y.shape[0:-1]), Y.shape[-1]), order='C')
        Y = Y.T
        return Y

    # function: load fMRI data (all runs)
    #-------------------------------------------------------------------------#
    def load_data_all(self, space=''):
        """
        Load Functional MRI NIfTI Files from All Runs
        Y = sess.load_data_all(space)
            
            space - string; image space (e.g. "T1w")
            
            Y     - n x V x r array; scan-by-voxel-by-run fMRI data
        """
        
        # prepare 3D array
        for j, run in enumerate(runs):
            filename = self.get_bold_nii(run, space)
            if os.path.isfile(filename):
                Y = self.load_data(run, space)
                break
        Y = np.zeros((Y.shape[0], Y.shape[1], len(runs)))
        
        # load fMRI data
        for j, run in enumerate(runs):
            filename = self.get_bold_nii(run, space)
            if os.path.isfile(filename):
                Y[:,:,j] = self.load_data(run, space)
        
        # select available runs
        Y = Y[:,:,np.any(Y, axis=(0,1))]
        return Y
    
    # function: load surface fMRI data
    #-------------------------------------------------------------------------#
    def load_surf_data(self, run_no, hemi='L', space='fsnative'):
        """
        Load Functional MRI GIfTI Files
        Y = sess.load_surf_data(run_no, hemi, space)
            
            run_no - int; run number (e.g. 1)
            hemi   - string; brain hemisphere (e.g. "L")
            space  - string; image space (e.g. "fsnative")
            
            Y      - n x V matrix; scan-by-vertex fMRI data
        """
        
        # load image file
        filename = self.get_bold_gii(run_no, hemi, space)
        bold_gii = nib.load(filename)
        
        # extract fMRI data
        Y = np.array([y.data for y in bold_gii.darrays])
        Y = Y[:n,:]  # get only the first n images
        return Y
    
    # function: load surface fMRI data (all runs)
    #-------------------------------------------------------------------------#
    def load_surf_data_all(self, hemi='L', space='fsnative'):
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
            filename = self.get_bold_gii(run, hemi, space)
            if os.path.isfile(filename):
                Y = self.load_surf_data(run, hemi, space)
                valid_runs.append(run)
                r_scan.append(Y.shape[0])
        
        # Determine minimum scan number
        min_scans = min(r_scan)
        
        # Filter out runs with insufficient scans
        if min_scans <= 109:
            filtered_runs = [run for run, scans in zip(valid_runs, r_scan) if scans > 109]
            r_scan = [scans for scans in r_scan if scans > 109]
            valid_runs = filtered_runs
            min_scans = min(r_scan) if r_scan else 0
        
        # Prepare output array
        if not r_scan:
            return None
        
        Y = np.zeros((min_scans, Y.shape[1], len(r_scan)))
        
        # Load data
        for j, run in enumerate(valid_runs):
            Y[:,:,j] = self.load_surf_data(run, hemi, space)[:min_scans,:]
        
        # Remove empty runs
        Y = Y[:,:,np.any(Y, axis=(0,1))]
    
        
        return Y
    
    # function: get onsets and durations
    #-------------------------------------------------------------------------#
    def get_onsets(self, filenames=None):
        """
        Get Onsets and Durations for Single Subject and Session, all Runs
        ons, dur, stim = sess.get_onsets(filenames)
        
            filenames - list of strings; "events.tsv" filenames
        
            ons       - list of arrays of floats; t x 1 vectors of onsets [s]
            dur       - list of arrays of floats; t x 1 vectors of durations [s]
            stim      - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
            
        ons, dur, stim = sess.get_onsets(filenames) loads the "events.tsv" file
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
        if self.ses == 'visual':
            stimuli = {'1_dot': 1, '2_dot': 2, '3_dot': 3, '4_dot': 4, '5_dot': 5, \
                       '6_dot': 6, '7_dot': 7, '8_dot': 8, '9_dot': 9, '20_dot': 20}
        elif self.ses == 'digits':
            stimuli = {'1_digit': 1, '2_digit': 2, '3_digit': 3, '4_digit': 4, '5_digit': 5, \
                       '6_digit': 6, '7_digit': 7, '8_digit': 8, '9_digit': 9, '20_digit': 20}
        elif self.ses in ['audio1','audio2','spoken']:
            stimuli = {'1_audio': 1, '2_audio': 2, '3_audio': 3, '4_audio': 4, '5_audio': 5, \
                       '6_audio': 6, '7_audio': 7, '8_audio': 8, '9_audio': 9, '20_audio': 20}
        elif self.ses == 'congruent' or self.ses == 'incongruent':
            stimuli = {              '2_mixed': 2, '3_mixed': 3, '4_mixed': 4, '5_mixed': 5, '20_mixed': 20}
        
        # for all runs
        for j, run in enumerate(runs):
            
            # extract filename
            if filenames is None:
                filename = self.get_events_tsv(run)
            else:
                filename = filenames[j]
            
            # if onset file exists
            if os.path.isfile(filename):
                validity_check = self.get_confounds_tsv(run)
                if os.path.isfile(validity_check):
                    print('run ', run, ' is included to analysis.')
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

    # function: get confound variables
    #-------------------------------------------------------------------------#
    def get_confounds(self, labels, filenames=None):
        """
        Get Confound Variables for Single Subject and Session, all Runs
        X_c = sess.get_confounds(labels, filenames)
        
            labels    - list of strings; confound file header entries
            filenames - list of strings; "timeseries.tsv" filenames
            
            X_c       - n x c x r array; confound variables
                       (n = scans, c = variables, r = runs)
        
        X_c = sess.get_confounds(filenames) loads the "timeseries.tsv" file
        belonging to session sess and returns a scan-by-variable-by-run array
        of those confound variables indexed by the list labels. The function
        applies no preprocessing to the confounds.
        """
        
        # prepare confound variables as zero matrix
        c = len(labels)
        
        # Track valid runs and their scan numbers
        valid_runs = []
        run_scans = []
        
        # First pass: determine minimum scan number
        for j, run in enumerate(runs):
            if filenames is None:
                filename = self.get_confounds_tsv(run)
            else:
                filename = filenames[j]
            
            if os.path.isfile(filename):
                confounds = pd.read_csv(filename, sep='\t')
                valid_runs.append(run)
                run_scans.append(len(confounds))
        
        # Determine minimum scan number
        min_scans = min(run_scans)
        
        # Filter out runs with insufficient scans
        if min_scans <= 109:
            filtered_runs = [run for run, scans in zip(valid_runs, run_scans) if scans > 109]
            run_scans = [scans for scans in run_scans if scans > 109]
            valid_runs = filtered_runs
            min_scans = min(run_scans) if run_scans else 0
        
        # Prepare output array
        if not run_scans:
            return None
        
        # Initialize confound matrix with minimum scan number
        X_c = np.zeros((min_scans, c, len(valid_runs)))
        
        # Second pass: load confounds
        for j, run in enumerate(valid_runs):
            if filenames is None:
                filename = self.get_confounds_tsv(run)
            else:
                filename = filenames[j]
            
            confounds = pd.read_csv(filename, sep='\t')
            for k, label in enumerate(labels):
                X_c[:,k,j] = np.array(confounds[label][:min_scans])
        
        # select available runs
        X_c = X_c[:,:,np.any(X_c, axis=(0,1))]
        
        return X_c
    
# class: model/space
#-----------------------------------------------------------------------------#
class Model(Session):
    """
    A Model object is initialized by subject/session/space IDs and model name
    and allows for multiple operations related to numerosity estimation.
    """
    
    # function: initialize model
    #-------------------------------------------------------------------------#
    def __init__(self, subj_id, sess_id, mod_name, space_id='fsnative'):
        """
        Initialize a Model applied to a Session
        mod = EMPRISE.Model(subj_id, sess_id, mod_name, space_id)
        
            subj_id  - string; subject identifier (e.g. "001")
            sess_id  - string; session identifier (e.g. "visual")
            mod_name - string; name for the model (e.g. "NumAna")
            space_id - string; space identifier (e.g. "fsnative")
            
            mod      - a Session object
            o sub    - the subject ID
            o ses    - the session ID
            o model  - the model name
            o space  - the space ID
        """
        
        # store subject/session/space IDs and model name
        super().__init__(subj_id, sess_id)  # inherit parent class
        self.model = mod_name               # configure child object
        self.space = space_id
    
    # function: model directory
    #-------------------------------------------------------------------------#
    def get_model_dir(self):
        """
        Get Folder Name for Model
        mod_dir = mod.get_model_dir()
        
            mod_dir - string; directory where the model is saved
        """
        
        # create folder name
        nprf_dir = '/data/pt_02495/emprise7t/derivatives/' + 'numprf'
        mod_dir  = nprf_dir + '/sub-' + self.sub + '/ses-' + self.ses + '/model-' + self.model
        return mod_dir
    
    # function: results file
    #-------------------------------------------------------------------------#
    def get_results_file(self, hemi='L', fold='all'):
        """
        Get Results Filename for Model
        res_file = mod.get_results_file(hemi)
        
            hemi     - string; brain hemisphere (e.g. "L")
            fold     - string; data subset used ("all", "odd" or "even" runs)
        
            res_file - string; results file into which the model is written
        """
        
        # create filename
        mod_dir  = self.get_model_dir()
        filepath = mod_dir  + '/sub-' + self.sub + '_ses-' + self.ses + '_model-' + self.model + \
                             '_hemi-' + hemi + '_space-' + self.space + '_'
        if fold in ['odd', 'even']:
            filepath = filepath + 'runs-' + fold + '_'
        res_file = filepath + 'numprf.mat'
        return res_file
    
    # function: calculate runs/scans
    #-------------------------------------------------------------------------#
    def calc_runs_scans(self, fold='all'):
        """
        Calculate Number of Runs and Scans
        r0, n0 = mod.calc_runs_scans(fold)
        
            fold - string; data subset used ("all", "odd", "even" runs or "cv")
        
            r0   - int; number of runs analyzed, depending on averaging across runs
            n0   - int; number of scans per run, depending on averaging across epochs
        """
        
        # load results file
        res_file = self.get_results_file('L')
        NpRF     = sp.io.loadmat(res_file)
        
        # count number of runs
        r0  = 0
        for run in runs:
            filename = self.get_confounds_tsv(run)
            if os.path.isfile(filename):
                if (fold == 'all') or (fold == 'cv') or (fold == 'odd'  and run % 2 == 1) or (fold == 'even' and run % 2 == 0):
                    r0 = r0 + 1
        # Explanation: This is the number of available runs. Usually, there
        # are 8 runs, but in case of removed data, there can be fewer runs.
        
        # get number of scans
        avg = list(NpRF['settings']['avg'][0,0][0,:])
        # Explanation: This extracts averaging options from the model settings.
        r0  = [r0,1][avg[0]]
        # Explanation: If averaging across runs, there is only 1 (effective) run.
        n0  = [n,scans_per_epoch][avg[1]]
        # Explanation: If averaging across epochs, there are only 36 (effective) scans.
        
        # return runs and scans
        return r0, n0
    
    # function: load in-mask data
    #-------------------------------------------------------------------------#
    def load_mask_data(self, hemi='L'):
        """
        Load Functional MRI GIfTI Files and Mask
        Y = sess.load_mask_data(hemi, space)
            
            hemi  - string; brain hemisphere (e.g. "L")
            
            Y     - n x v x r array; scan-by-vertex-by-run fMRI data
        """
        
        # load and mask data
        Y = self.load_surf_data_all(hemi, self.space)
        M = np.all(Y, axis=(0,2))
        Y = Y[:,M,:]
        
        # return data and mask
        return Y, M
    
    # function: analyze numerosities
    #-------------------------------------------------------------------------#
    def analyze_numerosity(self, avg=[True, False], corr='iid', order=1, ver='V2', sh=False):
        """
        Estimate Numerosities and FWHMs for Surface-Based Data
        results = mod.analyze_numerosity(avg, corr, order, ver, sh)
        
            avg     - list of bool; see "NumpRF.estimate_MLE" (default: [True, False])
            corr    - string; see "NumpRF.estimate_MLE" (default: "iid")
            order   - int; see "NumpRF.estimate_MLE" (default: 1)
            ver     - string; version identifier (default: "V2")
            sh      - bool; split-half estimation (default: False)
            
            results - dict of dicts; results filenames
            o L     - results for left hemisphere
            o R     - results for right hemisphere
            
        results = mod.analyze_numerosity(avg, corr, order, ver, sh) loads the
        surface-based pre-processed data belonging to model mod, estimates
        tuning parameters using settings avg, corr, order, ver, sh and saves
        results into a single-subject results directory.
        
        The input parameter "sh" (default: False) specifies whether parameters
        are estimated in a split-half sense (if True: separately for odd and
        even runs) or across all runs (if False: across all available runs).
        
        The input parameter "ver" (default: "V2") controls which version of
        the routine is used (for details, see "NumpRF.estimate_MLE"):
            V0:       mu_grid   = [3, 1]
                      fwhm_grid = [10.1, 5] (see "NumpRF.estimate_MLE_rgs")
            V1:       mu_grid   = {0.05,...,6, 10,20,...,640,1280} (128)
                      fwhm_grid = {0.3,...,18, 24,48,96,192} (64)
            V2:       mu_grid   = {0.8,...,9.2, 20} (130) 
                      sig_grid  = {0.05,...,4} (60)
            V2-lin:   mu_grid   = {0.8,...,5.2, 20} (90)
                      sig_grid  = {0.05,...,3} (60)
        
        Note: "sig_grid" is calculated into FWHM values, if ver is "V2", and
        into linear sigma values, if ver is "V2-lin" (see "NumpRF.estimate_MLE").
        
        Note: "analyze_numerosity" uses the results dictionary keys "L" and "R"
        which are identical to the hemisphere labels used by fMRIprep.
        """
        
        # part 1: load subject data
        #---------------------------------------------------------------------#
        print('\n\n-> Subject "{}", Session "{}":'.format(self.sub, self.ses))
        mod_dir = self.get_model_dir()
        if not os.path.isdir(mod_dir): os.makedirs(mod_dir)
        
        # load onsets
        print('   - Loading onsets ... ', end='')
        ons, dur, stim = self.get_onsets()
        ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
        print('successful!')
        
        # load confounds
        print('   - Loading confounds ... ', end='')
        X_c = self.get_confounds(covs)
        X_c = standardize_confounds(X_c)
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
            mu_grid   = np.concatenate((np.arange(0.80, 5.25, 0.05), \
                                        np.array([20]))) # EMPRISE2 change the values
            sig_grid  = np.arange(0.05, 3.05, 0.05) # EMPRISE2 change the values
        else:
            err_msg = 'Unknown version ID: "{}". Version must be "V0" or "V1" or "V2"/"V2-lin".'
            raise ValueError(err_msg.format(ver))
        
        # specify folds
        i = -1                              # slice index (3rd dim)
        if not sh: folds = {'all': []}      # all runs vs. split-half
        else:      folds = {'odd': [], 'even': []}
        for run in runs:                    # for all possible runs
            filename = self.get_confounds_tsv(run)
            if os.path.isfile(filename):    # if data from this run exist
                i = i + 1                   # increase slice index
                if not sh:
                    folds['all'].append(i)  # add slice to all runs
                else:
                    if run % 2 == 1:        # add slice to odd runs
                        folds['odd'].append(i)
                    else:                   # add slice to even runs
                        folds['even'].append(i)
        
        # part 2: analyze both hemispheres
        #---------------------------------------------------------------------#
        hemis   = ['L', 'R']
        results = {}
        for hemi in hemis:
            
            # load data
            print('\n-> Hemisphere "{}", Space "{}":'.format(hemi, self.space))
            print('   - Loading fMRI data ... ', end='')
            Y, M = self.load_mask_data(hemi)
            Y    = standardize_signals(Y)
            V    = M.size
            print('successful!')
            
            # analyze all folds
            results[hemi] = {}
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
                          format(fold, self.model, sett))
                    print('   - Saving results file ... ', end='')
                    filepath = mod_dir  + '/sub-' + self.sub + '_ses-' + self.ses + \
                                          '_model-' + self.model + '_hemi-' + hemi + '_space-' + self.space + '_'
                    if sh: filepath = filepath + 'runs-' + fold + '_'
                    results[hemi][fold] = filepath + 'numprf.mat'
                    res_dict = {'mod_dir': mod_dir, 'settings': {'avg': avg, 'corr': corr, 'order': order}, \
                                'mu_est':  mu_est,  'fwhm_est': fwhm_est, 'beta_est':  beta_est, \
                                'MLL_est': MLL_est, 'MLL_null': MLL_null, 'MLL_const': MLL_const, \
                                'k_est':   k_est,   'k_null':   k_null,   'k_const':   k_const, \
                                'corr_est':corr_est,'version':  ver,      'time':      difference}
                    sp.io.savemat(results[hemi][fold], res_dict)
                    print('successful!')
                    del sett, res_dict
                    
                    # save results (surface images)
                    para_est = {'mu': mu_est, 'fwhm': fwhm_est, 'beta': beta_est}
                    for name in para_est.keys():
                        print('   - Saving "{}" image ... '.format(name), end='')
                        para_map    = np.zeros(V, dtype=np.float32)
                        para_map[M] = para_est[name]
                        surface     = nib.load(self.get_bold_gii(1,hemi,self.space))
                        filename    = filepath + name + '.surf.gii'
                        para_img    = save_surf(para_map, surface, filename)
                        print('successful!')
                    del para_est, para_map, surface, filename, para_img
        
        # return results filename
        return results
    
    # function: calculate R-squared maps
    #-------------------------------------------------------------------------#
    def calculate_Rsq(self, folds=['all', 'odd', 'even', 'cv']):
        """
        Calculate R-Squared Maps for Numerosity Model
        maps = mod.calculate_Rsq(folds)
        
            folds - list of strings; runs for which to calculate
            
            maps  - dict of dicts; calculated R-squared maps
            o all  - dict of strings; all runs
            o odd  - dict of strings; odd runs
            o even - dict of strings; even runs
            o cv   - dict of strings; cross-validated R-squared
              o left  - R-squared map for left hemisphere
              o right - R-squared map for right hemisphere
        
        
        maps = mod.calculate_Rsq(folds) loads results from numerosity analysis
        and calculates R-squared maps for all runs in folds.
        
        Note: "calculate_Rsq" uses the results dictionary keys "left" and "right"
        which are identical to the hemisphere labels used by surfplot.
        """
        
        # part 1: prepare calculations
        #---------------------------------------------------------------------#
        print('\n\n-> Subject "{}", Session "{}", Model "{}":'.format(self.sub, self.ses, self.model))
        mod_dir = self.get_model_dir()
        
        # specify slices
        i = -1                              # slice index (3rd dim)
        slices = {'all': [], 'odd': [], 'even': []}
        for run in runs:                    # for all possible runs
            filename = self.get_confounds_tsv(run)
            if os.path.isfile(filename):    # if data from this run exist
                i = i + 1                   # increase slice index
                slices['all'].append(i)
                if run % 2 == 1: slices['odd'].append(i)
                else:            slices['even'].append(i)
        
        # part 2: analyze both hemispheres
        #---------------------------------------------------------------------#
        hemis = {'L': 'left', 'R': 'right'}
        maps  = {}
        for fold in folds: maps[fold] = {}
        
        # for both hemispheres
        for hemi in hemis.keys():
            print('   - {} hemisphere:'.format(hemis[hemi]))
            
            # for all folds
            for fold in folds:
                
                # load analysis results
                filepath = mod_dir + '/sub-' + self.sub + '_ses-' + self.ses + \
                                     '_model-' + self.model + '_hemi-' + hemi + '_space-' + self.space + '_'
                if fold in ['odd', 'even']:
                    filepath = filepath + 'runs-' + fold + '_'
                res_file = filepath + 'numprf.mat'
                mu_map   = filepath + 'mu.surf.gii'
                NpRF     = sp.io.loadmat(res_file)
                surface  = nib.load(mu_map)
                mask     = surface.darrays[0].data != 0
                
                # calculate R-squared (all, odd, even)
                avg   = list(NpRF['settings']['avg'][0,0][0,:])
                MLL1  = np.squeeze(NpRF['MLL_est'])
                MLL00 = np.squeeze(NpRF['MLL_const'])
                r0,n0 = self.calc_runs_scans(fold)
                n1    = r0*n0
                Rsq   = NumpRF.MLL2Rsq(MLL1, MLL00, n1)
                
                # calculate R-squared (cross-validated)
                if fold == 'cv':
                    
                    # load session data
                    print('     - Calculating split-half cross-validated R-squared ... ')
                    Y, M           = self.load_mask_data(hemi)
                    Y              = standardize_signals(Y)
                    X_c            = self.get_confounds(covs)
                    X_c            = standardize_confounds(X_c)
                    ons, dur, stim = self.get_onsets()
                    ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
                    
                    # cycle through CV folds
                    sets   = ['odd', 'even']
                    oosRsq = np.zeros((len(sets),Rsq.size))
                    for i in range(len(sets)):
                        
                        # load parameters from this fold
                        res_file = self.get_results_file(hemi, sets[i])
                        NpRF     = sp.io.loadmat(res_file)
                        mu1      = np.squeeze(NpRF['mu_est'])
                        fwhm1    = np.squeeze(NpRF['fwhm_est'])
                        beta1    = np.squeeze(NpRF['beta_est'])
                        
                        # get data from the other fold
                        xfold  = sets[1-i]
                        Y2     = Y[:,:,slices[xfold]]
                        ons2   = [ons[i]  for i in slices[xfold]]
                        dur2   = [dur[i]  for i in slices[xfold]]
                        stim2  = [stim[i] for i in slices[xfold]]
                        Xc2    = X_c[:,:,slices[xfold]]
                        
                        # obtain fit across folds
                        ds          = NumpRF.DataSet(Y2, ons2, dur2, stim2, TR, Xc2)
                        oosRsq[i,:] = ds.calculate_Rsq(mu1, fwhm1, beta1, avg)
                        
                    # calculate cross-validated R-squared
                    Rsq = np.mean(oosRsq, axis=0)
                    print()
                
                # threshold tuning maps
                print('     - Saving R-squared image for {} runs ... '.format(fold), end='')
                para_map       = np.zeros(mask.size, dtype=np.float32)
                para_map[mask] = Rsq
                if fold in ['all', 'odd', 'even']:
                    filename   = filepath + 'Rsq.surf.gii'
                else:
                    filename   = filepath + 'cvRsq.surf.gii'
                para_img       = save_surf(para_map, surface, filename)
                maps[fold][hemis[hemi]] = filename
                print('successful!')
                del para_map, surface, filename, para_img
                
        # return results filename
        return maps
    
    # function: threshold tuning maps
    #-------------------------------------------------------------------------#
    def threshold_maps(self, crit='Rsqmb', cv=True):
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
        print('\n\n-> Subject "{}", Session "{}", Model "{}":'.format(self.sub, self.ses, self.model))
        mod_dir = self.get_model_dir()
        
        # get runs and scans
        r0, n0 = self.calc_runs_scans()
        n1     = r0*n0          # effective number of observations in model
        p1     = [4,2][int(cv)] # number of explanatory variables used for R^2
        
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
        hemis = {'L': 'left', 'R': 'right'}
        maps  = {'mu': {}, 'fwhm': {}, 'beta': {}, 'Rsq': {}}
        if cv: maps['cvRsq'] = maps.pop('Rsq')
        for hemi in hemis.keys():
            
            # load numerosity map
            print('   - {} hemisphere:'.format(hemis[hemi]))
            filepath = mod_dir + '/sub-' + self.sub + '_ses-' + self.ses + \
                                 '_model-' + self.model + '_hemi-' + hemi + '_space-' + self.space + '_'
            res_file = filepath + 'numprf.mat'
            mu_map   = filepath + 'mu.surf.gii'
            NpRF     = sp.io.loadmat(res_file)
            surface  = nib.load(mu_map)
            mask     = surface.darrays[0].data != 0
            
            # load estimation results
            mu    = np.squeeze(NpRF['mu_est'])
            fwhm  = np.squeeze(NpRF['fwhm_est'])
            beta  = np.squeeze(NpRF['beta_est'])
            
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
                k1    = NpRF['k_est'][0,0]
                k0    = NpRF['k_null'][0,0]
                Rsq   = NumpRF.MLL2Rsq(MLL1, MLL00, n1)
                # See: https://statproofbook.github.io/P/rsq-mll
                dAIC  = (-2*MLL0 + 2*k0) - (-2*MLL1 + 2*k1)
                # See: https://statproofbook.github.io/P/mlr-aic
                dBIC  = (-2*MLL0 + k0*np.log(n1)) - (-2*MLL1 + k1*np.log(n1))
                # See: https://statproofbook.github.io/P/mlr-bic
            
            # compute quantities for thresholding
            print('     - Applying threshold criteria "{}" ... '.format(crit), end='')
            ind_m = np.logical_or(mu<mu_thr[0], mu>mu_thr[1])
            ind_f = np.logical_or(fwhm<fwhm_thr[0], fwhm>fwhm_thr[1])
            ind_b = np.logical_or(beta<beta_thr[0], beta>beta_thr[1])
            
            # apply conditions for exclusion
            ind = mu > np.inf
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
            para_est = {'mu': mu, 'fwhm': fwhm, 'beta': beta, 'Rsq': Rsq}
            if cv: para_est['cvRsq'] = para_est.pop('Rsq')
            for name in para_est.keys():
                print('     - Saving thresholded "{}" image ... '.format(name), end='')
                para_map       = np.zeros(mask.size, dtype=np.float32)
                para_thr       = para_est[name].copy()
                para_thr[ind]  = np.nan
                para_map[mask] = para_thr
                filename       = filepath + name + '_thr-' + crit + '.surf.gii'
                para_img       = save_surf(para_map, surface, filename)
                maps[name][hemis[hemi]] = filename
                print('successful!')
            del para_est, para_map, para_thr, surface, filename, para_img
        
        # return results filename
        return maps
    
    # function: visualize tuning maps
    #-------------------------------------------------------------------------#
    def visualize_maps(self, crit='', img=''):
        """
        Visualize Numerosity, FWHM and Scaling Maps after Thresholding
        figs = mod.visualize_maps(crit, img)
        
            crit - string; criteria used for thresholding maps OR
            img  - string; image filename between "hemi-L/R" and "surf.gii"
            
            figs - dict of figures; visualized tuning maps
            o mu   - figure object; containing estimated numerosity maps
            o fwhm - figure object; containing FWHM tuning widths maps
            o beta - figure object; containing scaling parameter maps
            o Rsq  - figure object; containing variance explained maps
            o img  - figure object; containing maps specified by filename
        """
        
        # specify auxiliary files
        mesh_files = self.get_mesh_files(self.space)
        sulc_files = self.get_sulc_files(self.space)
        
        # threshold tuning maps
        mod_dir = self.get_model_dir()
        maps    = {}
        if crit:
            filepath = mod_dir  + '/sub-' + self.sub + '_ses-' + self.ses + '_model-' + self.model + '_space-' + self.space + '_'
            maps     = self.threshold_maps(crit)
        elif img:
            filepath    = mod_dir  + '/sub-' + self.sub + '_ses-' + self.ses + '_model-' + self.model + '_'
            maps['img'] = {'left' : filepath+'hemi-L_'+img+'.surf.gii',
                           'right': filepath+'hemi-R_'+img+'.surf.gii'}
        
        # plot and save maps
        figs    = {}
        Rsq_thr = Rsq_def
        if ',' in crit:
            Rsq_thr = float(crit.split(',')[1])
        for name in maps.keys():
            
            # prepare surface plot
            if name == 'mu' or name == 'img':
                caxis = mu_thr
                cmap  = 'gist_rainbow'
                clabel= 'estimated numerosity'
            elif name == 'fwhm':
                caxis = fwhm_thr
                cmap  = 'rainbow'
                clabel= 'estimated tuning width'
            elif name == 'beta':
                caxis = [0,4]
                cmap  = 'hot'
                clabel= 'estimated scaling parameter'
            elif name == 'Rsq' or name == 'cvRsq':
                caxis = [Rsq_thr,1]
                cmap  = 'hot'
                clabel= 'variance explained'
            
            # display and save plot
            figs[name] = plot_surf(maps[name], mesh_files, sulc_files, caxis, cmap, clabel)
            if crit:
                figs[name].savefig(filepath+name+'_thr-'+crit+'.png', dpi=150)
            elif img:
                figs[name].savefig(filepath+img+'.png', dpi=150)
        
        # return results filename
        return figs
    
    # function: threshold and cluster
    #-------------------------------------------------------------------------#
    def threshold_and_cluster(self, hemi='L', crit='Rsqmb', mesh='pial', ctype='coords', d=3, k=100, cv=True):
        """
        Threshold and Cluster Vertices from Surface-Based Results
        verts, trias = mod.threshold_and_cluster(hemi, crit, mesh, ctype, d, k)
        
            hemi  - string; brain hemisphere ("L" or "R")
            crit  - string; criteria for thresholding (see "threshold_maps")
            mesh  - string; mesh file ("inflated", "pial", "white" or "midthickness")
            ctype - string; method of clustering ("coords" or "edges")
            d     - float; maximum distance of vertex to cluster
            k     - int; minimum number of vertices in cluster
            cv    - bool; indicating use of cross-validated R-squared
            
            verts - array; v x 9 matrix of vertex properties
            o 1st           column: vertex index
            o 2nd           column: cluster index
            o 3rd, 4th, 5th column: mu, fwhm, beta
            o 6th           column: R-squared
            o 7th, 8th, 9th column: x, y, z
            trias - array; t x 3 matrix of surface triangles
            o 1st, 2nd, 3rd column: vertex indices
        
        verts, trias = mod.threshold_and_cluster(hemi, crit, mesh, ctype, d, k, cv)
        loads estimated tuning parameter maps, thresholds them according to
        some criteria, clusters them according to some clustering settings
        and returns tabular data from all supra-threshold vertices.
        
        Note that, for the input parameter "ctype", only the option "coords"
        is currently implemented.
        """
        
        # specify surface images
        res_file = self.get_results_file(hemi)
        filepath = res_file[:res_file.find('numprf.mat')]
        mu_map   = filepath + 'mu.surf.gii'
        fwhm_map = filepath + 'fwhm.surf.gii'
        beta_map = filepath + 'beta.surf.gii'
        
        # load surface images
        mu   = nib.load(mu_map).darrays[0].data
        fwhm = nib.load(fwhm_map).darrays[0].data
        beta = nib.load(beta_map).darrays[0].data
        mask = mu != 0
        v    = mask.size
        
        # load mesh files
        hemis     = {'L': 'left', 'R': 'right'}
        mesh_file = self.get_mesh_files(self.space, surface=mesh)[hemis[hemi]]
        mesh_gii  = nib.load(mesh_file)
        XYZ       = mesh_gii.darrays[0].data
        trias     = mesh_gii.darrays[1].data
        # XYZ   is a v x 3 array of coordinates.
        # trias is a t x 3 array of triangles.
        # Source: https://nben.net/MRI-Geometry/#surface-geometry-data
        
        # load estimation results
        NpRF  = sp.io.loadmat(res_file)
        MLL1  = np.squeeze(NpRF['MLL_est'])
        MLL0  = np.squeeze(NpRF['MLL_null'])
        MLL00 = np.squeeze(NpRF['MLL_const'])
        k1    = NpRF['k_est'][0,0]
        k0    = NpRF['k_null'][0,0]
        n1    = np.prod(self.calc_runs_scans())
        p1    = [4,2][int(cv)]
        
        # calculate thresholding quantities
        dAIC       = np.nan * np.ones(v, dtype=np.float32)
        dBIC       = np.nan * np.ones(v, dtype=np.float32)
        Rsq        = np.nan * np.ones(v, dtype=np.float32)
        dAIC[mask] = (-2*MLL0 + 2*k0) - (-2*MLL1 + 2*k1)
        dBIC[mask] = (-2*MLL0 + k0*np.log(n1)) - (-2*MLL1 + k1*np.log(n1))
        Rsq[mask]  = NumpRF.MLL2Rsq(MLL1, MLL00, n1)
        ind_m      = np.logical_or(mu<mu_thr[0], mu>mu_thr[1])
        ind_f      = np.logical_or(fwhm<fwhm_thr[0], fwhm>fwhm_thr[1])
        ind_b      = np.logical_or(beta<beta_thr[0], beta>beta_thr[1])
        
        # extract thresholds for R-squared
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
        
        # apply conditions for exclusion
        ind = mu > np.inf
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
        Rsq[ind] = np.nan
        
        # Step 0: preallocate clusters
        print('\n-> Subject "{}", Session "{}", Model "{}",\n   Space "{}", Surface "{}", Hemisphere "{}":'. \
              format(self.sub, self.ses, self.model, self.space, mesh, hemi))
        clst = np.nan * np.ones(v, dtype=np.int32)
        y    = Rsq
        c    = 0
        # Note: Currently, only ctype=="coords" is implemented!
        
        # Step 1: assign clusters
        print('   - Step 1: assign clusters ... ', end='')
        for j in range(v):
            if not np.isnan(y[j]) and y[j] != 0:
                XYZ_j     = XYZ[j,:]
                new_clust = True
                for i in range(1,c+1):
                    dist_clust = np.sqrt( np.sum( (XYZ[clst==i,:] - XYZ_j)**2, axis=1 ) )
                    conn_clust = dist_clust < d
                    if np.any(conn_clust):
                        new_clust = False
                        clst[j]   = i
                        break
                if new_clust:
                    c       = c + 1
                    clst[j] = c
        print('successful!')
        del XYZ_j, dist_clust, conn_clust, new_clust
        
        # Step 2: merge clusters
        print('   - Step 2: merge clusters ... ', end='')
        for i1 in range(1,c+1):
            for i2 in range(i1+1,c+1):
                XYZ_i1 = XYZ[clst==i1,:]
                single_clust = False
                for j in np.where(clst==i2)[0]:
                    dist_clust = np.sqrt( np.sum( (XYZ_i1 - XYZ[j,:])**2, axis=1 ) )
                    conn_clust = dist_clust < d
                    if np.any(conn_clust):
                        single_clust = True
                        break
                if single_clust:
                    clst[clst==i2] = i1
        print('successful!')
        del XYZ_i1, dist_clust, conn_clust, single_clust
        
        # Step 3: remove clusters
        print('   - Step 3: remove clusters ... ', end='')
        for i in range(1,c+1):
            if np.sum(clst==i) < k:
                clst[clst==i] = np.nan
        print('successful!')
        
        # Step 4: relabel clusters
        print('   - Step 4: relabel clusters ... ', end='')
        clst_nums = np.unique(clst)
        for i in range(len(clst_nums)):
            if not np.isnan(clst_nums[i]):
                clst[clst==clst_nums[i]] = i+1
        print('successful!')
        del clst_nums
        
        # generate vertex table
        verts = np.zeros((0,9))
        for j in range(v):
            if not np.isnan(clst[j]):
                verts = np.r_[verts, \
                              np.array([[j, clst[j], mu[j], fwhm[j], beta[j], \
                                         Rsq[j], XYZ[j,0], XYZ[j,1], XYZ[j,2]]])]
        return verts, trias
    
    # function: threshold, AFNI, cluster
    #-------------------------------------------------------------------------#
    def threshold_AFNI_cluster(self, crit='Rsqmb', mesh='pial', cv=True):
        """
        Threshold, then AFNI SurfClust, then Extract Clusters
        verts, trias = mod.threshold_AFNI_cluster(crit, mesh, cv)
        
            crit  - string; criteria for thresholding (see "threshold_maps")
            mesh  - string; mesh file ("inflated", "pial", "white" or "midthickness")
            cv    - bool; indicating whether cross-validated R-squared is used
            
            verts - dict of arrays; vertex properties
            o left  - array; v x 8 matrix of left hemisphere vertices
            o right - array; v x 8 matrix of right hemisphere vertices
            trias - dict of arrays; surface triangles
            o left  - array; t x 3 matrix of left hemisphere triangles
            o right - array; t x 3 matrix of right hemisphere triangles
            verts, trias - see "threshold_and_cluster"
        
        verts, trias = mod.threshold_AFNI_cluster(crit, mesh, cv) loads
        estimated NumpRF model results, (i) thresholds tuning parameter maps
        according to criteria crit (see "threshold_maps"), (ii) uses AFNI to
        perform surface clustering by edge distance and (iii) returns supra-
        threshold vertices and surface triangles.
        """
        
        # Step 1: R-squared map thresholding
        #---------------------------------------------------------------------#
        hemis    = {'L': 'left', 'R': 'right'}
        res_file = self.get_results_file('L')
        filepath = res_file[:res_file.find('numprf.mat')]
        Rsq_str  = ['Rsq','cvRsq'][int(cv)]
        Rsq_thr  = filepath + Rsq_str + '_thr-' + crit + '.surf.gii'
        
        # display message
        print('\n-> Subject "{}", Session "{}", Model "{}",\n   Space "{}", Surface "{}":'. \
              format(self.sub, self.ses, self.model, self.space, mesh))
        
        # threshold maps
        print('   - Step 1: threshold R-squared maps ... ', end='')
        if not os.path.isfile(Rsq_thr):
            maps = self.threshold_maps(crit, cv)
            # dictionary "maps":
            # - keys "mu", "fwhm", "beta", "Rsq"
            #   - sub-keys "left", "right"
            print()
        else:
            print('already done.')
            maps  = {}
            paras = ['mu','fwhm','beta',Rsq_str]
            for para in paras:
                maps[para] = {}
                for hemi in hemis.keys():
                    res_file = self.get_results_file(hemi)
                    filepath = res_file[:res_file.find('numprf.mat')]
                    maps[para][hemis[hemi]] = filepath + para + '_thr-' + crit + '.surf.gii'
        
        # Step 2: AFNI surface clustering
        #---------------------------------------------------------------------#
        cls_sh    = tool_dir[:tool_dir.find('Python/code_Joram/')] + 'Shell/' + 'cluster_surface'
        if self.space == 'fsnative':  cls_sh = cls_sh + '.sh'
        if self.space == 'fsaverage': cls_sh = cls_sh + '_fsa.sh'
        img_str   = 'space-' + self.space + '_' + Rsq_str + '_thr-' + crit
        mesh_file = self.get_mesh_files(self.space, surface=mesh)['left']
        if self.space == 'fsnative':
            anat_pref = mesh_file[mesh_file.find('sub-')+len('sub-000/'):mesh_file.find('hemi-L')]
        if self.space == 'fsaverage':
            anat_pref = mesh_file[mesh_file.find('fsaverage/')+len('fsaverage/'):mesh_file.find('left.gii')]
        Rsq_cls   = filepath + Rsq_str + '_thr-' + crit + '_cls-' + 'SurfClust' + '.surf.gii'
        
        # cluster surface
        print('   - Step 2: surface cluster using AFNI ... ', end='')
        if not os.path.isfile(Rsq_cls):
            print('\n')
            AFNI_cmd = 'AFNI {} {} {} {} {} {}'. \
                        format(cls_sh, self.sub, self.ses, self.model, img_str, anat_pref)
            os.system(AFNI_cmd)
            # import subprocess
            # subprocess.run(AFNI_cmd.split())
            print()
        else:
            print('already done.')
        
        # Step 3: surface cluster extraction
        #---------------------------------------------------------------------#
        mesh_files = self.get_mesh_files(self.space, surface=mesh)
        
        # extract clusters
        print('   - Step 3: extract surface clusters:')
        verts = {}
        trias = {}
        for hemi in hemis.keys():
            
            # display message
            h = hemis[hemi]
            print('     - {} hemisphere ... '.format(h), end='')
            
            # load surface images
            mu   = nib.load(maps['mu'][h]).darrays[0].data
            fwhm = nib.load(maps['fwhm'][h]).darrays[0].data
            beta = nib.load(maps['beta'][h]).darrays[0].data
            Rsq  = nib.load(maps[Rsq_str][h]).darrays[0].data
            
            # load surface mesh
            mesh_gii    = nib.load(mesh_files[h])
            XYZ         = mesh_gii.darrays[0].data
            trias[hemi] = mesh_gii.darrays[1].data
            
            # load cluster indices
            res_file = self.get_results_file(hemi)
            filepath = res_file[:res_file.find('numprf.mat')]
            Rsq_cls  = filepath + Rsq_str + '_thr-' + crit + '_cls-' + 'SurfClust' + '_cls' + '.surf.gii'
            clst     = nib.load(Rsq_cls).darrays[0].data
            
            # generate vertex table
            verts[hemi] = np.zeros((0,9))
            num_clst    = np.max(clst)
            for i in range(1,num_clst+1):
                verts_new   = np.c_[(clst==i).nonzero()[0], clst[clst==i], \
                                    mu[clst==i], fwhm[clst==i], beta[clst==i],
                                    Rsq[clst==i], XYZ[clst==i,:]]
                verts[hemi] = np.r_[verts[hemi], verts_new]
            del verts_new
            print('successful!')
        
        # return vertices and triangles
        print()
        return verts, trias
    
    # function: run GLM analysis
    #-------------------------------------------------------------------------#
    def run_GLM_analysis(self, avg=False, corr='iid', cons=[]):
        """
        Run General Linear Model with Contrast-Based Inference
        results = mod.run_GLM_analysis(avg, corr, cons)
        
            avg     - bool; indicating whether signals are averaged over runs
            corr    - string; method for serial correlations ('iid' or 'ar1')
            cons    - list of dicts; contrasts to be evaluated
            o type  - string; type of contrast ('t' or 'F')
            o c     - array; (t-contrast) vector or (F-contrast) matrix
            
            results - dict of strings; results filenames
            o L     - results for left hemisphere
            o R     - results for right hemisphere
            
        results = mod.run_GLM_analysis(avg, corr, cons) loads the surface-based
        pre-processed data belonging to model mod, estimates a standard
        general linear model using settings avg/corr, evaluates contrasts
        specified by cons and saves results into a single-subject results
        directory.
        """
        
        # part 1: load subject data
        #---------------------------------------------------------------------#
        import PySPMs as PySPM
        print('\n\n-> Subject "{}", Session "{}":'.format(self.sub, self.ses))
        mod_dir = deri_out + 'pyspm' + '/sub-' + self.sub + '/ses-' + self.ses + '/model-' + self.model
        if not os.path.isdir(mod_dir): os.makedirs(mod_dir)
        
        # load onsets of the valid runs
        print('   - Loading onsets ... ', end='')
        ons, dur, stim = self.get_onsets()
        ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
        print('successful!')
        
        # load confounds
        print('   - Loading confounds ... ', end='')
        X_c = self.get_confounds(covs)
        n   = X_c.shape[0]    # number of scans 
        X_c = standardize_confounds(X_c)
        print('successful!')
        
        # part 2: create design matrices
        #---------------------------------------------------------------------#
        r  = X_c.shape[2]      # number of runs
        print('number of effective runs: ', r)
        b  = 6 # 1-5,20        # number of conditions
        c  = X_c.shape[1]      # number of confounds
        Xr = np.zeros((n,b+c+1,r))
        
        # prepare HRF convolution
        names    = ['1', '2', '3', '4', '5', '20']
        settings = {'n': n, 'TR': TR, 'mtr': mtr, 'mto': mto, \
                    'HRF': 'spm_hrf', 'mc': False}
        
        # create run-wise design matrices
        for j in range(r):
            onsets     = [[o for o,s in zip(ons[j],stim[j]) if s==int(name)] for name in names]
            durations  = [[d for d,s in zip(dur[j],stim[j]) if s==int(name)] for name in names]
            X, L       = PySPM.get_des_mat(names, onsets, durations, None, X_c[:,:,j], settings)
            Xr[:,:,j]  = np.c_[X, np.ones((n,1))]
        L.append('const.')
        del X
        
        # part 3: analyze both hemispheres
        #---------------------------------------------------------------------#
        hemis   = ['L', 'R']
        results = {}
        for hemi in hemis:
            
            # load data
            print('\n-> Hemisphere "{}", Space "{}":'.format(hemi, self.space))
            print('   - Loading fMRI data ... ', end='')
            Yr, M = self.load_mask_data(hemi)
            Yr    = standardize_signals(Yr)
            print('successful!')
            
            # part 3a: average measured signals
            #-----------------------------------------------------------------#
            p = Xr.shape[1]                 # number of regressors
            v = Yr.shape[1]                 # number of voxels
            
            # if averaging across runs, regress out confounds first
            if avg:
                for j in range(r):
                    glm       = PySPM.GLM(Yr[:,:,j], Xr[:,b:,j])
                    B_est     = glm.OLS()
                    # subtract confounds from signal, then re-add constant regressor
                    Yr[:,:,j] = glm.Y - glm.X @ B_est + glm.X[:,[-1]] @ B_est[[-1],:]
                del glm
            
            # then, average across runs, but not across epochs
            if avg:
                Y, t = average_signals(Yr, t=None, avg=[True,False])
            
            # part 3b: prepare correlation matrices
            #-----------------------------------------------------------------#
            a     = 0.4                     # AR parameter
            Q_set = [np.eye(n),             # covariance components
                     sp.linalg.toeplitz(np.power(a, np.arange(0,n))) - np.eye(n)]
            q     = len(Q_set)              # number of components
            V_est = {'Q': Q_set}
            
            # invoke identity matrices, if i.i.d. errors
            if corr == 'iid':
                if avg:
                    h   = np.array([1]+[0 for x in range(q-1)])
                    V   = np.eye(n)
                    V_est.update({'h': h, 'V': V})
                else:
                    h   = np.tile(np.array([1]+[0 for x in range(q-1)]), (r,1))
                    Vr  = np.repeat(np.expand_dims(np.eye(n), 2), r, axis=2)
                    V_est.update({'h': h, 'V': Vr})
            
            # perform ReML estimation, if AR(1) process
            elif corr == 'ar1':
                print('\n-> Restricted maximum likelihood estimation ({} rows, {} columns):'. \
                      format(Y.shape[0], Y.shape[1]))
                if avg:
                    X0  = np.c_[Xr[:,:b,0], Xr[:,-1:,0]]
                    V, Eh, Ph, F, Acc, Com = PySPM.spm_reml(Y, X0, Q_set)
                    V   = (n/np.trace(V)) * V
                    V_est.update({'h': h, 'V': V})
                else:
                    Eh  = np.zeros((q,r))
                    Vr  = np.zeros((n,n,r))
                    for j in range(r):
                        print('   Run {}:'.format(j+1))
                        Vr[:,:,j], Eh[:,[j]], Ph, F, Acc, Com = PySPM.spm_reml(Yr[:,:,j], Xr[:,:,j], Q_set)
                        Vr[:,:,j] = (n/np.trace(Vr[:,:,j])) * Vr[:,:,j]
                    V_est.update({'h': Eh.T, 'V': Vr})
                del Eh, Ph, F, Acc, Com
            
            # otherwise, raise error
            else:
                err_msg = 'Unknown correlation method: "{}". Method must be "iid" or "ar1".'
                raise ValueError(err_msg.format(corr))
            
            # part 3c: estimate GLM parameters
            #-----------------------------------------------------------------#
            if corr == 'ar1': print('\n-> Hemisphere "{}", Space "{}":'.format(hemi, self.space))
            
            # if averaging across runs, only keep conditions in design matrix
            if avg:
                r = 1
                X = np.c_[Xr[:,:b,0], Xr[:,-1:,0]]
                # Y: already set during 3a
                # V: already set during 3b
            
            # otherwise, concatenate data, design and correlation matrices
            else:
                Y = np.zeros((r*n,v))
                X = np.zeros((r*n,r*p))
                V = np.zeros((r*n,r*n))
                for j in range(r):
                    Y[j*n:(j+1)*n, :]           = Yr[:,:,j]
                    X[j*n:(j+1)*n, j*p:(j+1)*p] = Xr[:,:,j]
                    V[j*n:(j+1)*n, j*n:(j+1)*n] = Vr[:,:,j]
            
            # specify and estimate GLM
            start_time    = time.time()
            print('   - Estimating GLM parameters ... ', end='')
            glm           = PySPM.GLM(Y, X, V)
            B_est, s2_est = glm.MLE()
            print('successful!')
            
            # prepare contrast-based inference
            q    = len(cons)
            CONs = np.zeros((q,v))
            SPMs = np.zeros((q,v))
            for k, con in enumerate(cons):
                # t-contrast vector
                if con['type'] == 't':
                    c = con['c']
                    if c.shape[0] <   p:
                        c = np.r_[c, np.zeros(p-c.size)]
                    if c.shape[0] < r*p:
                        c = np.tile(c, r)
                    cons[k]['c'] = c
                # F-contrast matrix
                elif con['type'] == 'F':
                    C = con['c']
                    if C.shape[0] <   p:
                        C = np.r_[C, np.zeros((p-C.shape[0],C.shape[1]))]
                    if C.shape[0] < r*p:
                        C = np.tile(C, (r,1))
                    cons[k]['c'] = C
            if any([con['type']=='F' for con in cons]):
                covB = np.linalg.inv(X.T @ glm.P @ X)
            
            # perform contrast-based inference
            print('   - Evaluating contrast maps ... ', end='')
            for k, con in enumerate(cons):
                # t-contrast vector
                if con['type'] == 't':
                    c = con['c']
                    hyp, pval, stats = glm.tcon(c)
                    SPMs[k,:] = stats['tstat']
                    CONs[k,:] = c.T @ B_est
                # F-contrast matrix
                elif con['type'] == 'F':
                    C = con['c']
                    hyp, pval, stats = glm.Fcon(C)  
                    SPMs[k,:] = stats['Fstat']
                    CON       = C.T @ B_est
                    inv_CcC   = np.linalg.inv(C.T @ covB @ C)
                    for j in range(v):
                        CONs[k,j] = CON[:,j].T @ inv_CcC @ CON[:,j]
                # unknown contrast type
                else:
                    err_msg = 'Unknown contrast type: "{}". Type must be "t" or "F".'
                    raise ValueError(err_msg.format(con['type']))
            if cons:
                del hyp, pval, stats
            print('successful!')
            end_time      = time.time()
            difference    = end_time - start_time
            del start_time, end_time
            
            # part 3d: save GLM estimates
            #-----------------------------------------------------------------#
            sett = str(avg)+'_'+str(corr)
            print('\n-> Model "{}", Settings "{}":'.format(self.model, sett))
            filepath = mod_dir + '/hemi-' + hemi + '_space-' + self.space + '_'
            
            # save results (mat-file)
            print('   - Saving results file ... ', end='')
            results[hemi] = filepath + 'PySPM.mat'
            res_dict = {'mod_dir': mod_dir, 'settings': {'avg': avg, 'corr': corr}, \
                        'X': X, 'V': V, 'r': r, 'n': n, 'p': p, 'v': v, \
                        'V_est': V_est, 'cons': cons, \
                        'mask': M, 'labels': L, 'time': difference}
            sp.io.savemat(results[hemi], res_dict)
            print('successful!')
            del sett, res_dict
            
            # save results (parameter estimates)
            p_all    = B_est.shape[0]
            para_map = np.zeros(M.size, dtype=np.float32)
            surface  = nib.load(self.get_bold_gii(1,hemi,self.space))
            for k in range(p_all+1):
                if k < p_all:
                    print('   - Saving "beta_{:04d}" image ... '.format(k+1), end='')
                    para_map[M] = B_est[k,:]
                    filename    = filepath + 'beta_{:04d}'.format(k+1) + '.surf.gii'
                else:
                    print('   - Saving "ResMS" image ... ', end='')
                    para_map[M] = s2_est
                    filename    = filepath + 'ResMS' + '.surf.gii'
                para_img = save_surf(para_map, surface, filename)
                print('successful!')
            
            # save results (contrast maps)
            para_map = np.zeros(M.size, dtype=np.float32)
            surface  = nib.load(self.get_bold_gii(1,hemi,self.space))
            for k, con in enumerate(cons):
                if con['type'] == 't':
                    print('   - Saving "con_{:04d}" image ... '.format(k+1), end='')
                    filename = filepath + 'con_{:04d}'.format(k+1) + '.surf.gii'
                elif con['type'] == 'F':
                    print('   - Saving "ess_{:04d}" image ... '.format(k+1), end='')
                    filename = filepath + 'ess_{:04d}'.format(k+1) + '.surf.gii'
                para_map[M] = CONs[k,:]
                para_img    = save_surf(para_map, surface, filename)
                print('successful!')
                print('   - Saving "spm{}_{:04d}" image ... '.format(con['type'].upper(), k+1), end='')
                para_map[M] = SPMs[k,:]
                filename    = filepath + 'spm{}_{:04d}'.format(con['type'].upper(), k+1) + '.surf.gii'
                para_img    = save_surf(para_map, surface, filename)
                print('successful!')
            del para_map, surface, filename, para_img
        
        # return results filename
        return results
    
    # function: threshold statistical map
    #-------------------------------------------------------------------------#
    def threshold_SPM(self, ind=1, alpha=0.001):
        """
        Threshold Statistical Parametric Map from General Linear Model
        fig = mod.threshold_SPM(ind, alpha)
            
            ind   - int; index of the contrast to be queried (starting from 1)
            alpha - float; significance level of the (t- or F-)contrast
            
            fig   - figure object; containing surface plots of thresholded SPMs
            
        fig = mod.threshold_SPM(ind, alpha) loads statistical parametric
        map from contrast ind belonging to an estimated general linear model
        mod, thresholds it according to the significance level alpha, saves
        the thresholded SPM and displays a corresponding surface plot.
        """
        
        # get model directory
        if hasattr(self, 'ana'):            # group-level analysis
            mod_dir  = deri_out + 'pyspm' + '/sub-' + self.sub + '/ses-' + self.ses + '/model-' + self.ana
        else:                               # single-subject model
            mod_dir  = deri_out + 'pyspm' + '/sub-' + self.sub + '/ses-' + self.ses + '/model-' + self.model
        res_file = mod_dir + '/hemi-' + 'L' + '_space-' + self.space + '_' + 'PySPM.mat'
        PySPM    = sp.io.loadmat(res_file)
        
        # calculate threshold
        con  = PySPM['cons'][0,ind-1]
        con  = {'type': con['type'][0,0][0], 'c': con['c'][0,0]}
        n, p = PySPM['X'].shape
        if con['type'] == 't':
            thr = sp.stats.t.ppf(1-alpha, n-p)
        elif con['type'] == 'F':
            q   = con['c'].shape[1]
            thr = sp.stats.f.ppf(1-alpha, q, n-p)
        thr_str = '{:.3f}'.format(alpha) # '{:1.2e}'.format(alpha)
        
        # analyze hemispheres
        hemis = {'L': 'left', 'R': 'right'}
        maps  = {}
        for hemi in hemis.keys():
            
            # load statistical map
            filepath = mod_dir + '/hemi-' + hemi + '_space-' + self.space + '_'
            res_file = filepath + 'PySPM.mat'
            spm_map  = filepath + 'spm{}_{:04d}'.format(con['type'].upper(), ind) + '.surf.gii'
            image    = nib.load(spm_map)
            spm      = image.darrays[0].data
            
            # threshold statistical map
            spm_thr          = spm
            spm_thr[spm<thr] = np.nan
            
            # save thresholded map
            filename = spm_map[:spm_map.find('.surf.gii')] + '_p-' + thr_str + '.surf.gii'
            spm_img  = save_surf(spm_thr, image, filename)
            maps[hemis[hemi]] = filename
            del spm_thr, spm_img, image, filename
        
        # specify plotting
        caxis    = [0, 4*thr] #[thr, 4*thr]
        cmap     = 'hot'
        clabel   = con['type']+'-value (p < '+thr_str+')'
        tick_pos = [0, thr, 2*thr, 3*thr, 4*thr]
        cbar     = {'n_ticks': len(tick_pos), 'decimals': 2, 'fontsize': 24}

        
        # specify mesh files
        mesh_files = self.get_mesh_files(self.space)
        sulc_files = self.get_sulc_files(self.space)
        
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
        
        # cusotomise the color bar
        cbar_obj = plt.gcf().axes[-1]  # Access the color bar (last axis in the figure)
        cbar_obj.set_xticks(tick_pos)  # Set custom tick positions
        cbar_obj.set_xticklabels([f'{tick:.2f}' for tick in tick_pos])  # Format tick labels to 2 decimal places

        fig.tight_layout()
        return fig
    
# class: subject group
#-----------------------------------------------------------------------------#
class Group(Model):
    """
    A Group object is initialized by group/session/model IDs and a list of
    subject IDs and allows for group-level GLM analyses to be performed.
    """
    
    # function: initialize group
    #-------------------------------------------------------------------------#
    def __init__(self, grp_id, sess_id, mod_name, ana_name, subj_ids):
        """
        Initialize a Group applied to a Model
        grp = EMPRISE.Group(grp_id, sess_id, mod_name, ana_name, subj_ids):
        
            grp_id   - string; group identifier (e.g. "all")
            sess_id  - string; session identifier (e.g. "visual")
            mod_name - string; name of subject-level model (e.g. "False_iid")
            ana_name - string; name of group-level analysis (e.g. "ttest1")
            subj_ids - list of strings; subject identifiers
            
            grp      - a Group object
            o sub    - the group ID
            o ses    - the session ID
            o model  - the model name
            o ana    - the analysis name
            o subs   - the subject IDs
            o space  - the space (= "fsaverage")
        """
        
        # store group analysis parameters   # inherit parent class
        super().__init__(grp_id, sess_id, mod_name, space_id='fsaverage')
        self.ana  = ana_name                # configure child object
        self.subs = subj_ids
        
    # function: run GLM analysis (group)
    #-------------------------------------------------------------------------#
    def run_GLM_analysis_group(self, y, X=None, cons=[]):
        """
        Run Group-Level General Linear Model with Contrast-Based Inference
        results = grp.run_GLM_analysis_group(y, X, cons):
        
            y       - string; filename ending of the subject-level image
            X       - array; NumPy array specifying the GLM's design matrix
            cons    - list of dicts; contrasts to be evaluated
            o type  - string; type of contrast ('t' or 'F')
            o c     - array; (t-contrast) vector or (F-contrast) matrix
            
            results - dict of strings; results filenames
            o L     - results for left hemisphere
            o R     - results for right hemisphere
            
        results = grp.run_GLM_analysis_group(y, X, cons) loads surface-based
        images from subject-level filename y, estimates a standard general
        linear model  with design matrix X, evaluates contrasts specified by
        cons and saves results into a group-level results directory.
        
        This function currently only supports between-subject designs,
        i.e. no repeated-measures analyses in which more than one image per
        subject would enter the model. The group-level analyses that can be
        realized with this function include one-sample t-test; two-sample
        t-test; one-way ANOVA; two-way ANOVA; multiple regression; ANCOVA.
        """
        
        # part 1: prepare group analysis
        #---------------------------------------------------------------------#
        import PySPMs as PySPM
        print('\n\n-> Group "{}", Session "{}", Analysis "{}":'. \
              format(self.sub, self.ses, self.ana))
        pspm_dir = deri_out + 'pyspm'
        mod_dir  = pspm_dir + '/sub-' + self.sub + '/ses-' + self.ses + '/model-' + self.ana
        if not os.path.isdir(mod_dir): os.makedirs(mod_dir)
        
        # design matrix
        print('   - Design matrix ... ', end='')
        if X is None: X = np.ones((len(self.subs),1))
        print('ready.')
        
        # model dimensions
        n = len(self.subs)      # number of subjects
        p = X.shape[1]          # number of predictors
        V = np.eye(n)           # covariance matrix
        N = n
        
        # part 2: analyze group data
        #---------------------------------------------------------------------#
        hemis   = ['L', 'R']
        results = {}
        for hemi in hemis:
            
            # load data
            print('\n-> Model "{}", Hemisphere "{}", Space "{}":'. \
                  format(self.model, hemi, self.space))
            print('   - Loading fMRI data ... ', end='')
            y1 = pspm_dir + '/sub-' + self.subs[0] + '/ses-' + self.ses + '/model-' + self.model + \
                            '/hemi-' + hemi + '_space-' + self.space + '_' + y
            v  = nib.load(y1).darrays[0].data.size        
            Y  = np.zeros((N,v))            # number of voxels
            for i in range(N):
                yi = pspm_dir + '/sub-' + self.subs[i] + '/ses-' + self.ses + '/model-' + self.model + \
                                '/hemi-' + hemi + '_space-' + self.space + '_' + y
                image  = nib.load(yi)
                Y[i,:] = image.darrays[0].data
            print('successful!')
            
            # calculate mask
            M = np.all(Y, axis=0)           # mask image
            Y = Y[:,M]                      # masked data
            v = Y.shape[1]                  # in-mask voxels
            
            # part 2a: estimate GLM parameters
            #-----------------------------------------------------------------#
            print('   - Estimating GLM parameters ... ', end='')
            
            # specify and estimate GLM
            start_time    = time.time()
            glm           = PySPM.GLM(Y, X, V)
            B_est, s2_est = glm.MLE()
            print('successful!')
            
            # prepare contrast-based inference
            q    = len(cons)
            CONs = np.zeros((q,v))
            SPMs = np.zeros((q,v))
            for k, con in enumerate(cons):
                # t-contrast vector
                if con['type'] == 't':
                    c = con['c']
                    if c.shape[0] < p:
                        c = np.r_[c, np.zeros(p-c.size)]
                    cons[k]['c'] = c
                # F-contrast matrix
                elif con['type'] == 'F':
                    C = con['c']
                    if C.shape[0] < p:
                        C = np.r_[C, np.zeros((p-C.shape[0],C.shape[1]))]
                    cons[k]['c'] = C
            if any([con['type']=='F' for con in cons]):
                covB = np.linalg.inv(X.T @ glm.P @ X)
            
            # perform contrast-based inference
            print('   - Evaluating contrast maps ... ', end='')
            for k, con in enumerate(cons):
                # t-contrast vector
                if con['type'] == 't':
                    c = con['c']
                    hyp, pval, stats = glm.tcon(c)
                    SPMs[k,:] = stats['tstat']
                    CONs[k,:] = c.T @ B_est
                # F-contrast matrix
                elif con['type'] == 'F':
                    C = con['c']
                    hyp, pval, stats = glm.Fcon(C)  
                    SPMs[k,:] = stats['Fstat']
                    CON       = C.T @ B_est
                    inv_CcC   = np.linalg.inv(C.T @ covB @ C)
                    for j in range(v):
                        CONs[k,j] = CON[:,j].T @ inv_CcC @ CON[:,j]
                # unknown contrast type
                else:
                    err_msg = 'Unknown contrast type: "{}". Type must be "t" or "F".'
                    raise ValueError(err_msg.format(con['type']))
            if cons:
                del hyp, pval, stats
            print('successful!')
            end_time      = time.time()
            difference    = end_time - start_time
            del start_time, end_time
            
            # part 2b: save GLM estimates
            #-----------------------------------------------------------------#
            print('\n-> Analysis "{}", {} Regressor{}, {} Contrast{}:'. \
                  format(self.ana, p, ['','s'][int(p>1)], q, ['','s'][int(q>1)]))
            filepath = mod_dir + '/hemi-' + hemi + '_space-' + self.space + '_'
            
            # save results (mat-file)
            print('   - Saving results file ... ', end='')
            results[hemi] = filepath + 'PySPM.mat'
            res_dict = {'mod_dir': mod_dir, \
                        'X': X, 'V': V, 'n': n, 'p': p, 'v': v, \
                        'cons': cons, 'mask': M, 'time': difference}
            sp.io.savemat(results[hemi], res_dict)
            print('successful!')
            del res_dict
            
            # save results (parameter estimates)
            para_map = np.zeros(M.size, dtype=np.float32)
            surface  = nib.load(y1)
            for k in range(p+1):
                if k < p:
                    print('   - Saving "beta_{:04d}" image ... '.format(k+1), end='')
                    para_map[M] = B_est[k,:]
                    filename    = filepath + 'beta_{:04d}'.format(k+1) + '.surf.gii'
                else:
                    print('   - Saving "ResMS" image ... ', end='')
                    para_map[M] = s2_est
                    filename    = filepath + 'ResMS' + '.surf.gii'
                para_img = save_surf(para_map, surface, filename)
                print('successful!')
            
            # save results (contrast maps)
            para_map = np.zeros(M.size, dtype=np.float32)
            surface  = nib.load(y1)
            for k, con in enumerate(cons):
                if con['type'] == 't':
                    print('   - Saving "con_{:04d}" image ... '.format(k+1), end='')
                    filename = filepath + 'con_{:04d}'.format(k+1) + '.surf.gii'
                elif con['type'] == 'F':
                    print('   - Saving "ess_{:04d}" image ... '.format(k+1), end='')
                    filename = filepath + 'ess_{:04d}'.format(k+1) + '.surf.gii'
                para_map[M] = CONs[k,:]
                para_img    = save_surf(para_map, surface, filename)
                print('successful!')
                print('   - Saving "spm{}_{:04d}" image ... '.format(con['type'].upper(), k+1), end='')
                para_map[M] = SPMs[k,:]
                filename    = filepath + 'spm{}_{:04d}'.format(con['type'].upper(), k+1) + '.surf.gii'
                para_img    = save_surf(para_map, surface, filename)
                print('successful!')
            del para_map, surface, filename, para_img
        
        # return results filename
        return results
    
# function: average signals
#-----------------------------------------------------------------------------#
def average_signals(Y, t=None, avg=[True, False]):
    """
    Average Signals Measured during EMPRISE Task
    Y, t = average_signals(Y, t, avg)
    
        Y   - n x v x r array; scan-by-voxel-by-run signals
        t   - n x 1 vector; scan-wise fMRI acquisition times
        avg - list of bool; indicating whether signals are averaged (see below)
        
        Y   - n0 x v x r array; if averaged across epochs OR
              n  x v matrix; if averaged across runs OR
              n0 x v matrix; if averaged across runs and epochs (n0 = scans per epoch)
        t   - n0 x 1 vector; if averaged across epochs OR
              n  x 1 vector; identical to input otherwise
    
    Y, t = average_signals(Y, t, avg) averages signals obtained with the 
    EMPRISE experiment across either runs, or epochs within runs, or both.
    
    If the input variable "t" is not specified, it is automatically set to
    the vector [0, 1*TR, 2*TR, ..., (n-2)*TR, (n-1)*TR].
    
    The input variable "avg" controls averaging. If the first entry of avg is
    true, then signals are averaged over runs. If the second entry of avg is
    true, then signals are averaged over epochs within runs. If both are
    true, then signals are first averaged over runs and then epochs. By
    default, only the first entry is true, causing averaging across runs.
    """
    
    # create t, if necessary
    if t is None:
        t = np.arange(0, n*TR, TR)
    
    # average over runs
    if avg[0]:
        
        # if multiple runs
        if len(Y.shape) > 2:
            Y = np.mean(Y, axis=2)
    
    # average over epochs
    if avg[1]:
        
        # remove discard scans
        Y = Y[num_scan_disc:]
        loc_scans_per_epoch = int(Y.shape[0]/num_epochs) # There is a global variable but redfined here for flexibility.
        
        
        # if averaged over runs
        if len(Y.shape) < 3:
            Y_epochs = np.zeros((loc_scans_per_epoch,Y.shape[1],num_epochs))
            for i in range(num_epochs):
                Y_epochs[:,:,i] = Y[(i*loc_scans_per_epoch):((i+1)*loc_scans_per_epoch),:]
            Y = np.mean(Y_epochs, axis=2)
        
        # if not averaged over runs
        else:
            Y_epochs = np.zeros((loc_scans_per_epoch,Y.shape[1],Y.shape[2],num_epochs))
            for i in range(num_epochs):
                Y_epochs[:,:,:,i] = Y[(i*loc_scans_per_epoch):((i+1)*loc_scans_per_epoch),:,:]
            Y = np.mean(Y_epochs, axis=3)
        
        # correct time vector
        t = t[num_scan_disc:]
        t = t[:loc_scans_per_epoch] - num_scan_disc*TR
    
    # return averaged signals
    return Y, t

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

# function: transform onsets and durations
#-----------------------------------------------------------------------------#
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
    
# function: create fsaverage midthickness mesh
#-----------------------------------------------------------------------------#
def create_fsaverage_midthick():
    """
    Calculate Midthickness Coordinates for FSaverage Space
    create_fsaverage_midthick()
    
    This routine creates a midthickness mesh for the fsaverage space by
    averaging surface coordinates from pial and white meshs [1].
    
    [1] https://neurostars.org/t/midthickness-for-fsaverage/16676/2
    """
    
    # load pial and white meshs
    please      = Session('001','visual')
    fsavg_pial  = please.get_mesh_files('fsaverage','pial')
    fsavg_white = please.get_mesh_files('fsaverage','white')
    XYZ_pial    = {}
    XYZ_white   = {}
    for hemi in fsavg_pial.keys():
        XYZ_pial[hemi]  = nib.load(fsavg_pial[hemi]).darrays[0].data
        XYZ_white[hemi] = nib.load(fsavg_white[hemi]).darrays[0].data

    # save midthickness mesh
    fsavg_midthick = please.get_mesh_files('fsaverage','midthickness')
    XYZ_midthick   = {}
    for hemi in fsavg_midthick.keys():
        image = nib.load(fsavg_pial[hemi])
        trias = image.darrays[1].data
        XYZ_midthick[hemi] = (XYZ_pial[hemi] + XYZ_white[hemi])/2
        img_midthick       = nib.gifti.GiftiImage(header=image.header,
                                                  darrays=[nib.gifti.GiftiDataArray(XYZ_midthick[hemi]),\
                                                           nib.gifti.GiftiDataArray(trias)])
        nib.save(img_midthick, fsavg_midthick[hemi])
    
    # return output filenames
    return fsavg_midthick

# function: save single volume image (3D)
#-----------------------------------------------------------------------------#
def save_vol(data, img, fname):
    """
    Save Single Volume Image
    img = save_vol(data, img, fname)
    
        data  - 1 x V vector; data to be written
        img   - Nifti1Image; template image object
        fname - string; filename of resulting image
        
        img   - Nifti1Image; resulting image object
    """
    
    # create and save image
    data_map = data.reshape(img.shape, order='C')
    data_img = nib.Nifti1Image(data_map, img.affine, img.header)
    nib.save(data_img, fname)
    
    # load and return image
    data_img = nib.load(fname)
    return data_img

# function: save single surface image (2D)
#-----------------------------------------------------------------------------#
def save_surf(data, img, fname):
    """
    Save Single Surface Image
    img = save_vol(data, img, fname)
    
        data  - 1 x V vector; data to be written
        img   - GiftiImage; template image object
        fname - string; filename of resulting image
        
        img   - GiftiImage; resulting image object
    """
    
    # create and save image
    data_img = nib.gifti.GiftiImage(header=img.header, \
                                    darrays=[nib.gifti.GiftiDataArray(data)])
    nib.save(data_img, fname)
    
    # load and return image
    data_img = nib.load(fname)
    return data_img

# function: visualize data on surface
#-----------------------------------------------------------------------------#
def plot_surf(surf_files, mesh_files, sulc_files, caxis=[0,1], cmap='viridis', clabel='estimate'):
    """
    Visualize Data on Brain Surface
    fig = plot_surf(surf_files, mesh_files, sulc_files, caxis, cmap, clabel)
    
        surf_files - dict of strings; images to be plotted on surface
        mesh_files - dict of strings; inflated anatomical surface images
        sulc_files - dict of strings; FreeSurfer-processed sulci files
        o left     - images/files for left hemisphere
        o right    - images/files for right hemisphere
        caxis      - list of float; color axis limits
        cmap       - string; color map name
        clabel     - string; color bar label
        
        fig        - figure object; into which the surface images are plotted
    """
    
    # load surface images
    surf_imgs = {}
    sulc_data = {}
    for hemi in surf_files.keys():
        surf_img = surface.load_surf_data(surf_files[hemi])
        surf_img = surf_img.astype(np.float32)
        surf_img[surf_img < caxis[0]] = np.nan
        surf_img[surf_img > caxis[1]] = np.nan
        surf_imgs[hemi] = surf_img
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
    # from xvfbwrapper import Xvfb
    # vdisplay = Xvfb()
    # vdisplay.start()
    fig = plot.build(colorbar=True, cbar_kws={'n_ticks': 5, 'decimals': 1, 'fontsize': 24})
    fig.tight_layout()
    # vdisplay.stop()
    
    # return figure object
    return fig

# test area / debugging section
#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    
    # import packages
    import matplotlib.pyplot as plt
    # enter "%matplotlib qt" in Spyder before
    
    # specify what to test
    what_to_test = 'threshold_SPM'
    
    # test filenames
    if what_to_test == 'filenames':
        ses = Session('001','visual')
        print(ses.get_mask_nii(1,'T1w'))
        print(ses.get_bold_nii(2,'T1w'))
        print(ses.get_bold_gii(3,'L','fsnative'))
        print(ses.get_events_tsv(4))
        print(ses.get_confounds_tsv(5))
        mod = Model('001','visual','True_False_iid_1','fsaverage')
        print(mod.get_model_dir())
        print(mod.get_results_file('L'))
        print(mod.get_mesh_files('fsnative', 'midthickness'))
        print(mod.get_mesh_files('fsaverage', 'inflated'))
        print(mod.get_sulc_files('fsnative'))
        print(mod.get_sulc_files('fsaverage'))
    
    # test "load_mask"
    if what_to_test == 'load_mask':
        sess = Session('001','visual')
        M    = sess.load_mask(1,'T1w')
        print('The images have {} voxels of which {} are inside the brain mask.'. \
              format(M.size, np.sum(M==1)))
    
    # test "load_data"
    if what_to_test == 'load_data':
        sess   = Session('001','visual')
        M      = sess.load_mask(1,'T1w')
        Y      = sess.load_data(1,'T1w')
        Y_mask = Y[:,M==1]
        print("The data are a {} x {} matrix. When masked, it's a {} x {} matrix.". \
              format(Y.shape[0], Y.shape[1], Y_mask.shape[0], Y_mask.shape[1]))

    # test "load_data_all"
    if what_to_test == 'load_data_all':
        sess = Session('001','visual')
        M    = sess.load_mask(1,'T1w')
        Y    = sess.load_data_all('T1w')
        Y    = Y[:,M==1,:]
        print('Masked data from all runs were loaded into a {} x {} x {} array.'. \
              format(Y.shape[0], Y.shape[1], Y.shape[2]))
    
    # test "load_surf_data"
    if what_to_test == 'load_surf_data':
        sess   = Session('001','visual')
        Y      = sess.load_surf_data(1, 'L', 'fsnative')
        Y_mask = Y[:,np.all(Y, axis=0)]
        print("The data are a {} x {} matrix. When masked, it's a {} x {} matrix.". \
              format(Y.shape[0], Y.shape[1], Y_mask.shape[0], Y_mask.shape[1]))
    
    # test "load_surf_data_all"
    if what_to_test == 'load_surf_data_all':
        sess = Session('001','visual')
        Y    = sess.load_surf_data_all('L', 'fsnative')
        Y    = Y[:,np.all(Y, axis=(0,2)),:]
        print('Masked data from all runs were loaded into a {} x {} x {} array.'. \
              format(Y.shape[0], Y.shape[1], Y.shape[2]))
    
    # test "get_onsets"
    if what_to_test == 'get_onsets':
        sess = Session('001','visual')
        ons, dur, stim = sess.get_onsets()
        print(ons[0])
        print(dur[0])
        print(stim[0])
        
    # test "get_confounds"
    if what_to_test == 'get_confounds':
        
        # load confounds
        sess = Session('001','visual')
        X_c  = sess.get_confounds(covs)
        X_c  = standardize_confounds(X_c)
                
        # plot confounds
        plt.rcParams.update({'font.size': 24})
        fig = plt.figure(figsize=(32,18))
        axs = fig.subplots(1,X_c.shape[2])
        fig.suptitle('confound variables')
        for j, ax in enumerate(axs):
            ax.imshow(X_c[:,:,j], aspect='auto')
        fig.show()
        
    # test "calc_runs_scans"
    if what_to_test == 'calc_runs_scans':
        mod    = Model('001','visual','True_False_iid_1','fsnative')
        r0, n0 = mod.calc_runs_scans()
        n1     = r0*n0
        print('{} effective run{} x {} effective scans = {} data points'. \
              format(r0, ['','s'][int(bool(r0-1))], n0, n1))
        
    # test "load_mask_data"
    if what_to_test == 'load_mask_data':
        mod  = Model('001','visual','True_False_iid_1','fsnative')
        Y, M = mod.load_mask_data('L')
        print('Data were loaded from {} scans in {} runs. There are {} in-mask vertices.'. \
              format(Y.shape[0], Y.shape[2], Y.shape[1]))
    
    # test "analyze_numerosity"
    if what_to_test == 'analyze_numerosity':
        
        # analyze numerosity
        mod = Model('001','visual','True_False_iid_1_V2','fsnative')
        mod.analyze_numerosity()
    
    # test "threshold_maps"
    if what_to_test == 'threshold_maps':
        
        # threshold maps
        mod = Model('001','visual','True_False_iid_1','fsnative')
        mod.threshold_maps('AICb')
        mod.threshold_maps('BICb')
        mod.threshold_maps('Rsqb')
    
    # test "visualize_maps"
    if what_to_test == 'visualize_maps':
        
        # visualize maps
        mod = Model('001','visual','True_False_iid_1','fsnative')
        mod.visualize_maps(crit='AICb')
        mod.visualize_maps(img='space-fsnative_mu_thr-Rsq_cls-SurfClust')
    
    # test "threshold_and_cluster"
    if what_to_test == 'threshold_and_cluster':
        
        # threshold and cluster
        mod = Model('001','visual','True_False_iid_1','fsnative')
        verts, trias = mod.threshold_and_cluster('L', 'Rsqmb', 'pial')
        print(verts.shape)
        print(trias.shape)
    
    # test "threshold_AFNI_cluster"
    if what_to_test == 'threshold_AFNI_cluster':
        
        # threshold, AFNI, cluster
        mod = Model('003','visual','True_False_iid_1','fsnative')
        verts, trias = mod.threshold_AFNI_cluster('Rsqmb,0.2', 'pial')
        print(verts['L'].shape, verts['R'].shape, trias['L'].shape, trias['R'].shape)
        mod = Model('009','audio','True_False_iid_1','fsnative')
        verts, trias = mod.threshold_AFNI_cluster('Rsqmb,0.2', 'pial')
        print(verts['L'].shape, verts['R'].shape, trias['L'].shape, trias['R'].shape)
        
    # test "run_GLM_analysis"
    if what_to_test == 'run_GLM_analysis':
        
        # perform GLM analysis
        mod  = Model('001','visual','False_iid','fsnative')
        cons = [{'type': 'F', 'c': np.array([[1/5, 1/5, 1/5, 1/5, 1/5, -1]]).T}, \
                {'type': 't', 'c': np.array([+1/5, +1/5, +1/5, +1/5, +1/5, -1])}, \
                {'type': 't', 'c': np.array([-1/5, -1/5, -1/5, -1/5, -1/5, +1])}]
        mod.run_GLM_analysis(False, 'iid', cons)
        
    # test "threshold_SPM"
    if what_to_test == 'threshold_SPM':
        
        # prepare thresholding
        mod  = Model('001','visual','False_iid','fsnative')
        cons = [1, 2, 3]
        thrs = [0.001, NumpRF.Rsq2pval(Rsq=0.2, n=145, p=4)]
        
        # threshold statistical maps
        mod_dir = deri_out + 'pyspm' + '/sub-' + mod.sub + '/ses-' + mod.ses + '/model-' + mod.model
        for ind in cons:
            for alpha in thrs:
                fig     = mod.threshold_SPM(ind, alpha)
                thr_str = '{:1.2e}'.format(alpha)
                if ind == 1:
                    filename = mod_dir + '/spmF_{:04d}'.format(ind) + '_p-' + thr_str + '.png'
                else:
                    filename = mod_dir + '/spmT_{:04d}'.format(ind) + '_p-' + thr_str + '.png'
                fig.savefig(filename, dpi=150, transparent=True)
        
    # test "average_signals"
    if what_to_test == 'average_signals':
        
        # load data
        sess = Session('001','visual')
        M    = sess.load_mask(1,'T1w')
        print('mask image: 1 x {} vector'.format(M.shape[0]))
        Y    = sess.load_data_all('T1w')
        print('all data: {} x {} x {} array.'.format(Y.shape[0], Y.shape[1], Y.shape[2]))
        Ym   = Y[:,M==1,:]
        del Y
        
        # average data
        print('masked data: {} x {} x {} array.'.format(Ym.shape[0], Ym.shape[1], Ym.shape[2]))
        Y, t0= average_signals(Ym, avg=[False, False])
        print('not averaged: {} x {} x {} array.'.format(Y.shape[0], Y.shape[1], Y.shape[2]))
        Y, t = average_signals(Ym, t0, avg=[True, False])
        print('averaged across runs: {} x {} matrix.'.format(Y.shape[0], Y.shape[1]))
        Y, t = average_signals(Ym, t0, avg=[False, True])
        print('averaged across epochs: {} x {} x {} array.'.format(Y.shape[0], Y.shape[1], Y.shape[2]))
        Y, t = average_signals(Ym, t0, avg=[True, True])
        print('averaged across both: {} x {} matrix.'.format(Y.shape[0], Y.shape[1]))
        
    # test "standardize_signals"
    if what_to_test == 'standardize_signals':
        Y   = np.random.normal(5, 0.1, size=(100,1,1))
        Ys  = standardize_signals(Y.copy(), [True, True])
        fig = plt.figure(figsize=(16,9))
        axs = fig.subplots(2,1)
        axs[0].plot(Y[:,0,0])
        axs[0].set_title('non-standardized', fontsize=24)
        axs[1].plot(Ys[:,0,0])
        axs[1].set_title('standardized', fontsize=24)
        fig.show()
    
    # test "standardize_confounds"
    if what_to_test == 'standardize_confounds':
        X   = np.random.normal(10, 1, size=(100,1,1))
        Xs  = standardize_confounds(X.copy(), [True, True])
        fig = plt.figure(figsize=(16,9))
        axs = fig.subplots(2,1)
        axs[0].plot(X[:,0,0])
        axs[0].set_title('non-standardized', fontsize=24)
        axs[1].plot(Xs[:,0,0])
        axs[1].set_title('standardized', fontsize=24)
        fig.show()
    
    # test "correct_onsets"
    if what_to_test == 'correct_onsets':
        sess = Session('001','visual')
        ons, dur, stim = sess.get_onsets()
        ons, dur, stim = (ons, dur, stim, 'closed')
        ons, dur, stim = correct_onsets(ons[0], dur[0], stim[0])
        print(ons)
        print(dur)
        print(stim)
    
    # test "onsets_trials2blocks"
    if what_to_test == 'onsets_trials2blocks':
        sess = Session('001','visual')
        ons, dur, stim = sess.get_onsets()
        print('trials: {} onsets, {} durations, {} stimuli.'. \
              format(len(ons[0]), len(dur[0]), len(stim[0])))
        ons, dur, stim = (ons, dur, stim, 'closed')
        print('blocks: {} onsets, {} durations, {} stimuli.'. \
              format(len(ons[0]), len(dur[0]), len(stim[0])))
    
    # test "create_fsaverage_midthick"
    if what_to_test == 'create_fsaverage_midthick':
        print(create_fsaverage_midthick())
    
    # test "save_vol"
    if what_to_test == 'save_vol':
        
        # load data and template
        sess = Session('001','visual')
        Y    = sess.load_data(1,'T1w')
        y    = Y[0,:]
        temp = nib.load(sess.get_bold_nii(1,'T1w'))
        temp = temp.slicer[:,:,:,0]
        
        # save and display image
        filename = sess.get_bold_nii(1,'T1w')
        file,ext = os.path.splitext(filename)
        file,ext = os.path.splitext(filename)
        filename = file+'_scan-1.nii.gz'
        y_img    = save_vol(y, temp, filename)
        print(y_img)
    
    # test "save_surf"
    if what_to_test == 'save_surf':
        
        # load data and template
        sess = Session('001','visual')
        Y    = sess.load_surf_data(1,'L','fsnative')
        y    = Y[0,:]
        temp = nib.load(sess.get_bold_gii(1,'L','fsnative'))
        
        # save and display image
        filename = sess.get_bold_gii(1,'L','fsnative')
        file,ext = os.path.splitext(filename)
        file,ext = os.path.splitext(filename)
        filename = file+'_scan-1.surf.gii'
        y_img    = save_surf(y, temp, filename)
        print(y_img)
        
    # test "plot_surf"
    if what_to_test == 'plot_surf':
        
        # specify images
        mod = Model('001', 'visual', 'True_False_iid_1', 'fsnative')
        res_file_L = mod.get_results_file('L')
        res_file_R = mod.get_results_file('R')
        filepath_L = res_file_L[:res_file_L.find('numprf.mat')]
        filepath_R = res_file_R[:res_file_R.find('numprf.mat')]
        surf_files = {'left':  filepath_L+'Rsq_thr-Rsqmb,0.2.surf.gii',
                      'right': filepath_R+'Rsq_thr-Rsqmb,0.2.surf.gii'}
        mesh_files = mod.get_mesh_files('fsnative')
        sulc_files = mod.get_sulc_files('fsnative')
        fig = plot_surf(surf_files, mesh_files, sulc_files, caxis=[0.2,1], cmap='hot', clabel='R²')