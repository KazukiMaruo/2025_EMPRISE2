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
"""


# import packages
#-----------------------------------------------------------------------------#
import os
import glob
import time
import re
import numpy as np
import scipy as sp
import pandas as pd
import nibabel as nib
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
    deri_out = r'/data/u_jeong_software/derivatives/'
    
    """ 
    else:
    stud_dir = r'C:/Joram/projects/MPI/EMPRISE/'
    data_dir = stud_dir + 'data/'
    deri_out = data_dir + 'derivatives/' 
    """
deri_dir = data_dir + 'derivatives/'
tool_dir = os.getcwd() + '/'   # /data/u_jeong_software/

# define identifiers
#-----------------------------------------------------------------------------#
sub   = '001'                   # pilot subject
ses   = 'visual'                # pilot session
sess  =['visual', 'audio', 'digits', 'spoken']
task  = 'harvey'
acq   =['mprageised', 'fMRI1p75TE24TR2100iPAT3FS']
runs  =[1,2,3,4,5,6,7,8]
spaces=['fsnative', 'fsaverage']
meshs =['inflated', 'pial', 'white', 'midthickness']
desc  =['brain', 'preproc', 'confounds']

# define subject groups
#-----------------------------------------------------------------------------#
adults = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012']
childs = ['101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '116']

# specify scanning parameters
#-----------------------------------------------------------------------------#
TR               = 2.1          # fMRI repetition time
mtr              = 41           # microtime resolution (= bins per TR)
mto              = 21           # microtime onset (= reference slice)
n                = 145          # number of scans per run
b                = 4*2*6        # number of blocks per run
num_epochs       = 4            # number of epochs within run
num_scan_disc    = 1            # number of scans to discard before first epoch
scans_per_epoch  = int((n-num_scan_disc)/num_epochs)
blocks_per_epoch = int(b/num_epochs)

# specify thresholding parameters
#-----------------------------------------------------------------------------#
dAIC_thr = 0                    # AIC diff must be larger than this
dBIC_thr = 0                    # BIC diff must be larger than this
Rsq_def  = 0.3                  # R-squared must be larger than this
mu_thr   =[1, 5]                # numerosity must be inside this range
fwhm_thr =[0, 24]               # tuning width must be inside this range
beta_thr =[0, np.inf]           # scaling parameter must be inside this range
crit_def = 'Rsqmb'              # default thresholding option (see "threshold_maps")

# specify default covariates
#-----------------------------------------------------------------------------#
covs = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
        'white_matter', 'csf', 'global_signal', \
        'cosine00', 'cosine01', 'cosine02']

# specify additional covariates
#-----------------------------------------------------------------------------#
covs_add = []

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
        
            subj_id - string; subject identifier (e.g. "EDY7")
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
    
    # function: get spike regressors "sub-{sub}_ses-{ses}_run-{run}_spikeregressor.csv" filenames
    #-------------------------------------------------------------------------#
    def get_spregressor_csv(self, run_no):
        """
        Get Filename for Confounds TSV File
        filename = sess.get_spregressor_csv(run_no)
        
            run_no   - int; run number (e.g. 1)
            
            filename - string; filename of "spikeregressor.csv"
        
        filename = get_spregressor_csv(run_no) returns the filename of the
        comma-separated regressor columns file belonging to session sess and run run_no.
        """
        
        # create filename
        filename = '/data/u_jeong_software' + '/EMPRISE' + '/code' + '/Python' +'/Tables' \
                   '/sub-' + self.sub + '_ses-' + self.ses + '_run-' + str(run_no) + '_spikeregressor.csv'
        
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
            
            # specify mesh files (New preprocessing - the name of inflated surface file is /sub-000_ses-visual_run-0_hemi-L_inflated.surf.gii)
            prep_dir  = deri_dir + 'fmriprep'
            mesh_path = prep_dir + '/sub-' + self.sub + '/anat' + \
                                   '/sub-' + self.sub + '*' + '_hemi-'
            mesh_file = mesh_path + 'L' + '_' + surface + '.surf.gii'
            if not glob.glob(mesh_file):
                for ses in sess:
                    mesh_path = prep_dir + '/sub-' + self.sub + '/ses-' + ses + '/anat' + \
                                           '/sub-' + self.sub +  '*' + '_hemi-'
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
            mesh_dict = {'inflated':     'infl', \
                         'pial':         'pial', \
                         'white':        'white', \
                         'midthickness': 'midthick'}
            
            # specify mesh files
            if surface not in mesh_dict.keys():
                mesh_files = {'left' : 'n/a', \
                              'right': 'n/a'}
            else:
                free_dir   = deri_dir + 'freesurfer'
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
            sulc_files = {'left' : sulc_path + '/lh.sulc', \
                          'right': sulc_path + '/rh.sulc'}
        
        # if average image space
        elif space == 'fsaverage':
            
            # specify mesh files
            free_dir   = deri_dir + 'freesurfer'
            sulc_path  = free_dir + '/fsaverage/sulc'
            sulc_files = {'left' : sulc_path + '_left.gii', \
                          'right': sulc_path + '_right.gii'}
        
        # return mesh files
        return sulc_files
    
    #function: get valid runs 
    #-------------------------------------------------------------------------#
    def get_valid_runs(self):

        """

        valid_runs = sess.get_valid_runs(self, crs_val = False)

            valid_runs = { valid run :  its scan number} returns a dictionary.
            o key   - Valid run, integer
            o value - Scan numbers of a valid run, integer


        
        The choice of valid runs are dependent on the following criteria:
         1. The maximum number of scan is equal to the full scan numbers of an exepriment.
            (In EMPRISE7t case, 145)
         2. Only the runs have more than 3/4 of the maximum number of scan are valid runs
            to analyze.(An arbitarary choice for EMPRISE7t)
         3. Among all runs, there is at least one run which its scan number is equal to
           the maximum number of scan.
        """
        
        # extract list of exist runs and their scan number
        
        exist_runs = []
        scan_num = []
        for run in runs:
            filename = self.get_bold_nii(run, space='')
            if os.path.isfile(filename):
                Y = self.get_bold_nii(run, space='')
                Y = nib.load(Y)
                Y = Y.get_fdata()
                n = Y.shape[-1]
                scan_num.append(n)
                exist_runs.append(run)

        max_num = max(scan_num)
        
        # Runs that their scan number meets the criteria
        valid_runs = {}
        
        for j, run in enumerate(exist_runs):
            if scan_num[j] > max_num*3/4:
                valid_runs[run] = scan_num[j]

        return valid_runs        
           

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
            return Y """
    
        # perparing 3D array 
        
        VR = self.get_valid_runs()
        runs = list(VR.keys())

        # get n
        n = min(VR.values())
        
        # get V (assumption: all runs have the same spacial dimension)
        Y = self.load_data(runs[0],space)
        
        # 3D array with zeros
        Y = np.zeros((n, Y.shape[1],len(VR)))

        #load fMRI data
        for j, run in enumerate(runs):
            filename = self.get_bold_nii(run, space)
            if os.path.isfile(filename):
                Y[:,:,j] = self.load_data(run, space)[:n, :]

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
        return Y
    
    # function: load surface fMRI data (all runs)
    #-------------------------------------------------------------------------#
    def load_surf_data_all(self, hemi='L', space='fsnative', crs_val = False):
        """
        Load Functional MRI GIfTI Files from All Runs
        Y = sess.load_surf_data_all(hemi, space)
            
            hemi  - string; brain hemisphere (e.g. "L")
            space - string; image space (e.g. "fsnative")


            if crs_val = False:
            Y     - n x V x r array; scan-by-vertex-by-run fMRI data

            if crs_val = True:
            Y     - dictionary; {'even': n x V x even runs, 'odd': n x V x odd runs}
        """

        # preparing 3D array

        VR = self.get_valid_runs() 
        runs = list(VR.keys())
        
        # length of scans
        n = min(VR.values())

        # size of vertex
        Y = self.load_surf_data(runs[0], hemi, space)
        V = Y.shape[1]

        # load fMRI data
        if not crs_val:
            # 3D array
            Y = np.zeros((n, V, len(runs)))
            for j, run in enumerate(runs):
                filename = self.get_bold_gii(run, hemi, space)
                if os.path.isfile(filename):
                    Y[:,:,j] = self.load_surf_data(run, hemi, space)[:n, :]
        else:
            # Dictionary of 3D array
            Y = {}
            # even and odd runs
            odd_runs = [runs[i] for i in range(0,len(runs)) if i%2 == 0]
            even_runs = [runs[i] for i in range(0,len(runs)) if i%2 == 1]
            # 3D array for even and odd runs
            Y_even = np.zeros((n, V, len(even_runs)))
            Y_odd = np.zeros((n, V, len(odd_runs)))

            for j, run in enumerate(even_runs):
                filename = self.get_bold_gii(run, hemi, space)
                if os.path.isfile(filename):
                    Y_even[:,:,j] = self.load_surf_data(run, hemi, space)[:n, :]
            for j, run in enumerate(odd_runs):
                filename = self.get_bold_gii(run, hemi, space)
                if os.path.isfile(filename):
                    Y_odd[:,:,j] = self.load_surf_data(run, hemi, space)[:n, :]

            Y['even'] = Y_even
            Y['odd'] = Y_odd

        return Y
    
    # function: get onsets and durations
    #-------------------------------------------------------------------------#
    def get_onsets(self):
        """
        Get Onsets and Durations for Single Subject and Session, all Runs
        ons, dur, stim = sess.get_onsets()
        
            ons  - list of arrays of floats; t x 1 vectors of onsets [s]
            dur  - list of arrays of floats; t x 1 vectors of durations [s]
            stim - list of arrays of floats; t x 1 vectors of stimuli (t = trials)
            
        ons, dur, stim = sess.get_onsets() loads the "events.tsv" file
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
            stimuli = {'1_dot': 1, '2_dot': 2, '3_dot': 3, '4_dot': 4, '5_dot': 5, '20_dot': 20}
        elif self.ses == 'digits':
            stimuli = {'1_digit': 1, '2_digit': 2, '3_digit': 3, '4_digit': 4, '5_digit': 5, '20_digit': 20}
        elif self.ses == 'audio' or self.ses == 'spoken':
            stimuli = {'1_audio': 1, '2_audio': 2, '3_audio': 3, '4_audio': 4, '5_audio': 5, '20_audio': 20}
        
        # all dictionary of valid runs (updated)
        VR = self.get_valid_runs()

        # for all valid runs
        for run in VR:        
            
            # if onset file exists
            filename = self.get_events_tsv(run)
            if os.path.isfile(filename):
            
                # extract events of interest
                events = pd.read_csv(filename, sep='\t')
                events = events[events['trial_type']!='button_press']
                for code in stimuli.keys():
                    if code not in events['trial_type'].values:
                        events['num'] = events['trial_type'].str.split('_').str[0]
                        events['num'] = events['num'].str.split('.').str[0]
                        events['type'] = events['trial_type'].str.split('_').str[1]
                        events['trial_type'] = events['num']+ '_' + events['type']
                        events.drop(['num', 'type'], axis=1, inplace=True)
                    else:
                        events.loc[events['trial_type']==code+'_attn','trial_type'] = code
                
                # save onsets, durations, stimuli
                stims = [stimuli[trl] for trl in events['trial_type']]
                ons.append(np.array(events['onset']))
                dur.append(np.array(events['duration']))
                stim.append(np.array(stims))
            else:
                print('No events file for run', run)    
        # return onsets
        return ons, dur, stim

    # function: get confound variables
    #-------------------------------------------------------------------------#
    def get_confounds(self, labels):
        """
        Get Confound Variables for Single Subject and Session, all Runs
        X_c = sess.get_confounds(labels)
        
            labels  - list of strings; confound file header entries
            
            X_c     - n x c x r array; confound variables
                     (n = scans, c = variables, r = runs)
            
        X_c = sess.get_confounds() loads the "timeseries.tsv" file belonging
        to session sess and returns a scan-by-variable-by-run array of those
        confound variables indexed by the list labels. The function applies
        no preprocessing to the confounds.
        """

        """ 
        # prepare confound variables as zero matrix
        c   = len(labels)
        r   = len(runs)
        X_c = np.zeros((n,c,r))
        
        # for all runs
        for j, run in enumerate(runs):
            
            # if confound file exists
            filename = self.get_confounds_tsv(run)
            if os.path.isfile(filename):
            
                # save confound variables
                confounds = pd.read_csv(filename, sep='\t')
                for k, label in enumerate(labels):
                    X_c[:,k,j] = np.array(confounds[label])
        
        # select available runs
        X_c = X_c[:,:,np.any(X_c, axis=(0,1))]
        
        # return confounds
        return X_c 
        
        """

        # prepare confound variables as zero matrix
        VR = self.get_valid_runs()
        runs = list(VR.keys())
        c = len(labels)
        r = len(runs)
        n = min(VR.values())
        X_c = np.zeros((n,c,r))

        #for all valid runs
        for j, run in enumerate(runs):

            #save confound variables
            filename = self.get_confounds_tsv(run)
            confounds = pd.read_csv(filename, sep='\t')
            for k, label in enumerate(labels):
                X_c[:,k,j] = np.array(confounds[label][:n])

        return X_c        
    

        # function: get aCompCor confound variables (white_matter + csf)
    #-------------------------------------------------------------------------#
    def get_acom_confounds(self):
        """
        Get Confound Variables for Single Subject and Session, all Runs
        X_acom_list = sess.get_acom_confounds()
        
            X_acom_list     - list with length r; each item is n x c array; confound variables
                        explaining over 50% of variance in white matter and csf signals
                     (n = scans, c = variables, r = runs)
            
        X_acom_list = sess.get_acom_confounds() loads the "timeseries.tsv" file belonging
        to session sess and returns a scan-by-variable array of those
        confound variables indexed by the list labels and make a list which its length is the numbers of runs. 
        The function applies no preprocessing to the confounds.
        """


        # prepare confound variables as zero matrix
        VR = self.get_valid_runs()
        runs = list(VR.keys())
        r = len(runs)
        n = min(VR.values())

        # initialize X_c_list as list
        X_acom_list= [] 

        #for all valid runs
        for j, run in enumerate(runs):

            #save confound variables
            filename = self.get_confounds_tsv(run)
            confounds = pd.read_csv(filename, sep='\t')
            labels = list(confounds.columns.values)
            pattern = "a_comp_cor_" + ".*"
            labels = [label for label in labels if re.match(pattern, label)]
            c = len(labels)
            X_c = np.zeros((n,c)) # initialize X_c as 2D array   
            X_c[:, :] = np.array(confounds[labels][:n]) # Fill confound data
            X_acom_list.append(X_c)     

        return X_acom_list     

    # function: get spike regressor variables
    #-------------------------------------------------------------------------#
    def get_spregressor(self, threshold = None):
        """
        Get Confound Variables for Single Subject and Session, all Runs
        X_c_list = sess.get_confounds(threshold)
        
            threshold  - regressor file header entries
            
            X_c_list   - a list of n x c arrays which its length is r; confound variables
                     (n = scans, c = variables, r = runs)
            
        X_c_list = sess.get_spregressor() loads the "spikeregressor.csv" file belonging
        to session sess and returns a list of scan-by-variable array along its run of those
        confound variables indexed by the list labels. The function applies
        no preprocessing to the confounds.
        """
        # get valid runs
        VR = self.get_valid_runs()

        # prepare confound variables as zero matrix
        runs = list(VR.keys())
        r = len(runs)
        n = min(VR.values()) 
        
        X_c_list= [] # initialize X_c_list as list

        # For all runs
        for j, run in enumerate(runs):
            # Get confound labels and data
            filename = self.get_spregressor_csv(run)
            confounds = pd.read_csv(filename)
            labels = list(confounds.columns.values)
            pattern = "th_" + str(threshold) + ".*"
            labels = [label for label in labels if re.match(pattern, label)]
            c = len(labels)
            X_c = np.zeros((n,c)) # initialize X_c as 2D array   
            X_c[:, :] = np.array(confounds[labels][:n]) # Fill confound data
            # Check if there is null column, if so, remove it
            zero_columns = np.all(X_c == 0, axis=0)
            X_c = X_c[:, ~zero_columns]
            X_c_list.append(X_c)     

        return X_c_list

# class: subject/session
#-----------------------------------------------------------------------------#
class Model(Session):
    """
    A Model object is initialized by subject/session/space IDs and model name
    and allows for multiple operations related to numerosity estimation.
    """
    
    # function: initialize model
    #-------------------------------------------------------------------------#
    def __init__(self, subj_id, sess_id, mod_name, space_id='fsnative', hemis = None):
        """
        Initialize a Model applied to a Session
        mod = EMPRISE.Model(subj_id, sess_id, mod_name, space_id)
        
            subj_id  - string; subject identifier (e.g. "EDY7")
            sess_id  - string; session identifier (e.g. "visual")
            mod_name - string; name for the model (e.g. "NumAna")
            space_id - string; space identifier (e.g. "fsnative")
            hemis    - string; hemisphere (e.g. "L", None means both hemispheres)
            
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
        self.hemi = hemis
    
    # function: model directory
    #-------------------------------------------------------------------------#
    def get_model_dir(self):
        """
        Get Folder Name for Model
        mod_dir = mod.get_model_dir()
        
            mod_dir - string; directory where the model is saved
        """
        
        # create folder name
        nprf_dir = deri_out + 'numprf'
        mod_dir  = nprf_dir + '/sub-' + self.sub + '/ses-' + self.ses + '/model-' + self.model
        return mod_dir
    
    # function: results file
    #-------------------------------------------------------------------------#
    def get_results_file(self, hemi='L', SPR_th=None):
        """
        Get Results Filename for Model
        res_file = mod.get_results_file(hemi)
        
            hemi     - string; brain hemisphere (e.g. "L")
            SPR_th   - float; spike regressor threshold (e.g. 0,5), one should use comma to separate the integer and decimal part
            res_file - string; results file into which the model is written
        """
        
        # create filename
        mod_dir  = self.get_model_dir()
        if SPR_th == None:

            res_file = mod_dir  + f'/sub-{self.sub }_ses-{self.ses}_model-{self.model}_hemi-{hemi}_space-{self.space}_numprf.mat'

        else:
            res_file = mod_dir  + f'/sub-{self.sub }_ses-{self.ses}_model-{self.model}_hemi-{hemi}_space-{self.space}_spikereg-{SPR_th}_numprf.mat'
        
        return res_file
    
    # function: calculate runs/scans
    #-------------------------------------------------------------------------#
    def calc_runs_scans(self, SPR_th=None):
        """
        Calculate Number of Runs and Scans
        r0, n0 = mod.calc_runs_scans()
        
            r0 - int; number of runs analyzed, depending on averaging across runs
            n0 - int; number of scans per run, depending on averaging across epochs
        """
        
        # load results file
        if SPR_th == None:
            res_file = self.get_results_file()

        else:
            res_file = self.get_results_file('L', SPR_th)

        NpRF     = sp.io.loadmat(res_file)
        
        # count number of runs
        r0  = 0
        for run in runs:
            filename = self.get_events_tsv(run)
            if os.path.isfile(filename): r0 = r0 + 1
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
    def load_mask_data(self, hemi='L', crs_val=False):
        """
        Load Functional MRI GIfTI Files and Mask
        Y = sess.load_mask_data(hemi, space)
            
            hemi  - string; brain hemisphere (e.g. "L")
            
            Y     - n x v x r array; scan-by-vertex-by-run fMRI data
            M     - v x 1 vector; mask of vertices

            if crs_val = True:
            
            Y     - dictionary; {'even': n x V x even runs, 'odd': n x V x odd runs}
            M     - dictionary; {'even': V x 1 vector, 'odd': V x 1 vector}
        """
        
        # load 
        if not crs_val:
            Y = self.load_surf_data_all(hemi, self.space)

            # load mask
            M = np.all(Y, axis=(0,2))
            Y = Y[:,M,:]

        else:
            Y = self.load_surf_data_all(hemi, self.space, crs_val=True)
            M = {}
            M['even'] = np.all(Y['even'], axis=(0,2))
            M['odd']= np.all(Y['odd'], axis=(0,2))
            Y['even'] = Y['even'][:,M['even'],:]
            Y['odd'] = Y['odd'][:,M['odd'],:]
    
        
        # return data and mask
        return Y, M
    
    # function: analyze numerosities
    #-------------------------------------------------------------------------#
    def analyze_numerosity(self, avg=[True, False], corr='iid', order=1, ver='V2', crs_val=False, SPR_th=None):
        """
        Estimate Numerosities and FWHMs for Surface-Based Data
        results = mod.analyze_numerosity(avg, corr, order, ver)
        
            avg     - list of bool; see "NumpRF.estimate_MLE" (default: [True, False])
            corr    - string; see "NumpRF.estimate_MLE" (default: "iid")
            order   - int; see "NumpRF.estimate_MLE" (default: 1)
            ver     - string; version identifier (default: "V2")
            
            results - dict of strings; results filenames
            o L     - results for left hemisphere
            o R     - results for right hemisphere
            
        results = mod.analyze_numerosity(avg, corr, order, ver) loads the
        surface-based pre-processed data belonging to model mod, estimates
        tuning parameters using settings avg, corr, order, ver and saves
        results into a single-subject results directory.
        
        The input parameter "ver" (default: "V2") controls which version of
        the routine is used (for details, see "NumpRF.estimate_MLE"):
            V0:       mu_grid   = [3, 1]
                      fwhm_grid = [10.1, 5] (see "NumpRF.estimate_MLE_rgs")
            V1:       mu_grid   = {0.05,...,6, 10,20,...,640,1280} (128)
                      fwhm_grid = {0.3,...,18, 24,48,96,192} (64)
            V2:       mu_grid   = {0.8,...,5.2, 20} (90)
                      sig_grid  = {0.05,...,3} (60)

        The input parameter "SPR_th" (default: None) controls whether spike regressors
        classfied by the frame-wise displacement threshold are used as confounds. For example, 
        if "SPR_th" is set to 0.5, spike regressors classifeid with FD > 0,5 are used as confounds. 
        If "SPR_th" is set to None, standard confounds in the cov list are used. 
        In our study, there are three options of the threshold: 0.5, 0.8, and 1.7.

        For the cross validation test: crs_val = True, the function returns the results for even and odd runs.
        
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
        if SPR_th == None and not crs_val:
                                
            X_c = self.get_confounds(covs)
            X_c = standardize_confounds(X_c)
            
            print('successful!')

        # load confounds for cross validation
        elif SPR_th == None and crs_val:    
            All_X_c = self.get_confounds(covs)
            All_X_c = standardize_confounds(All_X_c)

            # classify runs as even and odd
            odd_runs = [i for i in range(0,All_X_c.shape[2]) if i%2 == 0]
            even_runs = [i for i in range(0,All_X_c.shape[2]) if i%2 == 1]
            X_c = {}
            X_c['even'] = All_X_c[:,:,even_runs]
            X_c['odd'] = All_X_c[:,:,odd_runs]

            print('Even/Odd runs: successful!')

        # load spike regressors and aCompCor confounds
        elif SPR_th is not None:    
            # load spike regressors
            X_sp = self.get_spregressor(threshold=SPR_th)
            # load aCompCor confounds
            # X_acom = self.get_acom_confounds()
            # combine spike regressors and aCompCor confounds
            #X_ex_com = []  # initialize X_ex_com as list
            #for i in range(len(X_acom)):
            #    X_ex_com.append(np.concatenate((X_acom[i], X_sp[i]), axis=1))

            #X_sp = X_ex_com
            X_c = self.get_confounds(covs)
            #un_wanted = {'white_matter', 'csf'} # remove white matter and csf confounds
            #X_c = np.delete(X_c, [covs.index(cov) for cov in un_wanted], axis=1)
            X_c = standardize_confounds(X_c)

            print('successful! \n   - Using spike regressors with threshold {} ... '.format(SPR_th))
            
        
        # specify grids
        if ver == 'V0':
            mu_grid   = [ 3.0, 1.0]
            fwhm_grid = [10.1, 5.0]  
        elif ver == 'V1':
            mu_grid   = np.concatenate((np.arange(0.05, 6.05, 0.05), \
                                        10*np.power(2, np.arange(0,8))))
            fwhm_grid = np.concatenate((np.arange(0.3, 18.3, 0.3), \
                                        24*np.power(2, np.arange(0,4))))
        elif ver == 'V2':
            mu_grid   = np.concatenate((np.arange(0.80, 5.25, 0.05), \
                                        np.array([20])))
            sig_grid  = np.arange(0.05, 3.05, 0.05)
        else:
            err_msg = 'Unknown version ID: "{}". Version must be "V0" or "V1" or "V2".'
            raise ValueError(err_msg.format(ver))
        
        # part 2: analyze both hemispheres
        #---------------------------------------------------------------------#
        if self.hemi == None:
            hemis   = ['L', 'R']
        else:
            hemis = [self.hemi]

        results = {}

        for hemi in hemis:
            
            # load data
            print('\n-> Hemisphere "{}", Space "{}":'.format(hemi, self.space))
            print('   - Loading fMRI data ... ', end='')

            if not crs_val:
                Y, M = self.load_mask_data(hemi)
                Y    = standardize_signals(Y)
                V    = M.size
                print('successful!')
            else:
                Y, M = self.load_mask_data(hemi, crs_val=True)
                Y['even'] = standardize_signals(Y['even'])
                Y['odd'] = standardize_signals(Y['odd'])
                V ={}
                V['even'] = M['even'].size
                V['odd'] = M['odd'].size
                print('Even/Odd runs: successful!')
            
            # analyze data
            print('   - Estimating parameters ... ', end='\n')
            if SPR_th is None:
                ds = NumpRF.DataSet(Y, ons, dur, stim, TR, X_c, None)
            else:
                ds = NumpRF.DataSet(Y, ons, dur, stim, TR, X_c, X_sp)

            start_time = time.time()

            if ver == 'V0' and not crs_val:
                mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                    ds.estimate_MLE_rgs(avg=avg, corr=corr, order=order, mu_grid=mu_grid, fwhm_grid=fwhm_grid)
                k_est, k_null, k_const =\
                    ds.free_parameters(avg, corr, order)
            elif ver == 'V1' and not crs_val:
                mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                    ds.estimate_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, fwhm_grid=fwhm_grid)
                k_est, k_null, k_const =\
                    ds.free_parameters(avg, corr, order)
            elif ver == 'V2' and not crs_val:
                mu_est, fwhm_est, beta_est, MLL_est, MLL_null, MLL_const, corr_est =\
                    ds.estimate_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, sig_grid=sig_grid)
                k_est, k_null, k_const =\
                    ds.free_parameters(avg, corr, order)
            elif ver == 'V2' and crs_val:
                mu_est = {}
                fwhm_est = {}
                beta_est = {}
                MLL_est = {}
                MLL_null = {}
                MLL_const = {}
                corr_est = {}
                k_est = {}
                k_null = {}
                k_const = {}

                for run in list(Y.keys()):
                    ds = NumpRF.DataSet(Y[run], ons, dur, stim, TR, X_c[run], None)
                    mu_est[run], fwhm_est[run], beta_est[run], MLL_est[run], MLL_null[run], MLL_const[run], corr_est[run] =\
                    ds.estimate_MLE(avg=avg, corr=corr, order=order, mu_grid=mu_grid, sig_grid=sig_grid)
                    k_est[run], k_null[run], k_const[run] =\
                    ds.free_parameters(avg, corr, order)    

            end_time   = time.time()
            difference = end_time - start_time
            del start_time, end_time
            
            # save results (mat-file)
            sett = str(avg[0])+','+str(avg[1])+','+str(corr)+','+str(order)
            print('\n-> Model "{}", Settings "{}":'.format(self.model, sett))
            print('   - Saving results file ... ', end='')
            if SPR_th == None and not crs_val:
                filepath = mod_dir + '/sub-' + self.sub + '_ses-' + self.ses + \
                                 '_model-' + self.model + '_hemi-' + hemi + '_space-' + self.space + '_'
            elif SPR_th == None and crs_val:
                filepath = mod_dir + '/sub-' + self.sub + '_ses-' + self.ses + \
                                 '_model-' + self.model + '_hemi-' + hemi + '_space-' + self.space + '_crs-val-'
            else:
                filepath = mod_dir + '/sub-' + self.sub + '_ses-' + self.ses + \
                                 '_model-' + self.model + '_hemi-' + hemi + '_space-' + self.space + '_spikereg-' + str(SPR_th) + '_'


            if type(mu_est) == dict:
                results[hemi] = {}
                for run in mu_est.keys():
                    results[hemi][run] = filepath + str(run) + '_numprf.mat'
                    res_dict = {'mod_dir': mod_dir, 'settings': {'avg': avg, 'corr': corr, 'order': order}, \
                                'mu_est':  mu_est[run],  'fwhm_est': fwhm_est[run], 'beta_est': beta_est[run], \
                                'MLL_est': MLL_est[run], 'MLL_null': MLL_null[run], 'MLL_const': MLL_const[run], \
                                'k_est':   k_est[run],   'k_null':   k_null[run],   'k_const':   k_const[run], \
                                'corr_est':corr_est[run], 'time':     difference}
                    sp.io.savemat(results[hemi][run], res_dict)
                print('Even/Odd runs: successful!')
                del sett, res_dict
            else:  
                results[hemi] = filepath + 'numprf.mat'
                res_dict = {'mod_dir': mod_dir, 'settings': {'avg': avg, 'corr': corr, 'order': order}, \
                            'mu_est':  mu_est,  'fwhm_est': fwhm_est, 'beta_est':  beta_est, \
                            'MLL_est': MLL_est, 'MLL_null': MLL_null, 'MLL_const': MLL_const, \
                            'k_est':   k_est,   'k_null':   k_null,   'k_const':   k_const, \
                            'corr_est':corr_est,'time':     difference}
                sp.io.savemat(results[hemi], res_dict)
                print('successful!')
                del sett, res_dict
            
            # save results (surface images)
            para_est = {'mu': mu_est, 'fwhm': fwhm_est, 'beta': beta_est}

            # prepare variables: we take it from the runs we used for anlaysis
            VR = self.get_valid_runs()
            runs = list(VR.keys())
            # prepare a surface image to map on the result
            surface = nib.load(self.get_bold_gii(runs[0],hemi,self.space)) 

            for name in para_est.keys():
                print('   - Saving "{}" image ... '.format(name), end='')
                if crs_val:
                    para_map = {}
                    for run in Y.keys():
                        para_map[run]    = np.zeros(V[run], dtype=np.float32)
                        para_map[run][M[run]] = para_est[name][run]
                        filename    = filepath + name + '_' + run + '.surf.gii'
                        para_img    = save_surf(para_map[run], surface, filename)
                    print('Even/Odd runs: successful!')
                else:
                    para_map    = np.zeros(V, dtype=np.float32)
                    para_map[M] = para_est[name]
                    filename    = filepath + name + '.surf.gii'
                    para_img    = save_surf(para_map, surface, filename)
                    print('successful!')
            del para_est, para_map, surface, filename, para_img
        
        # return results filename
        return results
    
    # function: threshold tuning maps
    #-------------------------------------------------------------------------#
    def threshold_maps(self, crit='Rsqmb'):
        """
        Threshold Numerosity, FWHM and Scaling Maps based on Criterion
        maps = mod.threshold_maps(crit)
        
            crit - string; criteria used for thresholding maps
                          (default: "Rsqmb"; see below for details)
            
            maps - dict of dicts; thresholding tuning maps
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
        o ","  : preceeding user-defined R^2 threshold (e.g. "Rsq,0.25")
        
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
        
        # part 2: threshold both hemispheres
        #---------------------------------------------------------------------#
        hemis = {'L': 'left', 'R': 'right'}
        maps  = {'mu': {}, 'fwhm': {}, 'beta': {}, 'Rsq': {}}
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
            MLL1  = np.squeeze(NpRF['MLL_est'])
            MLL0  = np.squeeze(NpRF['MLL_null'])
            MLL00 = np.squeeze(NpRF['MLL_const'])
            k1    = NpRF['k_est'][0,0]
            k0    = NpRF['k_null'][0,0]
            
            # compute quantities for thresholding
            print('     - Applying threshold criteria "{}" ... '.format(crit), end='')
            dAIC  = (-2*MLL0 + 2*k0) - (-2*MLL1 + 2*k1)
            # See: https://statproofbook.github.io/P/mlr-aic
            dBIC  = (-2*MLL0 + k0*np.log(n1)) - (-2*MLL1 + k1*np.log(n1))
            # See: https://statproofbook.github.io/P/mlr-bic
            Rsq   = NumpRF.MLL2Rsq(MLL1, MLL00, n1)
            # See: https://statproofbook.github.io/P/rsq-mll
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
                Rsq_thr = Rsq_def
                if ',' in crit:
                    Rsq_thr = float(crit.split(',')[1])
                ind = np.logical_or(ind, Rsq<Rsq_thr)
            if 'm' in crit:
                ind = np.logical_or(ind, ind_m)
            if 'f' in crit:
                ind = np.logical_or(ind, ind_f)
            if 'b' in crit:
                ind = np.logical_or(ind, ind_b)
            print('successful!')
            
            # threshold tuning maps
            para_est = {'mu': mu, 'fwhm': fwhm, 'beta': beta, 'Rsq': Rsq}
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
            elif name == 'Rsq':
                caxis = [Rsq_thr,1]
                cmap  = 'hot'
                clabel= 'variance explained'
            
            # display and save plot
            figs[name] = plot_surf(maps[name], mesh_files, sulc_files, caxis, cmap, clabel)
            if crit:
                figs[name].savefig(filepath+name+'_thr-'+crit+'.png', dpi=200)
            elif img:
                figs[name].savefig(filepath+img+'.png', dpi=200)
        
        # return results filename
        return figs
    
    # function: threshold and cluster
    #-------------------------------------------------------------------------#
    def threshold_and_cluster(self, hemi='L', crit='Rsqmb', mesh='pial', ctype='coords', d=3, k=100):
        """
        Threshold and Cluster Vertices from Surface-Based Results
        verts, trias = mod.threshold_and_cluster(hemi, crit, mesh, ctype, d, k)

            hemi  - string; brain hemisphere ("L" or "R")
            crit  - string; criteria for thresholding (see "threshold_maps")
            mesh  - string; mesh file ("inflated", "pial", "white" or "midthickness")
            ctype - string; method of clustering ("coords" or "edges")
            d     - float; maximum distance of vertex to cluster
            k     - int; minimum number of vertices in cluster
            
            verts - array; v x 9 matrix of vertex properties
            o 1st           column: vertex index
            o 2nd           column: cluster index
            o 3rd, 4th, 5th column: mu, fwhm, beta
            o 6th           column: R-squared
            o 7th, 8th, 9th column: x, y, z
            trias - array; t x 3 matrix of surface triangles, t means the number of triangles
            o 1st, 2nd, 3rd column: vertex indices

        verts, trias = mod.threshold_and_cluster(hemi, crit, mesh, ctype, d, k) 
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
        
        # apply conditions for exclusion
        ind = mu > np.inf
        if 'AIC' in crit:
            ind = np.logical_or(ind, dAIC<dAIC_thr)
        if 'BIC' in crit:
            ind = np.logical_or(ind, dBIC<dBIC_thr)
        if 'Rsq' in crit:
            Rsq_thr = Rsq_def
            if ',' in crit:
                Rsq_thr = float(crit.split(',')[1])
            ind = np.logical_or(ind, Rsq<Rsq_thr)
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
    def threshold_AFNI_cluster(self, crit='Rsqmb', mesh='pial'):
        """
        Threshold, then AFNI SurfClust, then Extract Clusters
        verts, trias = mod.threshold_AFNI_cluster(crit, mesh)
        
            crit  - string; criteria for thresholding (see "threshold_maps")
            mesh  - string; mesh file ("inflated", "pial", "white" or "midthickness")
            
            verts - dict of arrays; vertex properties
            o left  - array; v x 8 matrix of left hemisphere vertices
            o right - array; v x 8 matrix of right hemisphere vertices
            trias - dict of arrays; surface triangles
            o left  - array; t x 3 matrix of left hemisphere triangles
            o right - array; t x 3 matrix of right hemisphere triangles
            verts, trias - see "threshold_and_cluster"
        
        verts, trias = mod.threshold_AFNI_cluster(crit, mesh) loads estimated
        NumpRF model results, (i) thresholds tuning parameter maps according
        to threshold crit (see "threshold_maps"), (ii) uses AFNI to perform
        surface clustering by edge distance and (iii) returns supra-threshold
        vertices and surface triangles.
        """
        
        # Step 1: R-squared map thresholding
        #---------------------------------------------------------------------#
        hemis    = {'L': 'left', 'R': 'right'}
        res_file = self.get_results_file('L')
        filepath = res_file[:res_file.find('numprf.mat')]
        beta_thr = filepath + 'beta' + '_thr-' + crit + '.surf.gii'
        
        # display message
        print('\n-> Subject "{}", Session "{}", Model "{}",\n   Space "{}", Surface "{}":'. \
              format(self.sub, self.ses, self.model, self.space, mesh))
        
        # threshold maps
        print('   - Step 1: threshold R-squared maps ... ', end='')
        if not os.path.isfile(beta_thr):
            maps = self.threshold_maps(crit)
            # dictionary "maps":
            # - keys "mu", "fwhm", "beta", "Rsq"
            #   - sub-keys "left", "right"
            print()
        else:
            print('already done.')
            maps = {}
            for para in ['mu','fwhm','beta','Rsq']:
                maps[para] = {}
                for hemi in hemis.keys():
                    res_file = self.get_results_file(hemi)
                    filepath = res_file[:res_file.find('numprf.mat')]
                    maps[para][hemis[hemi]] = filepath + para + '_thr-' + crit + '.surf.gii'
        
        # Step 2: AFNI surface clustering
        #---------------------------------------------------------------------#
        cls_sh    = tool_dir[:tool_dir.find('Python')] + 'Shell/' + 'cluster_surface' # Specify the folder where AFNI scripts are stored
        if self.space == 'fsnative':  cls_sh = cls_sh + '.sh'
        if self.space == 'fsaverage': cls_sh = cls_sh + '_fsa.sh'
        img_str   = 'space-' + self.space + '_' + 'Rsq' + '_thr-' + crit
        mesh_file = self.get_mesh_files(self.space, surface=mesh)['left']
        if self.space == 'fsnative':
            anat_pref = mesh_file[mesh_file.find('sub-')+len('sub-000/'):mesh_file.find('hemi-L')]
        if self.space == 'fsaverage':
            anat_pref = mesh_file[mesh_file.find('fsaverage/')+len('fsaverage/'):mesh_file.find('left.gii')]
        Rsq_cls   = filepath + 'Rsq' + '_thr-' + crit + '_cls-' + 'SurfClust' + '.surf.gii'
        
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
            Rsq  = nib.load(maps['Rsq'][h]).darrays[0].data
            
            # load surface mesh
            mesh_gii    = nib.load(mesh_files[h])
            XYZ         = mesh_gii.darrays[0].data
            trias[hemi] = mesh_gii.darrays[1].data
            
            # load cluster indices
            res_file = self.get_results_file(hemi)
            filepath = res_file[:res_file.find('numprf.mat')]
            Rsq_cls  = filepath + 'Rsq' + '_thr-' + crit + '_cls-' + 'SurfClust' + '_cls' + '.surf.gii'
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

    """     
    # create t, if necessary
    if t is None:
        t = np.arange(0, n*TR, TR) 
    """
    # creat t 
    if t is None:
        t = np.arange(0, Y.shape[0]*TR, TR)

    # average over runs
    if avg[0]:
        
        # if multiple runs
        if len(Y.shape) > 2:
            Y = np.mean(Y, axis=2)
    
    # average over epochs
    if avg[1]:
        
        # remove discard scans
        Y = Y[num_scan_disc:]
        
        """
        # if averaged over runs     
        if len(Y.shape) < 3:
            Y_epochs = np.zeros((scans_per_epoch,Y.shape[1],num_epochs))
            for i in range(num_epochs):
                Y_epochs[:,:,i] = Y[(i*scans_per_epoch):((i+1)*scans_per_epoch),:]
            Y = np.mean(Y_epochs, axis=2) 
        """

        # if averaged over runs
        if len(Y.shape) <3:
            # When one of the runs has no full scan numbers
            if Y.shape[0]%scans_per_epoch != 0: 

                # We choose only the runs having more than 3/4 of the full scan numbers.
                # When your experiment has n cycles per run, then one should change this criteria in the get_valid_runs function.
                # When (n-1)/n is your threshold for the get_valid_runs function, you don't need to change the below code. 
                # Change the code with input function so one can give directly the wished number of scans to average across epochs. 
                # But temporariry, set 2 missing scans as the maximum missing scans of the last run.
                # therfore, there is a chance that the last quater epoch has a shorter scan number than other its prefious epochs.
                short_scan_num = Y.shape[0]- scans_per_epoch*(num_epochs-1)
                missing_scan = scans_per_epoch - short_scan_num
                if missing_scan > 2:
                    Y_epochs = np.zeros((scans_per_epoch,Y.shape[1],num_epochs-1))
                    for i in range(num_epochs-1):
                        Y_epochs[:,:,i] = Y[(i*scans_per_epoch):((i+1)*scans_per_epoch),:]
                    Y = np.mean(Y_epochs, axis=2)

                #if missing scan is smaller than 2
                else:
                    Y_epochs = np.zeros((short_scan_num,Y.shape[1],num_epochs))
                    for i in range(num_epochs):
                        Y_epochs[:,:,i] = Y[(i*(scans_per_epoch)):((i+1)*scans_per_epoch - missing_scan),:]
                    Y = np.mean(Y_epochs,axis=2)

            # When there is no missing scans from any runs        
            else:
                Y_epochs = np.zeros((scans_per_epoch,Y.shape[1],num_epochs))
                for i in range(num_epochs):
                    Y_epochs[:,:,i] = Y[(i*scans_per_epoch):((i+1)*scans_per_epoch),:]
                Y = np.mean(Y_epochs, axis=2)

        # if not averaged over runs
        else:
            Y_epochs = np.zeros((scans_per_epoch,Y.shape[1],Y.shape[2],num_epochs))
            for i in range(num_epochs):
                Y_epochs[:,:,:,i] = Y[(i*scans_per_epoch):((i+1)*scans_per_epoch),:,:]
            Y = np.mean(Y_epochs, axis=3)
        
        # correct time vector
        t = t[num_scan_disc:]
        t = t[:scans_per_epoch] - num_scan_disc*TR
    
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
        stim   - b0 x 1 vector; block-wise stimuli (b = blocks per epoch)
    
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
                size=(1600,600), zoom=1.5)
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
    what_to_test = 'create_fsaverage_midthick'
    
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
        sess = Session('EDY7','visual')
        M    = sess.load_mask(1,'T1w')
        print('The images have {} voxels of which {} are inside the brain mask.'. \
              format(M.size, np.sum(M==1)))
    
    # test "load_data"
    if what_to_test == 'load_data':
        sess   = Session('EDY7','visual')
        M      = sess.load_mask(1,'T1w')
        Y      = sess.load_data(1,'T1w')
        Y_mask = Y[:,M==1]
        print("The data are a {} x {} matrix. When masked, it's a {} x {} matrix.". \
              format(Y.shape[0], Y.shape[1], Y_mask.shape[0], Y_mask.shape[1]))

    # test "load_data_all"
    if what_to_test == 'load_data_all':
        sess = Session('EDY7','visual')
        M    = sess.load_mask(1,'T1w')
        Y    = sess.load_data_all('T1w')
        Y    = Y[:,M==1,:]
        print('Masked data from all runs were loaded into a {} x {} x {} array.'. \
              format(Y.shape[0], Y.shape[1], Y.shape[2]))
    
    # test "load_surf_data"
    if what_to_test == 'load_surf_data':
        sess   = Session('EDY7','visual')
        Y      = sess.load_surf_data(1, 'L', 'fsnative')
        Y_mask = Y[:,np.all(Y, axis=0)]
        print("The data are a {} x {} matrix. When masked, it's a {} x {} matrix.". \
              format(Y.shape[0], Y.shape[1], Y_mask.shape[0], Y_mask.shape[1]))
    
    # test "load_surf_data_all"
    if what_to_test == 'load_surf_data_all':
        sess = Session('EDY7','visual')
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
    
    # test "average_signals"
    if what_to_test == 'average_signals':
        
        # load data
        sess = Session('EDY7','visual')
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
        sess = Session('EDY7','visual')
        ons, dur, stim = sess.get_onsets()
        ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
        ons, dur, stim = correct_onsets(ons[0], dur[0], stim[0])
        print(ons)
        print(dur)
        print(stim)
    
    # test "onsets_trials2blocks"
    if what_to_test == 'onsets_trials2blocks':
        sess = Session('EDY7','visual')
        ons, dur, stim = sess.get_onsets()
        print('trials: {} onsets, {} durations, {} stimuli.'. \
              format(len(ons[0]), len(dur[0]), len(stim[0])))
        ons, dur, stim = onsets_trials2blocks(ons, dur, stim, 'closed')
        print('blocks: {} onsets, {} durations, {} stimuli.'. \
              format(len(ons[0]), len(dur[0]), len(stim[0])))
    
    # test "create_fsaverage_midthick"
    if what_to_test == 'create_fsaverage_midthick':
        print(create_fsaverage_midthick())
    
    # test "save_vol"
    if what_to_test == 'save_vol':
        
        # load data and template
        sess = Session('EDY7','visual')
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
        sess = Session('EDY7','visual')
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