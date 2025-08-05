# Install library
#-------------------------------------------------------------------------#
import os
import json
import nibabel as nib
import numpy as np
import EMPRISE

# Load data
#-------------------------------------------------------------------------#
"""
Modify the json file for your analysis
condition: digit, spoken, visual, audio1, audio2
code_dir: specify the directory containing "config.json" file.
"""
code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"

config_file = os.path.join(code_dir, "config.json")
with open(config_file) as f:
    config = json.load(f)
globals().update(config)


# Parameters
#-------------------------------------------------------------------------#
subject_lists = ["N001", "N003", "N004", "N005","N006", "N007","N008", "N009","N011", "N012"]
for subject in subject_lists:
# subject  = SUBJECT
    session  = SESSION
    model    = MODEL # model name
    space    = SPACE
    target   = "all" if CV else "split" # "all" or "split"

    # start Session class
    #-------------------------------------------------------------------------#
    sess = EMPRISE.Session(subject, session)

    # start Model class
    #-------------------------------------------------------------------------#
    mod = EMPRISE.Model(subject, session, model, space_id=space)

    print(f'\n\n-> Subject "{mod.sub}", Session "{mod.ses}", Space "{mod.space}":')


    # load the subcortical result
    #-------------------------------------------------------------------------#
    mod_dir = mod.get_model_dir()
    file    = f'sub-{subject}_ses-{session}_model-{model}_space-{space}_mu_thr-Rsqmb,p=0.05B.nii.gz'
    fname   = os.path.join(mod_dir, file)
    img     = nib.load(fname) # Load the NIfTI file
    data    = img.get_fdata() # Access the data as a NumPy array


    # count the significant voxels
    #-------------------------------------------------------------------------#
    mask = (~np.isnan(data)) & (data != 0) # Create a mask that excludes NaNs and zeros
    min_voxel = np.min(data[mask]) # Apply mask and get the minimum value
    max_voxel = np.max(data[mask])

    mask = (~np.isnan(data)) & (data>=min_voxel) & (data<= max_voxel)
    n_sigvoxels = np.sum(mask)

    print(f'\n\n-> min mu "{min_voxel}", max mu "{max_voxel}", n_voxels "{n_sigvoxels}":')

