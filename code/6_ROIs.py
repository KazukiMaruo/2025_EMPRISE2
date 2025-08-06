# install library
#-------------------------------------------------------------------------#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import nibabel as nib
import nibabel.freesurfer.io as fsio
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
import EMPRISE
from nilearn import surface
from surfplot import Plot
from nibabel.gifti import GiftiImage, GiftiDataArray
from matplotlib.colors import ListedColormap


# Parameters
#-------------------------------------------------------------------------#
count_thr     = 3 # number of particiapnts
hemis         = ['L', 'R']
subject_lists = ["N001", "N004", "N005","N006", "N007","N008","N011", "N012"]
surf_imgs     = {}

for hemi in hemis:

    subject_arrays = []
    for subject in subject_lists:
        session  = "visual"
        model    = "NumAna" # model name # if you want a spatial smoothing, put 'FWHM' and '3' for the value
        space    = "fsaverage" # "fsnative" or "T1w"
        target   = "all" # "all" or "split"
        code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"
        
        # start Session class
        #-------------------------------------------------------------------------#
        sess = EMPRISE.Session(subject, session)

        # start Model class
        #-------------------------------------------------------------------------#
        mod = EMPRISE.Model(subject, session, model, space_id=space)

        # load the surface mask of cluster
        #-------------------------------------------------------------------------#
        # parameter
        Rsq_str = 'Rsq'
        crit    = 'Rsqmb,p=0.05B'

        # load cluster indices
        res_file = mod.get_results_file(hemi)
        res_data = sp.io.loadmat(res_file)
        n_vert   = res_data['mu_est'].shape[1]
        filepath = res_file[:res_file.find('numprf.mat')] 
        Rsq_cls  = filepath + Rsq_str + '_thr-' + crit + '_cls-' + 'SurfClust' + '_cls' + '.surf.gii'
        if os.path.isfile(Rsq_cls):
            clst      = nib.load(Rsq_cls).darrays[0].data
            clst_mask = np.where(clst>0, 1, 0)
        else:
            print(f"file not found: {Rsq_cls}, skipping.")
            clst_mask = np.zeros(n_vert)
        # store each subject index
        subject_arrays.append(clst_mask)

    # count the vertices clustered
    #-------------------------------------------------------------------------#
    all_sub = np.stack(subject_arrays)
    counts  = np.sum(all_sub, axis=0)

    surf_imgs[hemi] = counts


# save the counts as .gii file
#-------------------------------------------------------------------------#
output_dir = f'/data/pt_02495/emprise7t_2_analysis/derivatives/numprf/sub-all/ses-{session}/model-{model}/'
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

for hemi in ['L', 'R']:
    counts      = surf_imgs[hemi].astype(np.float32)
    gii_img     = GiftiImage(darrays=[GiftiDataArray(data=counts)])
    output_path = os.path.join(output_dir, f"group_cluster_counts_{hemi}.gii")
    nib.save(gii_img, output_path)

# save the thresholded counts as .gii file
#-------------------------------------------------------------------------#
for hemi in ['L', 'R']:
    counts      = surf_imgs[hemi].astype(np.float32)
    counts_thr  = np.where(counts >= count_thr, 1, 0)
    gii_img     = GiftiImage(darrays=[GiftiDataArray(data=counts_thr.astype(np.int32))])
    output_path = os.path.join(output_dir, f"group_cluster_counts_{hemi}_thr-{count_thr}.gii")
    nib.save(gii_img, output_path)


#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#

# Convert gii to label
#-------------------------------------------------------------------------#
# Load GIFTI surface label or mask
hemis = ['L', 'R']
for hemi in hemis:

    path     = f'/data/pt_02495/emprise7t_2_analysis/derivatives/numprf/sub-all/ses-{session}/model-{model}/'
    file     = f'ses-{session}_model-{model}_hemi-{hemi}_counts_cls.surf.gii'
    filepath = os.path.join(path, file)
    gii      = nib.load(filepath)
    data     = gii.darrays[0].data  # Assumes 1D data array per node

    # counts the number of cluster defined
    unique_vals, counts = np.unique(data, return_counts=True)

    # Example: keep all nodes with value > 0 (e.g., binary mask or cluster label)
    for cluster_id in unique_vals[1:]:
        label_nodes = np.where(data == cluster_id)[0]

        # Dummy coordinate + value data for FreeSurfer .label format
        coords         = np.zeros((len(label_nodes), 4))
        coords[:, 0]   = label_nodes
        coords[:, 1:4] = 1.0  # Dummy xyz values (not used)

        # Write .label file
        output_path = f"{path}/ROIs_cluster_{hemi}_{cluster_id}.label"
        with open(output_path, "w") as f:
            f.write("#!ascii label, from GIFTI\n")
            f.write(f"{len(label_nodes)}\n")
            for row in coords:
                f.write(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} 0.0\n")

#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#
#-------------------------------------------------------------------------#


# Plot
#-------------------------------------------------------------------------#
rename_map = {'L': 'left', 'R': 'right'}
for old_key, new_key in rename_map.items():
    if old_key in surf_imgs:
        surf_imgs[new_key] = surf_imgs.pop(old_key)

# specify mesh files
mesh_files = mod.get_mesh_files(space)
sulc_files = mod.get_sulc_files(space)

# load sulc images
sulc_data = {}
hemis = ['left', 'right']
for hemi in hemis:
    sulc_data[hemi] = surface.load_surf_data(sulc_files[hemi])
    sulc_data[hemi] = np.where(sulc_data[hemi]>0.0, 0.6, 0.2)

# specify surface plot
caxis  = [1, len(subject_lists)]
cmap   = 'gist_rainbow'
clabel = 'participant count'
plot = Plot(mesh_files['left'], mesh_files['right'],
            layout='row', views='lateral', size=(1600,600), zoom=1.5)
plot.add_layer(sulc_data, color_range=(0,1),
                cmap='Greys', cbar=False)
plot.add_layer(surf_imgs, color_range=(caxis[0],caxis[1]),
                cmap=cmap, cbar_label=clabel)

# display surface plot
filepath = '/data/pt_02495/emprise7t_2_analysis/Figures/visual/NumAna/'
cbar     = {'n_ticks': 8, 'decimals': 0, 'fontsize': 24}
fig      = plot.build(colorbar=True, cbar_kws=cbar)
fig.tight_layout()
fig.savefig(filepath+'Clustered-Countmap_'+'ses-'+session+'_'+space+'.png', dpi=300, transparent=True)
print('\n\n-> Successful!')


# Plot: ROI mask > 3 participants
#-------------------------------------------------------------------------#
rename_map = {'L': 'left', 'R': 'right'}
for old_key, new_key in rename_map.items():
    if old_key in surf_imgs:
        surf_imgs[new_key] = surf_imgs.pop(old_key)

# specify mesh files
mesh_files = mod.get_mesh_files(space)
sulc_files = mod.get_sulc_files(space)

# load sulc images
sulc_data = {}
hemis = ['left', 'right']
for hemi in hemis:
    sulc_data[hemi] = surface.load_surf_data(sulc_files[hemi])
    sulc_data[hemi] = np.where(sulc_data[hemi]>0.0, 0.6, 0.2)

# specify surface plot
view = 'lateral' # lateral or medial
plot = Plot(mesh_files['left'], mesh_files['right'],
            layout='row', views=view, size=(1600,600), zoom=1.5)
plot.add_layer(sulc_data, color_range=(0,1),
                cmap='Greys', cbar=False)
from matplotlib.colors import ListedColormap
green_cmap = ListedColormap(['#49EB0E'])
ROIs = {
    hemi: np.where(surf_imgs[hemi] >= 3, 1, 0)
    for hemi in ['left', 'right']
}
plot.add_layer(ROIs,cmap=green_cmap, color_range=(0, 1), cbar=False)

# display surface plot
filepath = '/data/pt_02495/emprise7t_2_analysis/Figures/visual/NumAna/'
fig      = plot.build()
fig.tight_layout()
fig.savefig(filepath+'ROIs_'+view+'_ses-'+session+'_'+space+'.png', dpi=300, transparent=True)
print('\n\n-> Successful!')


