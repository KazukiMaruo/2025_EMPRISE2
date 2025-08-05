#!/bin/bash
#-----------------------------------------------------------------------------#
# AFNI surface clustering for NumpRF tuning parameter maps
# AFNI <full_path_to_this_script> <sub> <ses> <model> <img> <anat>
# 
#     <full_path_to_this_script>
#             = /data/u_jeong_software
#               /EMPRISE/code/Shell/cluster_surface.sh
#     <sub>   - subject ID (e.g. "001")
#     <ses>   - session ID (e.g. "visual")
#     <model> - model name (e.g. "True_False_iid_1")
#     <img>   - image suffix (e.g. "space-fsnative_mu_thr-Rsq")
#     <anat>  - anat prefix in fMRIprep derivatives folder (e.g.
#               "ses-visual/anat/sub-001_ses-visual_acq-mprageised_")
# 
# This script takes a thresholded tuning parameter map (thresholded using
# e.g. R^2 or AIC) as well as the surface mesh image to which it belongs and
# (i) selects only vertices within a certain range, (ii) forms clusters of
# vertices based on maximum distance and (iii) saves the resulting clusters
# as another GIfTI file.
# 
# The script uses AFNI's SurfClust function and needs to be executed on MPI
# CBS servers with AFNI environment available (i.e. run "AFNI afni" before).
#
# Before run the scirpt, you need to change the permission of the script.
# chmod u+rx cluster_surface.sh
#
# Anne-Sophie Kieslinger, MPI Leipzig
# 2023-09-08, 14:02: first version ("surfaceclustering.sh")
# Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
# 2023-10-09, 16:44: adapted version ("cluster_surface.sh")
# 2023-10-12, 12:55: allowed for distance clustering
# 2023-10-16, 10:52: extended to both hemispheres
# 2023-11-17, 09:18: allowed for variable anat folder
# 2023-11-28, 16:03: changed to output cluster indices
# 2023-11-30, 18:51: removed cluster size threshold
# 2023-11-30, 21:24: replaced anat folder by anat prefix
#-----------------------------------------------------------------------------#

# print function call
echo $0 $@

# fail whenever something is fishy
# use -x to get verbose logfiles
set -e -u -x

# get inputs
sub=$1      # subject ID
ses=$2      # session ID
model=$3    # model name
img=$4      # image suffix
anat=$5		# anat prefix

# set paramters
d=1         # maximum distance between node and cluster, in number of edges
k=1         # minimum size of activated cluster, in number of nodes

# set directories
deri_dir="/data/pt_02495/emprise7t/derivatives"
deri_out="/data/u_jeong_software/derivatives"
subj_dir="$deri_out/numprf/sub-${sub}/ses-${ses}/model-${model}"
prefix="$subj_dir/SurfClust"

# specify surface (left hemisphere)
surface="$deri_dir/fmriprep/sub-${sub}/${anat}hemi-L_pial.surf.gii"
input="$deri_out/numprf/sub-${sub}/ses-${ses}/model-${model}/sub-${sub}_ses-${ses}_model-${model}_hemi-L_${img}.surf.gii"
out_file="$subj_dir/sub-${sub}_ses-${ses}_model-${model}_hemi-L_${img}_cls-SurfClust"

# perform AFNI SurfClust, convert AFNI outputs to GII, clean up
SurfClust -i $surface -input $input 0 -rmm -$d -n $k -prefix $prefix -out_clusterdset -out_roidset -out_fulllist
ConvertDset -o_gii -input "${prefix}_Clustered_e${d}_n${k}.niml.dset" -prefix "${out_file}.surf.gii"
ConvertDset -o_gii -input "${prefix}_ClstMsk_e${d}_n${k}.niml.dset" -prefix "${out_file}_cls.surf.gii"
rm -f "${prefix}_Cl"*

# specify surface (right hemisphere)
surface="$deri_dir/fmriprep/sub-${sub}/${anat}hemi-R_pial.surf.gii"
input="$deri_out/numprf/sub-${sub}/ses-${ses}/model-${model}/sub-${sub}_ses-${ses}_model-${model}_hemi-R_${img}.surf.gii"
out_file="$subj_dir/sub-${sub}_ses-${ses}_model-${model}_hemi-R_${img}_cls-SurfClust"

# perform AFNI SurfClust, convert AFNI outputs to GII, clean up
SurfClust -i $surface -input $input 0 -rmm -$d -n $k -prefix $prefix -out_clusterdset -out_roidset -out_fulllist
ConvertDset -o_gii -input "${prefix}_Clustered_e${d}_n${k}.niml.dset" -prefix "${out_file}.surf.gii"
ConvertDset -o_gii -input "${prefix}_ClstMsk_e${d}_n${k}.niml.dset" -prefix "${out_file}_cls.surf.gii"
rm -f "${prefix}_Cl"*