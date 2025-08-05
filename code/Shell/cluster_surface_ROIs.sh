#!/bin/bash
#-----------------------------------------------------------------------------#
# AFNI surface clustering for paritcipant count maps (fsaverage)
# AFNI <full_path_to_this_script> <sub> <ses> <model> <img> <anat>
# 
#     <full_path_to_this_script>
#             = /data/u_kazuki_software/EMPRISE_2/code/Shell/cluster_surface_ROIs.sh
#     <ses>   - session ID (e.g. "visual")
#     <model> - model name (e.g. "True_False_iid_1")
#     <anat>  - anat prefix in FreeSurfer derivatives folder (e.g. "pial_")
# 
# This script takes a thresholded participant count map (thresholded using
# the subject number) as well as the surface mesh image to which it belongs and
# (i) selects only vertices within a certain range, (ii) forms clusters of
# vertices based on maximum distance and (iii) saves the resulting clusters
# as another GIfTI file.
# 
# The script uses AFNI's SurfClust function and needs to be executed on MPI
# CBS servers with AFNI environment available (i.e. run "AFNI afni" before).
# 
# Anne-Sophie Kieslinger, MPI Leipzig
# 2023-09-08, 14:02: first version ("surfaceclustering.sh")
# Joram Soch, MPI Leipzig <soch@cbs.mpg.de>
# 2023-11-30, 21:24: adapted version ("cluster_surface.sh")
# 2023-11-30, 21:47: this version ("cluster_surface_fsa.sh")
#-----------------------------------------------------------------------------#

# print function call
echo $0 $@

# fail whenever something is fishy
# use -x to get verbose logfiles
set -e -u -x

# get inputs
ses=$1      # session ID
model=$2    # model name
anat=$3		# anat prefix

# set paramters
d=1         # maximum distance between node and cluster, in number of edges
k=100       # minimum size of activated cluster, in number of nodes

# set directories
deri_dir="/data/pt_02495/emprise7t_2/derivatives"
deri_out="/data/pt_02495/emprise7t_2_analysis/derivatives"
res_dir="$deri_out/numprf/sub-all/ses-${ses}/model-${model}"
prefix="$res_dir/SurfClust"

# specify surface (left hemisphere)
surface="$deri_dir/freesurfer/fsaverage/${anat}left.gii"
input="$res_dir/group_cluster_counts_L_thr-3.gii"
out_file="$res_dir/ses-${ses}_model-${model}_hemi-L_counts"

# perform AFNI SurfClust, convert AFNI outputs to GII, clean up
SurfClust -i $surface -input $input 0 -rmm -$d -n $k -prefix $prefix -out_clusterdset -out_roidset -out_fulllist
ConvertDset -o_gii -input "${prefix}_Clustered_e${d}_n${k}.niml.dset" -prefix "${out_file}.surf.gii"
ConvertDset -o_gii -input "${prefix}_ClstMsk_e${d}_n${k}.niml.dset" -prefix "${out_file}_cls.surf.gii"
rm -f "${prefix}_Cl"*

# specify surface (left hemisphere)
surface="$deri_dir/freesurfer/fsaverage/${anat}right.gii"
input="$res_dir/group_cluster_counts_R_thr-3.gii"
out_file="$res_dir/ses-${ses}_model-${model}_hemi-R_counts"

# perform AFNI SurfClust, convert AFNI outputs to GII, clean up
SurfClust -i $surface -input $input 0 -rmm -$d -n $k -prefix $prefix -out_clusterdset -out_roidset -out_fulllist
ConvertDset -o_gii -input "${prefix}_Clustered_e${d}_n${k}.niml.dset" -prefix "${out_file}.surf.gii"
ConvertDset -o_gii -input "${prefix}_ClstMsk_e${d}_n${k}.niml.dset" -prefix "${out_file}_cls.surf.gii"
rm -f "${prefix}_Cl"*