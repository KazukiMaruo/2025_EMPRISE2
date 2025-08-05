#!/bin/bash

# Prepare numerosity population receptive field model results for participant count map.
# Participant count map is a surface visualization on fsaverage that shows
# for every node how many subjects have a numerosity map there.
# For that, we first cluster each individual subject's results map separately.
# This is done to determine which nodes can be said to belong to a topographic map and
# which do not belong to a map or show spurious/noisy activation.
# The clustering is based on the variance explained of the prf model and a var exp threshold, using afni.
# to be executed on cbs servers with AFNI environment available
#-----------------------------------------------------------------------
# CALL LIKE THIS
# AFNI <path to script> <path to prf directory within derivatives, without sub or ses subdirectories> \
# <subject label> <session label> <hemisphere label (left or right)>
#-----------------------------------------------------------------------------------

# Print function call
echo $0 $@

# Fail whenever something is fishy; use -x to get verbose logfiles
set -e -u -x

# get inputs
prf_results_dir=$1
participant=$2
session=$3
hemi=$4

# surface mesh on which the analysis was done and clustering will be done
surface="$prf_results_dir/../../freesurfer/fsaverage/infl_$hemi.gii"

# directory for prf results for this specific subject and session
prf_sub_ses_dir="$prf_results_dir/sub-$participant/ses-$session"

# perform AFNI surface clustering with threshold variance explained=0.3
SurfClust -i $surface -input "$prf_sub_ses_dir/${hemi}_space-fsaverage_rsq.gii" 0 -rmm \
    -2 -thresh 0.3 -prefix "$prf_sub_ses_dir/sub-${participant}_hemi-$hemi" -out_roidset -out_fulllist

# Convert AFNI outputs to gifti
ConvertDset -o_gii -input "$prf_sub_ses_dir/sub-${participant}_hemi-${hemi}_ClstMsk_e2.niml.dset" -prefix "$prf_sub_ses_dir/${hemi}_space-fsaverage_nummaps.gii"

rm -f "$prf_sub_ses_dir/sub-${participant}_hemi-${hemi}_Clst"*
