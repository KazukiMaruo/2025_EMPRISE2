# install library
#-------------------------------------------------------------------------#
import os
import re

# Parameters
#-------------------------------------------------------------------------#
"""
Modify parameters for your analysis in config.json and Figures.py
code_dir: specify the directory containing "config.json" file.
"""
code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"

# # Install functions
#-------------------------------------------------------------------------#
import Figures

Figures.WP1_Fig1(Figure='S2')
Figures.WP1_Fig2(Figure='2A')


# # # start Model class
# import EMPRISE
# # #-------------------------------------------------------------------------#
# model   = EMPRISE.Model(subject, session, model, space_id='fsnative')
# labels  = EMPRISE.covs
# X_c, valid_runs     = model.get_confounds(labels)
# X_c = X_c[1:,:,:]
# print(X_c.shape[0])
