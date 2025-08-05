# install library
#-------------------------------------------------------------------------#
import os


# Parameters
#-------------------------------------------------------------------------#
"""
Modify here only for your analysis
condition: digit, spoken, visual, audio1, audio2
code_dir: specify the directory containing "config.json" file.
"""
# subject_lists = ["N001", "N003", "N004", "N005","N006", "N007","N008", "N009","N011", "N012"]
# for subject in subject_lists:
subject  = "D005"
session  = "visual"
model    = "NumAna" # model name # if you want a spatial smoothing, put 'FWHM' and '3' for the value
space    = "fsaverage" # "fsnative" or "T1w"
target   = "all" # "all" or "split"
code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"


# Install functions
#-------------------------------------------------------------------------#
import EMPRISE
# import warnings
# warnings.filterwarnings("ignore")

# start Session class
#-------------------------------------------------------------------------#
sess = EMPRISE.Session(subject, session)
print(sess.get_bold_gii(1, space=space)) # example usage


# start Model class
#-------------------------------------------------------------------------#
mod = EMPRISE.Model(subject, session, model, space_id=space)
print(mod.get_model_dir()) # example usage


#
# Uncomment analysis you want to implement
#-------------------------------------------------------------------------#
"""
        1. "analyze_numerosity": Estimate Numerosities and FWHMs for Surface-Based Data
        2. "calculate_Rsq": Calculate R-Squared Maps for Numerosity Model
        3. "threshold_maps": Threshold Numerosity, FWHM and Scaling Maps based on Criterion: Bonferoni correction on p = .05
        4. "": 
"""



# analyze numerosities
#-------------------------------------------------------------------------#
if   target == "all":

        #------------ analyze numerosities in surface-based
        mod.analyze_numerosity(avg=[True, False], corr='iid', order=1, ver='V2', stim = False, hemis = ['L','R'], sh=False) # all runs
        # analyze numerosities in volumetric-based
        # mod.analyze_numerosity_volumetric(avg=[True, False], corr='iid', order=1, ver='V2', stim = False, sh=False) # all runs

        #------------  calculate R-squared in surface-based
        mod.calculate_Rsq(folds=['all'], stim = False) # all runs
        # calculate R-squared in volumetric space
        # mod.calculate_Rsq_volumetric(folds=['all'], stim = False) # all runs

        #------------  generate paramter's map in volumetric-based
        # mod.parameter_map_volumetric()

        #------------  threshold tuning maps in surface-based
        mod.threshold_maps(crit='Rsqmb,p=0.05B', cv=False)# all runs
        # threshold tuning maps in volumetric-based
        # mod.threshold_maps_subco(crit='Rsqmb,p=0.05B', cv=False)# all runs

elif target == "split":
        # analyze numerosities
        model.analyze_numerosity(avg=[True, False], corr='iid', order=1, ver='V2', stim = False, hemis = ['L','R'], sh=True) # even and odd runs
        # cross-validated R-squared
        model.calculate_Rsq(folds=['cv'], stim = False)# cross-validated
        # function: threshold tuning maps
        model.threshold_maps(crit='Rsqmb,p=0.05B', cv=True)# cross-validated
else:
        raise ValueError(f"Invalid target '{target}'. Expected 'all' or 'split'.")
