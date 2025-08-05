# install library
#-------------------------------------------------------------------------#
import os
import numpy as np
import matplotlib.pyplot as plt

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

# Parameters
#-------------------------------------------------------------------------#
"""
Modify here only for your analysis
condition: digit, spoken, visual, audio1, audio2
code_dir: specify the directory containing "config.json" file.
"""
subject_lists = ["N001", "N003", "N004", "N005","N006", "N007","N008", "N009","N011", "N012"]
dyscal_lists  = ["D001", "D002","D003","D004","D005"]
for subject in dyscal_lists:
    # subject  = "N001"
    session  = "visual"
    model    = "NumAna" # model name # if you want a spatial smoothing, put 'FWHM' and '3' for the value
    space    = "fsnative" # "fsnative" or "T1w"
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


    # load data
    #-------------------------------------------------------------------------#
    hemis = ['L', 'R']
    all_sd = []

    for hemi in hemis:
        print('\n-> Hemisphere "{}", Space "{}":'.format(hemi, mod.space))
        print('   - Loading fMRI data ... ', end='')
        Y, M  = mod.load_mask_data(hemi)
        Y     = standardize_signals(Y)
        V     = M.size
        print('successful!')


        # compute std of time course across 8 runs
        #-------------------------------------------------------------------------#
        # variance devide by the number of runs
        dvar = np.var(Y, axis=2, ddof=0)
        dvar_mean = np.mean(dvar, axis=0) 

        # collect results
        all_sd.append(dvar_mean)

    # concatanate SDs from both hemisphere
    all_sd_combined = np.concatenate(all_sd)

    mean = all_sd_combined.mean()

    print(f'average Variance: {mean:.2f}')

    # visualization
    #-------------------------------------------------------------------------#

    # output dir
    output_dir = f'/data/u_kazuki_software/EMPRISE_2/figures/{subject}/'
    os.makedirs(output_dir, exist_ok=True)

    # define full path
    fig_path = os.path.join(output_dir, f'variance_distribution_{session}.png')

    # histogram
    plt.figure(figsize=(8, 5))
    plt.hist(all_sd_combined, bins=30000, color='steelblue')
    plt.xlabel("Variance of Time Course")
    plt.ylabel("Number of Voxels")
    plt.title("Histogram of Voxelwise Time Course Variability")
    plt.tight_layout()
    plt.grid(True)
    plt.xlim([0, 20])  # 👈 Set y-axis limit here (adjust as needed)
    plt.ylim([1, 20000])
    plt.savefig(fig_path, dpi=300)
    # plt.yscale("log")
    # plt.show()