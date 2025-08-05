# install library
#-------------------------------------------------------------------------#
import os
import numpy as np
import matplotlib.pyplot as plt

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
    model    = "Volumetric" # model name # if you want a spatial smoothing, put 'FWHM' and '3' for the value
    space    = "T1w" # "fsnative" or "T1w"
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


    # specify default covariates
    #-----------------------------------------------------------------------------#
    covs = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', \
            'white_matter', 'csf', 'global_signal', \
            'cosine00', 'cosine01', 'cosine02']

    # get the valid runs
    #-------------------------------------------------------------------------#
    print('\n\n-> Subject "{}", Session "{}":'.format(mod.sub, mod.ses))

    # load confounds
    print('   - Loading confounds ... ', end='')
    X_c, v_run = mod.get_confounds(covs)


    # get the preprocessed volumetric data
    #-------------------------------------------------------------------------#
    # load data
    print('\n-> Space "{}":'.format(mod.space))
    print('   - Loading fMRI data ... ', end='')

    # fMRI data = scans * voxels * runs
    Y     = mod.load_data_all(v_run, mod.space)

    # Load binary brain mask for both cortical and subcortical
    M_cort     = mod.get_volumetric_cortmask(mod.space, v_run)
    M_subcort  = mod.get_volumetric_subcortmask(mod.space, v_run)

    # Mask out the voxels
    Cort     = Y[:,M_cort,:]
    SubCort  = Y[:,M_subcort,:] 

    print('successful!')


    # compute tSNR
    #-------------------------------------------------------------------------#
    # Step 1: Compute tSNR per voxel per run
    # tSNR = mean over time / std over time
    tsnr_cort = np.mean(Cort, axis=0) / np.std(Cort, axis=0)  # shape: (53357, 8)
    tsnr_subcort = np.mean(SubCort, axis=0) / np.std(SubCort, axis=0)  # shape: (53357, 8)

    # Step 2: Average across runs
    tsnr_avg_cort = np.mean(tsnr_cort, axis=1)  # shape: (53357,)
    tsnr_avg_subcort = np.mean(tsnr_subcort, axis=1)  # shape: (53357,)



    # Print out results
    #-------------------------------------------------------------------------#
    mean_cort = np.nanmean(tsnr_avg_cort)
    mean_subcort = np.nanmean(tsnr_avg_subcort)

    print("\n" + "-" * 50)
    print(f"AVERAGE tSNR in cortical: {mean_cort:.2f}")
    print(f"AVERAGE tSNR in subcortical: {mean_subcort:.2f}")
    print("-" * 50 + "\n")

    # visualization
    #-------------------------------------------------------------------------#
    # output dir
    output_dir = f'/data/u_kazuki_software/EMPRISE_2/figures/{subject}/'
    os.makedirs(output_dir, exist_ok=True)

    # define full path
    fig_path = os.path.join(output_dir, f'tSNR_distribution_{session}.png')

    # histogram
    plt.figure(figsize=(8, 5))

    # First histogram
    plt.hist(tsnr_avg_cort, bins=500, color='steelblue', alpha=0.6, label='Cortical')
    # Second histogram
    plt.hist(tsnr_avg_subcort, bins=500, color='darkorange', alpha=0.6, label='Subcortical')
    plt.xlabel("tSNR")
    plt.ylabel("Number of Voxels")
    plt.title("Histogram of tSNR: Cortical vs. Subcortical")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.xlim([0, 200])  # 👈 Set y-axis limit here (adjust as needed)
    plt.ylim([0, 1000])
    plt.savefig(fig_path, dpi=300)
    # plt.yscale("log")
    # plt.show()
