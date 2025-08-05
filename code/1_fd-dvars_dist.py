
# ~~~~~~~~~~ Function to get FD and Plot Distribution
def get_fd_and_plot(sub, preproc_path, figure_path, condition):
    """
    Get framwise displacement values and plot the density map acorss all 8 runs.

        sub             -string; subject name (e.g. "N001")
        preproc_path    -string; preprocessed folder (e.g. ".../derivatives/fmriprep/")
        figure_path     -string; folder for saving figures (e.g. ".../EMPRISE_2/figures/")
        condition       -string; experiment condition (e.g. "digit")
    """
    # get a subject folder name
    sub_folder = glob.glob(f"{preproc_path}/*{sub}*/ses-{condition}/func")
    if not sub_folder:
        print(f"No preprocessed folder found for subject: {sub}, skipping...")
        return
    else:
        print(f"Preprocessed folder found for subject: {sub}")
        sub_folder = sub_folder[0]  # convert list to string

    # Define figure save folder
    figure_subfolder = os.path.join(figure_path, sub)
    # Skip if the folder already exists
    if os.path.exists(figure_subfolder):
        print(f"Figure folder already exists for subject: {sub}, skipping re-creation...")
    else:
        os.makedirs(figure_subfolder, exist_ok=True)
        print(f"Folder created: {figure_subfolder}")
    # get .tsv file for all runs
    tar_filenames = [
        f for f in os.listdir(sub_folder) 
        if "NORDIC" in f and f.endswith("desc-confounds_timeseries.tsv")
    ]

    # sort the list of files by run number
    sorted_files = sorted(
        tar_filenames, 
        key=lambda f: int(re.search(r"run-(\d+)", f).group(1)), 
        reverse=False
    )
    print(f"The number of files or runs: {len(sorted_files)}")
    print(f"These files will be analyzed: {sorted_files}")

    """     
    # Create a DataFrame containing data from all runs
    df_list = []
    for file in sorted_files:
        df = pd.read_csv(os.path.join(sub_folder, file), sep="\t")  # read tsv file
        df["Run"] = re.search(r"run-(\d+)", file).group(1)
        df_list.append(df)

    # Concatenate all DataFrames into one
    final_df = pd.concat(df_list, ignore_index=True) 
    """

    # Create a dataframe
    for run, file in enumerate(sorted_files):
        run = run+1
        final_df = pd.read_csv(os.path.join(sub_folder, file), sep="\t")
        # Calculate the average and percentage of FD values greater than 0.5
        average = final_df["framewise_displacement"].dropna().mean()
        percentage = (final_df["framewise_displacement"].dropna() > 0.5).mean() * 100
        print(f"Average of FD values (Run_{run}): {average:.2f}")
        print(f"Percentage of FD values > 0.5 (Run_{run}): {percentage:.2f}%")

        # Plot distribution
        sns.kdeplot(final_df["framewise_displacement"].dropna(), fill=True, common_norm=False)
        plt.xlabel("FD values (mm)")
        plt.ylabel("Density")
        
        # Set x-axis limits
        plt.xlim(0, 1)
        
        # Set x-axis step size (e.g., 0.1 intervals)
        plt.xticks(np.arange(0, 1, 0.1))
        
        # Remove y-axis ticks
        plt.yticks([])
        
        # Save the figure
        save_path = os.path.join(figure_subfolder, f"fd_distribution_{condition}_{run}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save as PNG with high quality
        plt.show()
# ~~~~~~~~~~ Function to get FD and Plot Distribution ~~~~~~~~~~


# ~~~~~~~~~~ Function to get dvars and Plot Distribution
def get_dvars_and_plot(sub, preproc_path, figure_path, condition):
    """
    Get standardized dvars values and plot the density map acorss all 8 runs.

        sub             -string; subject name (e.g. "N001")
        preproc_path    -string; preprocessed folder (e.g. ".../derivatives/fmriprep/")
        figure_path     -string; folder for saving figures (e.g. ".../EMPRISE_2/figures/")
        condition       -string; experiment condition (e.g. "digit")
    """
    # get a subject folder name
    sub_folder = glob.glob(f"{preproc_path}/*{sub}*/ses-{condition}/func")
    if not sub_folder:
        print(f"No preprocessed folder found for subject: {sub}, skipping...")
        return
    else:
        print(f"Preprocessed folder found for subject: {sub}")
        sub_folder = sub_folder[0]  # convert list to string

    # Define figure save folder
    figure_subfolder = os.path.join(figure_path, sub)
    # Skip if the folder already exists
    if os.path.exists(figure_subfolder):
        print(f"Figure folder already exists for subject: {sub}, skipping re-creation...")
    else:
        os.makedirs(figure_subfolder, exist_ok=True)
        print(f"Folder created: {figure_subfolder}")

    # get .tsv file for all runs
    tar_filenames = [
        f for f in os.listdir(sub_folder) 
        if "NORDIC" in f and f.endswith("desc-confounds_timeseries.tsv")
    ]

    # sort the list of files by run number
    sorted_files = sorted(
        tar_filenames, 
        key=lambda f: int(re.search(r"run-(\d+)", f).group(1)), 
        reverse=False
    )
    print(f"The number of files or runs: {len(sorted_files)}")
    print(f"These files will be analyzed: {sorted_files}")

    """

    # Create a DataFrame containing data from all runs
    df_list = []
    for file in sorted_files:
        df = pd.read_csv(os.path.join(sub_folder, file), sep="\t")  # read tsv file
        df["Run"] = re.search(r"run-(\d+)", file).group(1)
        df_list.append(df)

    # Concatenate all DataFrames into one
    final_df = pd.concat(df_list, ignore_index=True)
    """

    for run, file in enumerate(sorted_files):
        run = run+1
        final_df = pd.read_csv(os.path.join(sub_folder, file), sep="\t")
     
        # Calculate the average and percentage of FD values greater than 0.5
        average = final_df["std_dvars"].dropna().mean()
        percentage = (final_df["std_dvars"].dropna() > 1.5).mean() * 100
        print(f"Average of dvars values (Run_{run}): {average:.2f}")
        print(f"Percentage of dvars values > 1.5 (Run_{run}): {percentage:.2f}%")

        # Plot distribution
        sns.kdeplot(final_df["std_dvars"].dropna(), fill=True, common_norm=False)
        plt.xlabel("DVARS values")
        plt.ylabel("Density")
        
        # Set x-axis limits
        plt.xlim(0, 3)
        
        # Set x-axis step size (e.g., 0.1 intervals)
        plt.xticks(np.arange(0, 3, 0.5))
        
        # Remove y-axis ticks
        plt.yticks([])
        
        # Save the figure
        save_path = os.path.join(figure_subfolder, f"dvars_distribution_{condition}_{run}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # Save as PNG with high quality
        plt.show()
# ~~~~~~~~~~ Function to get dvars and Plot Distribution ~~~~~~~~~~


# ~~~~~~~~~~ library install
import os
import glob
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
# ~~~~~~~~~~ library install ~~~~~~~~~~


# ~~~~~~~~~~ Set directories
"""
Modify here only for your analysis
condition: digit, spoken, visual, audio1, audio2
code_dir: specify the directory containing "config.json" file.
"""
code_dir = "/data/u_kazuki_software/EMPRISE_2/code/"

config_file = os.path.join(code_dir, "config.json")
with open(config_file) as f:
    config = json.load(f)
globals().update(config)

# path specification
data_path = DATA_DIR
project_path = PROJECT_DIR
condition = SESSION
source_path = os.path.join(data_path, f"sourcedata/{condition}/")
preproc_path = os.path.join(data_path, "derivatives/fmriprep/")
figure_path = os.path.join(project_path, "figures/")
# ~~~~~~~~~~ Set directories ~~~~~~~~~~


# ~~~~~~~~~~ Get available subject data in the raw data folder
files = set(f.split("_")[0] for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path,f)))
subject_id = sorted(files)
# ~~~~~~~~~~ Get available subject data in the raw data folder ~~~~~~~~~~


# ~~~~~~~~~~ Loop Over Subjects and Call FD and dvars Function
for sub in subject_id:
    get_fd_and_plot(sub, preproc_path, figure_path, condition)
    get_dvars_and_plot(sub, preproc_path, figure_path, condition)
# ~~~~~~~~~~ Loop Over Subjects and Call FD Function ~~~~~~~~~~
