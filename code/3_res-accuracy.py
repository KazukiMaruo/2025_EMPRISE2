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
data_path       = DATA_DIR
project_path    = PROJECT_DIR
condition       = SESSION
source_path     = os.path.join(data_path, f"sourcedata/{condition}/")
preproc_path    = os.path.join(data_path, "derivatives/fmriprep/")
figure_path     = os.path.join(project_path, "figures/")
behav_path      = os.path.join(project_path, "behavioral/")
# ~~~~~~~~~~ Set directories ~~~~~~~~~~


# ~~~~~~~~~~ Get available subject data in the raw data folder
files = set(f.split("_")[0] for f in os.listdir(behav_path) )
subject_id = sorted(files)
# subject_id = subject_id[5:]
# ~~~~~~~~~~ Get available subject data in the raw data folder ~~~~~~~~~~


# ~~~~~~~~~~ Get all runs file in the subject folder
for sub in subject_id:

    folder = os.path.join(behav_path, sub, condition)
    files  = glob.glob(os.path.join(folder, '*.csv'))  # only .csv files
    # ~~~~~~~~~~ Get all runs file in the subject folder ~~~~~~~~~~

    # ~~~~~~~~~~ Sort the files for each run
    def get_run_number(filename):
        # Extract the number after 'run-' using regex
        match = re.search(r'run-(\d+)', filename)
        return int(match.group(1)) if match else -1

    sorted_files = sorted(files, key=get_run_number)

    for f in sorted_files:
        print(f)


    # ~~~~~~~~~~ read a csv file (each run)
    for run, csv_file in enumerate(sorted_files):
        df = pd.read_csv(csv_file)

        if 'match' in df.columns:
            # ~~~~~~~~~~ filter out
            df_f = df[df['colors']=='white']
            # ~~~~~~~~~~ count the number of attention trial
            atten_trials = df_f.shape[0]
            answer = df_f['match']
            answer_percent = answer.mean() *100
        else:
            print("Column 'match' not found.")
            # Find indices where color == 'white'
            white_idx = df.index[df['color'] == 'white']

            after_idx = white_idx + 1
            after_idx = after_idx[after_idx < len(df)]

            # Get rows immediately after those
            after_white_rows = df.loc[after_idx]  # shift index by 1

            # count totoal number of rows
            atten_trials = len(after_white_rows)

            # Count number of rows where button is '1'
            # Handles both numeric and string "1"
            button_is_1 = (after_white_rows['button'] == 1).sum()

            answer_percent = (button_is_1 / atten_trials) * 100  
        # ~~~~~~~~~~ count the number of attention trial ~~~~~~~~~~ 


        # ~~~~~~~~~~ print out the percentage of accuaracy
        print(f"Subject: {sub}")
        print(f"Trials: {atten_trials}")
        run = run+1
        print(f"Run: {run}")
        if answer_percent is not None:
            print(f"accuracy: {answer_percent:.2f}")
        else:
            print("accuracy: N/A")

        
