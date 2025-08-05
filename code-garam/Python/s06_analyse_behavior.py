
from argparse import ArgumentParser
from fnmatch import fnmatch
from pathlib import Path
from zipfile import ZipFile
import ast

from bids import BIDSLayout
from datalad.api import Dataset
import os
import pandas as pd
import numpy as np
import re


def make_answer_truth(content):
    """Make answer_truth column based on 'match' column and no button at all being pressed"""
    # this happens in old logs when no button at all was pressed
    conditions = [
        content['match'] == False,
        content['match'] == True
    ]
    choices = [
        'wrong',
        'correct'
    ]
    answer_truth = np.select(conditions, choices, default='NA')
    return answer_truth


def get_trial_response(file_path):

    content = pd.read_csv(file_path)

    # prepare dataframe for dividing into mod_change categories
    old_mod = content.iloc[:-7, content.columns.get_loc('mod')]
    current_mod = content.iloc[7:, content.columns.get_loc('mod')]
    old_mod.index = current_mod.index
    mod_change = old_mod + '_' + current_mod
    # treat mod changes in both directions the same
    mod_change[mod_change == 'audio_digit'] = 'digit_audio'
    mod_change[mod_change == 'audio_dot'] = 'dot_audio'
    mod_change[mod_change == 'dot_digit'] = 'digit_dot'
    # add empty NA to start of df without mod_change yet
    content['mod_change'] = mod_change.reindex(
        index=[i for i in range(mod_change.index[-1]+1)])

    # only some rows have content for match and answer_truth
    if 'answer_truth' in content:
        # 'answer_truth' will have missing values for single trials without button press
        # only one button press per block expected
        # last trial in block will have definite response
        events_mask = content.index[~content['answer_truth'].isna()]
    else:
        # ignore first block as no answer possible
        # always check last trial before pause if response
        events_mask = (content.index[content['mod'] == 'pause']-1)[1:]
        content['answer_truth'] = make_answer_truth(content)
    events = content.loc[events_mask, ['match', 'answer_truth', 'mod_change']]
    # sort trial responses into categories acc. to signal detection theory
    conditions = [
        (events['match'] == False) & (events['answer_truth'] == 'correct'),
        (events['match'] == False) & (events['answer_truth'] == 'wrong'),
        (events['match'] == True) & (events['answer_truth'] == 'correct'),
        (events['match'] == True) & (events['answer_truth'] == 'wrong')]
    choices = [
        'hit',
        'miss',
        'correct_rejection',
        'false_alarm']
    events['response'] = np.select(conditions, choices, default='NA')

    return [sum_counts(mod_change, events, choices) for mod_change in np.unique(events['mod_change'])]


def sum_counts(mod_change, events, choices):
    events = events[events['mod_change'] == mod_change]
    # count category members
    responses, counts = np.unique(events['response'], return_counts=True)
    # prefill with zero for every category
    responses_dict = {resp: 0 for resp in choices}
    for response, count in zip(responses, counts):
        responses_dict[response] = count
    responses_dict['n_events'] = np.sum(list(responses_dict.values()))
    responses_dict['mod_change'] = mod_change
    # return trial categories amounts
    return responses_dict


def main():

    basepath = '/ptmp/akieslinger/bids_3t/sourcedata/'
    basedir = Path(basepath)
    rows = []
    for f in basedir.glob('*/logs/*/*.csv'):
        sub = re.search(r'sub-(.*)_ses', str(f)).group(1)
        ses = re.search(r'ses-([^_]*)_', str(f)).group(1)
        run = re.search(r'run-([^_]*)_', str(f)).group(1)
        print(sub)
        print(ses)
        tmp_dict_list = get_trial_response(str(f))
        for dict_i in tmp_dict_list:
            dict_i['sub'] = sub
            dict_i['ses'] = ses
            dict_i['file'] = f.name
            dict_i['run'] = run
        rows.extend(tmp_dict_list)

    # make pandas dataframe with columns subject,session, hit, false alarm, miss, correct rejection
    output = pd.DataFrame(rows, columns=[
                          'file', 'sub', 'ses', 'run', 'mod_change', 'hit', 'miss', 'correct_rejection', 'false_alarm', 'NA', 'n_events'])
    output = output.sort_values(['sub', 'ses', 'run'])
    directory = os.path.dirname(os.path.abspath(__file__))
    output.to_csv(os.path.join(
        directory, 'response_summary_divided_modchange.csv'), index=False)


if __name__ == '__main__':
    main()
