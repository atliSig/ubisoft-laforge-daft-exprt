import os
import json
import multiprocessing as mp
import itertools
import math

import numpy as np
from tqdm import tqdm
from dtw import *


speakers = [
    "nfo",
    "acr",
    "mfo",
    "ges",
    "ael",
    "gma",
    "rgo",
    "mwa",
    "bsa",
    "tbo",
    "apr",
    "iva",
    "law",
    "kmq",
    "hmc",
    "jaw",
    "pch",
    "msm",
    "mbr",
    "jms",
    "cmi",
    "kti",
    "rph",
    "kfe",
    "gsi",
    "kmc",
    "mth",
    "ksh",
    "dcl",
    "mjd",
    "afo",
    "bga",
    "tka",
    "ber",
    "cve",
    "jac",
    "wro",
    "jgr",
    "ksh_2",
    "jja",
    "rzd",
    "zbg",
    "sde",
    "pyo",
    "mdw",
    "rco",
    "jsu",
    "ekl",
    "trg",
    "scr",
    "jlo",
    "cco",
    "csa"]
speaker_ids = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52]
num_speakers = len(speakers)
spkr2ids = {speakers[i]: speaker_ids[i] for i in range(len(speakers))}
ids2spkr = {speaker_ids[i]: speakers[i] for i in range(len(speakers))}

FEATURE_BASE = '/home/co-sigu1/rds/hpc-work/pabc_daft_prepro'


def get_stats(stats_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/pabc/stats.json'):
    with open(stats_path) as f:
        data = f.read()
        stats = json.loads(data)

    out = {}
    for i in range(num_speakers):
        out[speaker_ids[i]] = stats[f"spk {speaker_ids[i]}"]['pitch']
    return out

def get_lines(line_path:str='./all_max200.txt'):
    """
    each line e.g. :
    /home/co-sigu1/rds/hpc-work/pabc_daft_prepro/nfo|treasure_chp11_utt002|0
    """
    lines = []
    with open(line_path, 'r') as f:
        for line in f:
            _, utt_id, spkr_id = line.strip().split('|')
            spkr_id = int(spkr_id)
            lines.append([utt_id, spkr_id])
    return lines

def load_pitch(utt_id, spkr_id, normalize=True):
    with open(get_pitch_path(utt_id, spkr_id), 'r', encoding='utf-8') as f:
            lines = f.readlines()
    pitch = np.array([float(line.strip()) for line in lines])
    if normalize:
        zero_idxs = np.where(pitch == 0.)[0]
        pitch = (pitch - stats[spkr_id]['mean']) / stats[spkr_id]['std']
        pitch[zero_idxs] = 0.
    return pitch


def get_pitch_path(utt_id, spkr_id):
    return os.path.join(FEATURE_BASE, ids2spkr[spkr_id], f'{utt_id}.symbols_f0')


def line_to_text_line(line):
    utt_id, spkr_id = line[0], line[1]
    feature_dir = os.path.join(FEATURE_BASE, ids2spkr[spkr_id])
    return f'{feature_dir}|{utt_id}|{spkr_id}'


def write_lines(lines, out_path:str='dtw_results_all.txt'):
    with open(out_path, 'w') as f:
        for pair in lines:
            if pair[0] is not None:
                dist = pair[0]
                l_1 = pair[1]
                l_2 = pair[2]
                text_line = f'{dist}|{line_to_text_line(l_1)}|{line_to_text_line(l_2)}\n'
                f.write(text_line)


def print_line(pair):
    dist = pair[0]
    l_1 = pair[1]
    l_2 = pair[2]
    text_line = f'{dist}|{line_to_text_line(l_1)}|{line_to_text_line(l_2)}'
    print(text_line)

def load_pitch_mapped(args):
    return load_pitch(args[0], args[1])

"""
stats = get_stats()
lines = get_lines()
num_lines = len(lines)

print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
pitches = pool.map(load_pitch_mapped, lines)
pool.close()
"""

def mapped(idx_1, idx_2):
        """
        i x j (row, col)

        i = idx % num_lines
        j = idx - i*num_lines
        """
        if idx_1 == idx_2 or np.abs(len(pitches[idx_1]) - len(pitches[idx_2])) > 0.15*np.min([len(pitches[idx_1]),len(pitches[idx_2])]):
            return 1000
        return dtw(pitches[idx_1], pitches[idx_2]).distance


def plot_distance(in_path:str='./dtw_results_all.txt'):
    import matplotlib.pyplot as plt
    distances = []
    num_spkr_match, num_utt_match = 0, 0
    with open(in_path, 'r') as f:
        for line in f:
            (distance, _, target_utt_id, target_spkr_id, _, ref_utt_id, ref_spkr_id) = line.strip().split('|')
            if target_spkr_id == ref_spkr_id:
                num_spkr_match += 1
            if target_utt_id == ref_utt_id:
                num_utt_match += 1
            distances.append(float(distance))
    print(np.mean(distances))
    print(np.std(distances))
    print(num_spkr_match, num_utt_match)
    plt.hist(distances, bins=100)
    plt.title('DTW distance distribution')
    plt.ylabel("Frequency")
    plt.xlabel("DTW Distance")
    plt.savefig('distance_distribution.png')


def only_2stds(in_path:str='./dtw_results_all.txt', out_path:str='./dtw_results_all_1std.txt'):
    """
    generate results for utterances only 2stds away
    from the mean
    """
    import matplotlib.pyplot as plt
    distances = []
    lines = []
    num_spkr_match, num_utt_match = 0, 0
    with open(in_path, 'r') as f:
        for line in f:
            lines.append(line)
            (distance, _, target_utt_id, target_spkr_id, _, ref_utt_id, ref_spkr_id) = line.strip().split('|')
            distances.append(float(distance))
    mean = np.mean(distances)
    std = np.std(distances)
    out_dists = []
    out_lines = []
    for i in range(len(distances)):
        if np.abs(distances[i]-mean) < std:
            out_dists.append(distances[i])
            (distance, _, target_utt_id, target_spkr_id, _, ref_utt_id, ref_spkr_id) = lines[i].strip().split('|')
            out_lines.append(lines[i])
            if target_spkr_id == ref_spkr_id:
                num_spkr_match += 1
            if target_utt_id == ref_utt_id:
                num_utt_match += 1
    print(np.mean(out_dists))
    print(np.std(out_dists))
    print(num_spkr_match, num_utt_match)
    plt.hist(out_dists, bins=100)
    plt.title('DTW distance distribution on selected samples')
    plt.ylabel("Frequency")
    plt.xlabel("DTW Distance")
    plt.savefig('distance_distribution_1std.png')
    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(line)

def dtw_list_to_train_val(in_path, out_dir, num_val):
    import random
    in_lines = []
    with open(in_path, 'r') as f:
        for line in f:
            vals = line.split('|')
            wanted = '|'.join(vals[1:])
            in_lines.append(wanted)
    random.shuffle(in_lines)
    with open(os.path.join(out_dir, 'validation_english.txt'), 'w') as f:
        for i in range(0, num_val):
            f.write(in_lines[i])
    with open(os.path.join(out_dir, 'train_english.txt'), 'w') as f:
        for i in range(num_val, len(in_lines)):
            f.write(in_lines[i])


if __name__ == "__main__":
    #only_2stds()
    dtw_list_to_train_val(
        '/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/dtw_results_all_1std.txt',
        '/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/pabc_diff_speakers_v5',
        100)
    '''
    from itertools import repeat
    print(mp.cpu_count())
    print('starting ................')

    pool = mp.Pool(mp.cpu_count())
    out_lines = []
    for i in range(len(lines)):
        values = list(pool.starmap(mapped, zip(repeat(i), range(0, len(lines)))))
        best_ind = np.argmin(values)
        out_lines.append([values[best_ind], lines[i], lines[best_ind]])
        print_line(out_lines[-1])

    pool.close()

    write_lines(out_lines)
    '''