import os
import json
import multiprocessing as mp
import itertools
import math

import numpy as np


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


def get_stats(stats_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/cleaned_pabc/stats.json'):
    with open(stats_path) as f:
        data = f.read()
        stats = json.loads(data)

    out = {}
    for i in range(num_speakers):
        out[speaker_ids[i]] = stats[f"spk {speaker_ids[i]}"]['pitch']
    return out

def get_lines(line_path:str='./text_results/all_max200_evaluation.txt'):
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
    def get_pitch_path(utt_id, spkr_id):
        return os.path.join(FEATURE_BASE, ids2spkr[spkr_id], f'{utt_id}.symbols_f0')

    with open(get_pitch_path(utt_id, spkr_id), 'r', encoding='utf-8') as f:
            lines = f.readlines()
    pitch = np.array([float(line.strip()) for line in lines])
    if normalize:
        zero_idxs = np.where(pitch == 0.)[0]
        pitch = (pitch - stats[spkr_id]['mean']) / stats[spkr_id]['std']
        pitch[zero_idxs] = 0.
    return pitch


def load_pitch_mapped(args):
    return load_pitch(args[0], args[1])

stats = get_stats()
lines = get_lines()
pool = mp.Pool(mp.cpu_count())
pitches = pool.map(load_pitch_mapped, lines)
pool.close()

def pitches_to_var(pitches):
    var = []
    for p in pitches:
        var.append(np.var(p))
    order = np.argsort(var)[::-1]
    return order

def line_to_text_line(line):
    utt_id, spkr_id = line[0], line[1]
    feature_dir = os.path.join(FEATURE_BASE, ids2spkr[spkr_id])
    return f'{feature_dir}|{utt_id}|{spkr_id}'

order = pitches_to_var(pitches)
for idx in order:
    print(line_to_text_line(lines[idx]))