
import os
from shutil import copyfile

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

ids2spkrs = {speaker_ids[i]: speakers[i] for i in range(len(speakers))}
spkrs2ids = {j:i for i, j in ids2spkrs.items()}


def pabc_line_to_path(line):
    '''
    the speaker is always index 2 and
    the path is always /home/co-sigu1/rds/hpc-work/pabc_daft_out/<speaker_name>/wavs/<utt_id>.wav
    '''
    items = line.split('|')
    return os.path.join('/home/co-sigu1/rds/hpc-work/pabc_daft_out/', ids2spkrs[int(items[2])], 'wavs', f'{items[1]}.wav')


def copy_from_utt_list(utt_list_path:str='./ordered.txt', out_dir='./ordered', iterate_name=True):
    os.makedirs(out_dir, exist_ok=False)
    in_lines = []
    with open(utt_list_path, 'r') as f:
        for line in f:
            in_lines.append(line.strip())
    for i, line in enumerate(in_lines):
        wav_path = pabc_line_to_path(line)
        if iterate_name:
            fname = f'utt_{i}.wav'
        else:
            fname = os.path.split(wav_path)[1]
        target_path = os.path.join(out_dir, fname)
        copyfile(wav_path, target_path)


copy_from_utt_list()