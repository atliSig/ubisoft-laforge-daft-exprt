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


def read_golden_list(path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/golden_lists/mini_v3_same_speaker.txt'):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(line.strip().split('|'))
    return data


def make_comparison(main_dir='./pabc_gold_all', out_dir='./pabc_gold_ordered', num_samples=180):
    golden_data = read_golden_list()
    target_spkr_ids = [d[-2] for d in golden_data]
    exp_names = ['pabc_cleaned_mini', 'pabc_cleaned_mini_v2',
        'pabc_cleaned_mini_v3', 'pabc_cleaned_mini_v4', 'pabc_cleaned_mini_v5']
    exp_shorthands = ['v1', 'v2', 'v3', 'v4', 'v5']
    os.makedirs(out_dir, exist_ok=False)
    for i in range(num_samples):
        sample_out_dir = os.path.join(out_dir, f"sample_{i}")
        os.makedirs(sample_out_dir, exist_ok=False)

        for j, exp_name in enumerate(exp_names):
            exp_sample_dir = os.path.join(main_dir, exp_name, f'utt_{i}')
            # get ref wav
            # get synthesized wav
            # get griffin lim wav
            if j == 0:
                # copy the reference from one of the experiments
                ref_fname = f"ref_{i}.wav"
                copyfile(os.path.join(exp_sample_dir, ref_fname), os.path.join(sample_out_dir, 'ref.wav'))
            # synth fname:
            synth_fname = f'synth_{i}_spk_{target_spkr_ids[i]}_ref_ref_{i}_generated_e2e.wav'
            copyfile(os.path.join(exp_sample_dir, synth_fname), os.path.join(sample_out_dir, f'synth_{exp_shorthands[j]}.wav'))


def make_qualtreats_structure(main_dir='./same_speaker', out_dir='./same_speaker_qualtreats', num_samples=60):
    """
    qualtreats expects the following directory structure:

    /main_dir
        ref_speaker/
            fname_1.wav
            fname_2.wav
            ...
        test_speaker_1/
            ...
        test_speaker_2/
            ...
        ....
    """
    golden_data = read_golden_list()
    target_spkr_ids = [d[-2] for d in golden_data]
    exp_names = ['pabc_cleaned_mini', 'pabc_cleaned_mini_v2',
        'pabc_cleaned_mini_v3', 'pabc_cleaned_mini_v4', 'pabc_cleaned_mini_v5']
    os.makedirs(out_dir, exist_ok=False)
    exp_shorthands = ['v1', 'v2', 'v3', 'v4', 'v5']

    for j, exp_name in enumerate(exp_names):
        os.makedirs(os.path.join(out_dir, exp_name))
        if j == 0:
            os.makedirs(os.path.join(out_dir, 'ground_truth'))
        for i in range(num_samples):
            exp_sample_dir = os.path.join(main_dir, exp_name, f'utt_{i}')
            sample_out_dir = os.path.join(out_dir, exp_name)
            synth_fname = f'synth_{i}_spk_{target_spkr_ids[i]}_ref_ref_{i}_generated_e2e.wav'
            copyfile(os.path.join(exp_sample_dir, synth_fname), os.path.join(sample_out_dir, f'sample_{i}.wav'))
            if j == 0:
                # copy the reference from one of the experiments
                sample_out_dir = os.path.join(out_dir, 'ground_truth')
                ref_fname = f"ref_{i}.wav"
                copyfile(os.path.join(exp_sample_dir, ref_fname), os.path.join(sample_out_dir, f'sample_{i}.wav'))


def pabc_line_to_path(line):
    '''
    the speaker is always index 2 and
    the path is always /home/co-sigu1/rds/hpc-work/pabc_daft_out/<speaker_name>/wavs/<utt_id>.wav
    '''
    items = line.split('|')
    return os.path.join('/home/co-sigu1/rds/hpc-work/pabc_daft_out/', ids2spkrs[int(items[2])], 'wavs', f'{items[1]}.wav')


def spk2paths(path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/all_max200.txt'):
    from collections import defaultdict

    spkr2wavpaths = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            speaker = line.strip().split('|')[-1]
            spkr2wavpaths[speaker].append(pabc_line_to_path(line))
    return spkr2wavpaths

def make_flat_qualtreats_axy_structure(
    list_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/speaker_similarity_list.txt',
    qualtreats_main_dir='./speaker_similarity_qualtreats',
    out_dir='./speaker_similarity_qualtreats_flat',
    num_samples=30):

    import random

    spkr2wavpaths = spk2paths()
    ref_speakers, target_speakers = [], []
    with open(list_path, 'r') as f:
        for line in f:
            _, ref_speaker, _, target_speaker, _ = line.strip().split('|')
            ref_speakers.append(ref_speaker)
            target_speakers.append(target_speaker)

    exp_names = ['pabc_cleaned_mini', 'pabc_cleaned_mini_v2',
        'pabc_cleaned_mini_v3', 'pabc_cleaned_mini_v4', 'pabc_cleaned_mini_v5']

    os.makedirs(out_dir, exist_ok=False)
    synth_dir = os.path.join(out_dir, 'experiment_synths')
    target_dir = os.path.join(out_dir, 'target_speaker')
    ref_dir = os.path.join(out_dir, 'reference_speaker')
    os.makedirs(synth_dir)
    os.makedirs(target_dir)
    os.makedirs(ref_dir)

    counter = 0
    for j, exp_name in enumerate(exp_names):
        for i in range(num_samples):
            exp_sample_dir = os.path.join(qualtreats_main_dir, exp_name)
            copyfile(os.path.join(exp_sample_dir, f'sample_{i}.wav'), os.path.join(synth_dir, f'sample_{counter}.wav'))

            # sample a random utterance from the target speaker
            target_spkr_path = random.choice(spkr2wavpaths[target_speakers[i]])
            copyfile(target_spkr_path, os.path.join(target_dir, f'sample_{counter}.wav'))

            # sample a random utterance from the reference speaker
            ref_spkr_path = random.choice(spkr2wavpaths[ref_speakers[i]])
            copyfile(ref_spkr_path, os.path.join(ref_dir, f'sample_{counter}.wav'))

            counter += 1


def make_qualtreats_axy_structures(
    list_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/speaker_similarity_list.txt',
    qualtreats_main_dir='./speaker_similarity_qualtreats',
    out_dir='./speaker_similarity_qualtreats_ready',
    num_samples=30):

    import random

    spkr2wavpaths = spk2paths()
    ref_speakers, target_speakers = [], []
    with open(list_path, 'r') as f:
        for line in f:
            _, ref_speaker, _, target_speaker, _ = line.strip().split('|')
            ref_speakers.append(ref_speaker)
            target_speakers.append(target_speaker)

    exp_names = ['pabc_cleaned_mini', 'pabc_cleaned_mini_v2',
        'pabc_cleaned_mini_v3', 'pabc_cleaned_mini_v4', 'pabc_cleaned_mini_v5']

    os.makedirs(out_dir, exist_ok=False)

    for j, exp_name in enumerate(exp_names):

        os.makedirs(os.path.join(out_dir, exp_name))
        synth_dir = os.path.join(out_dir, exp_name, 'experiment_synths')
        target_dir = os.path.join(out_dir, exp_name, 'target_speaker')
        ref_dir = os.path.join(out_dir, exp_name, 'reference_speaker')
        os.makedirs(synth_dir)
        os.makedirs(target_dir)
        os.makedirs(ref_dir)

        for i in range(num_samples):
            exp_sample_dir = os.path.join(qualtreats_main_dir, exp_name)
            copyfile(os.path.join(exp_sample_dir, f'sample_{i}.wav'), os.path.join(synth_dir, f'sample_{i}.wav'))

            # sample a random utterance from the target speaker
            target_spkr_path = random.choice(spkr2wavpaths[target_speakers[i]])
            copyfile(target_spkr_path, os.path.join(target_dir, f'sample_{i}.wav'))

            # sample a random utterance from the reference speaker
            ref_spkr_path = random.choice(spkr2wavpaths[ref_speakers[i]])
            copyfile(ref_spkr_path, os.path.join(ref_dir, f'sample_{i}.wav'))


make_qualtreats_structure()