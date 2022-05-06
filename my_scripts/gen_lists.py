import os

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

def get_utt_id_to_text(path='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/utt_id_to_text.txt'):
    out = {}
    with open(path, 'r') as f:
        for line in f:
            utt_id, text = line.strip().split("|")
            out[utt_id] = text
    return out

utt_id_to_text = get_utt_id_to_text()
text_sub_150 = [u for u in utt_id_to_text.values() if len(u) < 75]
text_sub_40 = [u for u in utt_id_to_text.values() if len(u) < 75]

def random_sample_text(num):
    import random
    return random.sample(text_sub_150, num)


def random_small_sample_text(num):
    import random
    return random.sample(text_sub_150, num)



def gen_tiny_list():
    def read_pabc_list(path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/pabc_diff_speakers/train_english.txt'):
        from collections import defaultdict
        data = defaultdict(list)
        with open(path, 'r') as f:
            for line in f:
                spkr = line.split('|')[2]
                data[spkr].append(line)
        return data

    def write_tiny_list(data, num_per_speaker:int=5, out_path='./tiny_list.txt'):
        import random
        with open(out_path, 'w') as f:
            for _, utts in data.items():
                lines = random.sample(utts, num_per_speaker)
                for l in lines:
                    f.write(l)
    '''
    if __name__ == '__main__':
        d = read_pabc_list()
        write_tiny_list(d)
    '''

def gen_ravdess_list(ravdess_path:str="/home/co-sigu1/rds/hpc-work/ravdess", out_path:str='./ravdess.txt',
                     chosen_emotions=range(1,9), chosen_statements=[0,1], num_speakers=24):
    """
    Get 1 emotion for each statement from each speaker
    and write paths to a list.
    """
    all_paths = []
    statements = ["Kids are talking by the door", "Dogs are sitting by the door"]
    for speaker in os.listdir(ravdess_path)[:num_speakers]:
        # each utt is on the form *-*-emotion-intensity-statement-repition-actor
        # so emation_idx = 2
        #    intense_idx = 3
        #    statmnt_idx = 4
        actor_id = speaker.split('_')[1]
        for statement_idx in chosen_statements:
            for emotion_idx in chosen_emotions:
                fname = f'03-01-0{emotion_idx}-0{2}-0{statement_idx+1}-01-{actor_id}.wav'
                path = os.path.join(ravdess_path, speaker, fname)
                if not os.path.isfile(path):
                    # back down to 01 intensity (neutral)
                    fname = f'03-01-0{emotion_idx}-0{1}-0{statement_idx+1}-01-{actor_id}.wav'
                    path = os.path.join(ravdess_path, speaker, fname)
                if not os.path.isfile(path):
                    print(f'Path {path} was not found, skipping')
                else:
                    all_paths.append(path)

    with open(out_path, 'w') as f:
        for path in all_paths:
            f.write(f'{path}\n')


def gen_ravdess_validation_list(ravdess_path="/home/co-sigu1/rds/hpc-work/ravdess", curr_list='./ravdess.txt', out_list='./ravdess_validation.txt'):
    curr_set = set()
    all_paths = []
    with open(curr_list, 'r') as f:
        for line in f:
            curr_set.add(line.strip())

    for speaker in os.listdir(ravdess_path):
        for fname in os.listdir(os.path.join(ravdess_path, speaker)):
            path = os.path.join(ravdess_path, speaker, fname)
            if path not in curr_set:
                all_paths.append(path)

    with open(out_list, 'w') as f:
        for path in all_paths:
            f.write(f'{path}\n')


def gen_synth_list_from_val_list(val_list_path:str, out_path:str):
    """
    Pick 1 validation sample from each speaker

    format is either:
        /home/co-sigu1/rds/hpc-work/pabc_daft_prepro/nfo|treasure_chp24_utt079|0
    or
        /home/co-sigu1/rds/hpc-work/pabc_daft_prepro/mfo|emma_chp52_utt100|2|/home/co-sigu1/rds/hpc-work/pabc_daft_prepro/mth|emma_chp52_utt100|26

    the speaker is always index 2 and
    the path is always /home/co-sigu1/rds/hpc-work/pabc_daft_out/<speaker_name>/wavs/<utt_id>.wav
    """

    utt_base = '/home/co-sigu1/rds/hpc-work/pabc_daft_out/'
    utts = []

    with open(val_list_path, 'r') as f:
        for line in f:
            vals = line.strip().split('|')
            utt_id = vals[1]
            spkr_id = int(vals[2])
            spkr_name = ids2spkrs[spkr_id]

            utts.append(os.path.join(utt_base, spkr_name ,'wavs', f'{utt_id}.wav'))
            assert os.path.isfile(utts[-1]), f"file {utts[-1]} not found"
    with open(out_path, 'w') as f:
        for utt in utts:
            f.write(f'{utt}\n')


def gen_validation_list(
    in_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/all_max200.txt',
    out_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/all_max200_evaluation.txt',
    num:int=200):

    import random
    lines = []
    with open(in_path, 'r') as f:
        for line in f:
            lines.append(line.strip())
    random.shuffle(lines)
    with open(out_path, 'w') as f:
        for i in range(num):
            f.write(f'{lines[i]}\n')

def remove_validation_from_list(
    validation_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/all_max200_evaluation.txt',
    in_path='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/pabc_diff_speakers_v5/train_english.txt',
    out_path='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/pabc_diff_speakers_v5/train_english_cleaned.txt',
    using_ref:bool=False):

    from collections import defaultdict
    val_samples = defaultdict(list)

    with open(validation_path, 'r') as f:
        for line in f:
            path, utt_id, spkr_id = line.strip().split('|')
            val_samples[spkr_id].append(utt_id)

    out_lines = []
    with open(in_path, 'r') as f:
        for line in f:
            if using_ref:
                path, utt_id, spkr_id, ref_path, ref_utt_id, ref_spkr_id = line.strip().split('|')
                if utt_id not in val_samples[spkr_id] and ref_utt_id not in val_samples[ref_spkr_id]:
                    out_lines.append(line)
            else:
                path, utt_id, spkr_id = line.strip().split('|')
                if utt_id not in val_samples[spkr_id]:
                    out_lines.append(line)

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(line)

def gen_hifi_lists(
    in_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/all_max200.txt',
    validation_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/all_max200_evaluation.txt',
    out_train_path='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/hifi_pabc_train.txt',
    out_validation_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/hifi_pabc_validation.txt',
    base_out_path:str='/home/co-sigu1/rds/hpc-work/ft_datasets/pabc_v1/'
):
    # /home/co-sigu1/rds/hpc-work/pabc_daft_prepro/scr|emma_chp18_utt014|49
    # path, utt_id, spkr_id

    from collections import defaultdict
    val_samples = defaultdict(list)
    val_lines = []

    with open(validation_path, 'r') as f:
        for line in f:
            path, utt_id, spkr_id = line.strip().split('|')
            val_samples[spkr_id].append(utt_id)
            val_lines.append(line.strip())

    train_lines = []
    with open(in_path, 'r') as f:
        for line in f:
            path, utt_id, spkr_id = line.strip().split('|')
            if utt_id not in val_samples[spkr_id]:
                train_lines.append(line.strip())

    with open(out_validation_path, 'w') as f:
        for line in val_lines:
            _, utt_id, spkr_id = line.split('|')
            path = os.path.join(base_out_path, ids2spkrs[int(spkr_id)], f'{utt_id}.wav')
            f.write(f'{path}\n')

    with open(out_train_path, 'w') as f:
        for line in train_lines:
            _, utt_id, spkr_id = line.split('|')
            path = os.path.join(base_out_path, ids2spkrs[int(spkr_id)], f'{utt_id}.wav')
            f.write(f'{path}\n')


def gen_random_pairs(
    in_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/all_max200.txt',
    validation_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/all_max200_evaluation.txt',
    out_train_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/cleaned_pabc_v2/train_english.txt',
    out_validation_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/cleaned_pabc_v2/validation_english.txt'):

    from collections import defaultdict
    val_samples = defaultdict(list)

    with open(validation_path, 'r') as f:
        for line in f:
            path, utt_id, spkr_id = line.strip().split('|')
            val_samples[spkr_id].append(utt_id)

    import random
    lines = []
    with open(in_path, 'r') as f:
        for line in f:
            path, utt_id, spkr_id = line.strip().split('|')
            if utt_id not in val_samples[spkr_id]:
                lines.append(line.strip())
    random.shuffle(lines)
    with open(out_validation_path, 'w') as f:
        for i in range(1, 100):
            l_1 = lines[i-1]
            l_2 = lines[i]
            f.write(f'{l_1}|{l_2}\n')

    with open(out_train_path, 'w') as f:
        for i in range(100, len(lines)):
            l_1 = lines[i-1]
            l_2 = lines[i]
            f.write(f'{l_1}|{l_2}\n')


def ravdess_path_to_text_and_speaker(path):
    # /home/co-sigu1/rds/hpc-work/ravdess/Actor_06/03-01-06-02-01-01-06.wav
    #                                        ^                 ^
    #                                      spkr              text
    # 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door"

    items = path.split('/')
    speaker_id = items[-2]
    fname = items[-1]
    utt_id = fname.split('-')[4]
    if utt_id == '01':
        text = "Kids are talking by the door"
    else:
        text = "Dogs are sitting by the door"
    return text, speaker_id



def gen_ravdess_fixed_list(
    speaker_id,
    fixed_list_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/fixed_golden_list.txt',
    fixed_utt_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/fixed_golden_ravdess_wav_paths.txt',):
    # return in format
    # ref_wav_path|ref_speaker_id|ref_text|target_speaker_id|target_text
    sentences = ["Kids are talking by the door", "Dogs are sitting by the door"]
    utt_paths = []
    # collect all fixed samples
    with open(fixed_list_path, 'r') as f:
        for line in f:
            sentences.append(line.strip().split('|')[1])
    with open(fixed_utt_path, 'r') as f:
        for line in f:
            utt_paths.append(line.strip())
    lines = []
    for utt_path in utt_paths:
        ref_text, ref_speaker_id = ravdess_path_to_text_and_speaker(utt_path)
        for sentence in sentences:
            target_text = sentence
            lines.append(f'{utt_path}|{ref_speaker_id}|{ref_text}|{speaker_id}|{target_text}\n')
    return lines


def gen_pabc_fixed_list(
    speaker_id,
    utt_id_to_text,
    fixed_list_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/fixed_golden_list.txt',
    fixed_utt_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/fixed_golden_pabc_wav_paths.txt'):
    # return in format
    # ref_wav_path|ref_speaker_id|ref_text|target_speaker_id|target_text
    sentences = []
    utt_paths = []
    # collect all fixed samples
    with open(fixed_list_path, 'r') as f:
        for line in f:
            sentences.append(line.strip().split('|')[1])
    with open(fixed_utt_path, 'r') as f:
        for line in f:
            utt_paths.append(line.strip())
    lines = []
    for utt_path in utt_paths:
        spkr_name = utt_path.split('/')[6]
        ref_utt_id = os.path.splitext(os.path.split(utt_path)[1])[0]
        ref_speaker_id = spkrs2ids[spkr_name]
        ref_text = utt_id_to_text[ref_utt_id]
        for sentence in sentences:
            target_text = sentence
            lines.append(f'{utt_path}|{ref_speaker_id}|{ref_text}|{speaker_id}|{target_text}\n')
    return lines


def gen_utt_id_to_text(
    base_path:str='/home/co-sigu1/rds/hpc-work/pabc_daft_out/',
    out_path:str='./utt_id_to_text.txt'):

    from collections import defaultdict

    out = defaultdict(str)

    for spkr in os.listdir(base_path):
        md_file = os.path.join(base_path, spkr, 'metadata.csv')
        with open(md_file, 'r') as f:
            for line in f:
                utt_id, text = line.strip().split('|')
                out[utt_id] = text

    with open(out_path, 'w') as f:
        for utt_id, text in out.items():
            f.write(f'{utt_id}|{text}\n')


def gen_seen_by_all_list(out_path:str='./golden_lists/seen_all.txt'):
    bp = '/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/'
    in_paths = [
        os.path.join(bp, 'cleaned_pabc', 'train_english.txt'),
        os.path.join(bp, 'cleaned_pabc_v2', 'train_english.txt'),
        os.path.join(bp, 'cleaned_pabc_v3', 'train_english.txt'),
        os.path.join(bp, 'cleaned_pabc_v4', 'train_english.txt'),
        os.path.join(bp, 'cleaned_pabc_v5', 'train_english.txt'),
    ]

    lines = {}
    with open(in_paths[0], 'r') as f:
        for line in f:
            lines[line] = 1

    for i in range(1, len(in_paths)):
        with open(in_paths[i], 'r') as f:
            for line in f:
                items = line.split('|')
                line = '|'.join(items[:3])+'\n'
                if line in lines:
                    lines[line] += 1
    out_lines = []
    for line, count in lines.items():
        if count == 5:
            out_lines.append(line)

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(line)

def gen_unseen_golden(
    validation_lines='./text_results/all_max200_evaluation.txt',
    out_path='./golden_lists/unseen_pabc.txt',
    out_random_path:str='./golden_lists/unseen_pabc_random.txt'):

    eval_lines = []
    with open(validation_lines, 'r') as f:
        for line in f:
            eval_lines.append(line.strip())


    import random
    random_texts = random.sample(list(utt_id_to_text.values()), 15)
    random_refs = random.sample(eval_lines, 15)
    random_text_lines = []
    for target_speaker_id in speaker_ids:
        spkr_random_texts = random_texts + random.sample(list(utt_id_to_text.values()), 15)
        spkr_random_refs = random_refs + random.sample(eval_lines, 15)
        for i in range(30):
            line = spkr_random_refs[i]
            # generate target speaker and text
            ref_path = pabc_line_to_path(line)
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            ref_speaker_id = line.split('|')[2]
            ref_text = utt_id_to_text[line.split('|')[1]]
            target_text = spkr_random_texts[i]
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            random_text_lines.append(line)
    with open(out_random_path, 'w') as f:
        for line in random_text_lines:
            f.write(line)

    # now generate lines with actual text matches
    same_text_lines = []
    random_lines = random.sample(eval_lines, 30)
    # randomly sample 30 lines from the seen list
    for target_speaker_id in speaker_ids:
        for i in range(30):
            line = random_lines[i]
            ref_path = pabc_line_to_path(line)
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            ref_speaker_id = line.split('|')[2]
            ref_text = utt_id_to_text[line.split('|')[1]]
            target_text = ref_text
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            same_text_lines.append(line)
    with open(out_path, 'w') as f:
        for line in same_text_lines:
            f.write(line)


def gen_unseen_similar_golden(
    validation_lines='./text_results/all_max200_evaluation.txt',
    seen_list_path:str='./golden_lists/seen_all.txt',
    out_path='./golden_lists/unseen_similar_pabc.txt',
    out_random_path:str='./golden_lists/unseen_similar_pabc_random.txt'):

    seen_lines = []
    with open(seen_list_path, 'r') as f:
        for line in f:
            seen_lines.append(line.strip())
    seen_lines = limit_lines_to_length(seen_lines, 75)
    seen_lines = sort_lines_by_length(seen_lines, utt_id_to_text)
    eval_lines = []
    with open(validation_lines, 'r') as f:
        for line in f:
            eval_lines.append(line.strip())
    eval_lines = limit_lines_to_length(eval_lines, 75)
    import random
    #random_texts = random.sample(list(utt_id_to_text.values()), 15)
    random_refs = random.sample(list(enumerate(eval_lines)), 15)
    random_text_lines = []
    target_texts = []
    for i in range(15):
        ref_index, line = random_refs[i]
        target_texts.append(utt_id_to_text[sample_similar_text_length(ref_index, seen_lines).split('|')[1]])
    for target_speaker_id in speaker_ids:
        #spkr_random_texts = random_texts + random.sample(list(utt_id_to_text.values()), 15)
        spkr_random_refs = random_refs + random.sample(list(enumerate(eval_lines)), 15)
        spkr_target_texts = target_texts
        for i in range(15):
            ref_index, line = random_refs[i]
            spkr_target_texts.append(utt_id_to_text[sample_similar_text_length(ref_index, seen_lines).split('|')[1]])
        for i in range(30):
            ref_index, line = spkr_random_refs[i]
            target_text = spkr_target_texts[i]
            # generate target speaker and text
            ref_path = pabc_line_to_path(line)
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            ref_speaker_id = line.split('|')[2]
            ref_text = utt_id_to_text[line.split('|')[1]]
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            random_text_lines.append(line)
    with open(out_random_path, 'w') as f:
        for line in random_text_lines:
            f.write(line)

    # now generate lines with actual text matches
    same_text_lines = []
    random_lines = random.sample(eval_lines, 30)
    # randomly sample 30 lines from the seen list
    for target_speaker_id in speaker_ids:
        for i in range(30):
            line = random_lines[i]
            ref_path = pabc_line_to_path(line)
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            ref_speaker_id = line.split('|')[2]
            ref_text = utt_id_to_text[line.split('|')[1]]
            target_text = ref_text
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            same_text_lines.append(line)
    with open(out_path, 'w') as f:
        for line in same_text_lines:
            f.write(line)


def limit_lines_to_length(lines, max_len=60):
    utt_id_to_text = get_utt_id_to_text()
    allowed = [k for k,u in utt_id_to_text.items() if len(u) <= max_len]

    out_lines = []
    for line in lines:
        utt_id = line.split('|')[1]
        if utt_id in allowed:
            out_lines.append(line)
    return out_lines


def sort_lines_by_length(all_lines, utt_it_to_text):
    import numpy as np
    lengths = []
    for line in all_lines:
        lengths.append(len(utt_id_to_text[line.split('|')[1]]))
    ordering = np.argsort(lengths)
    out_lines = []
    for o in ordering:
        out_lines.append(all_lines[o])
    return out_lines

def sample_similar_text_length(center, lines):
    # sample around the center
    import random
    start = max(0, center - 100)
    stop = min(len(lines)-1, center + 100)
    selected = lines[start:center] + lines[center+1:stop]

    return random.choice(selected)

def gen_seen_similar_golden(
    out_path:str='./golden_lists/seen_similar.txt',
    out_random_path:str='./golden_lists/seen_similar_random.txt',
    seen_list_path:str='./golden_lists/seen_all.txt'):
    # ref_wav_path|ref_speaker_id|ref_text|target_speaker_id|target_text

    # 30 same text for each speaker
    # 30 (15 of them are fixed for all speakers) random text for each speaker

    utt_id_to_text = get_utt_id_to_text()

    seen_lines = []
    with open(seen_list_path, 'r') as f:
        for line in f:
            seen_lines.append(line.strip())
    seen_lines = limit_lines_to_length(seen_lines, 40)
    seen_lines = sort_lines_by_length(seen_lines, utt_id_to_text)


    # sample 15 lines
    import random
    random_refs = random.sample(list(enumerate(seen_lines)), 15)
    random_text_lines = []
    target_texts = []
    for i in range(15):
        ref_index, line = random_refs[i]
        target_texts.append(utt_id_to_text[sample_similar_text_length(ref_index, seen_lines).split('|')[1]])
    for target_speaker_id in speaker_ids:
        #spkr_random_texts = random_texts + random_sample_text(15)
        spkr_random_refs = random_refs + random.sample(list(enumerate(seen_lines)), 15)
        spkr_target_texts = target_texts
        for i in range(15):
            ref_index, line = random_refs[i]
            spkr_target_texts.append(utt_id_to_text[sample_similar_text_length(ref_index, seen_lines).split('|')[1]])
        for i in range(30):
            ref_index, line = spkr_random_refs[i]
            # sample similar text length
            target_text = spkr_target_texts[i]
            # generate target speaker and text
            ref_path = pabc_line_to_path(line)
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            ref_speaker_id = line.split('|')[2]
            ref_text = utt_id_to_text[line.split('|')[1]]
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            random_text_lines.append(line)

    with open(out_random_path, 'w') as f:
        for line in random_text_lines:
            f.write(line)

    # now generate lines with actual text matches
    same_text_lines = []
    random_lines = random.sample(seen_lines, 30)
    # randomly sample 30 lines from the seen list
    for target_speaker_id in speaker_ids:
        for i in range(30):
            line = random_lines[i]
            ref_path = pabc_line_to_path(line)
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            ref_speaker_id = line.split('|')[2]
            ref_text = utt_id_to_text[line.split('|')[1]]
            target_text = ref_text
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            same_text_lines.append(line)
    with open(out_path, 'w') as f:
        for line in same_text_lines:
            f.write(line)


def gen_seen_golden(
    out_path:str='./golden_lists/seen.txt',
    out_random_path:str='./golden_lists/seen_random.txt',
    seen_list_path:str='./golden_lists/seen_all.txt'):
    # ref_wav_path|ref_speaker_id|ref_text|target_speaker_id|target_text

    # 30 same text for each speaker
    # 30 (15 of them are fixed for all speakers) random text for each speaker

    utt_id_to_text = get_utt_id_to_text()

    seen_lines = []
    with open(seen_list_path, 'r') as f:
        for line in f:
            seen_lines.append(line.strip())
    # sample 15 lines
    import random
    random_texts = random_sample_text(15)
    random_refs = random.sample(seen_lines, 15)
    random_text_lines = []
    for target_speaker_id in speaker_ids:
        spkr_random_texts = random_texts + random_sample_text(15)
        spkr_random_refs = random_refs + random.sample(seen_lines, 15)
        for i in range(30):
            line = spkr_random_refs[i]
            # generate target speaker and text
            ref_path = pabc_line_to_path(line)
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            ref_speaker_id = line.split('|')[2]
            ref_text = utt_id_to_text[line.split('|')[1]]
            target_text = spkr_random_texts[i]
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            random_text_lines.append(line)
    with open(out_random_path, 'w') as f:
        for line in random_text_lines:
            f.write(line)

    # now generate lines with actual text matches
    same_text_lines = []
    random_lines = random.sample(seen_lines, 30)
    # randomly sample 30 lines from the seen list
    for target_speaker_id in speaker_ids:
        for i in range(30):
            line = random_lines[i]
            ref_path = pabc_line_to_path(line)
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            ref_speaker_id = line.split('|')[2]
            ref_text = utt_id_to_text[line.split('|')[1]]
            target_text = ref_text
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            same_text_lines.append(line)
    with open(out_path, 'w') as f:
        for line in same_text_lines:
            f.write(line)


def gen_unseen_ravdess(
    out_path:str='./golden_lists/unseen_ravdess.txt',
    out_random_path:str='./golden_lists/unseen_ravdess_random.txt',
    ravdess_path:str='./text_results/ravdess.txt'):

    """
    Create 30 utterances with same text and 30 utterances with different
    text using ravdess references. Random text is sampled from PABC.

    15 of the random text / reference pairs should be same across all
    speakers.
    """
    utt_id_to_text = get_utt_id_to_text()

    ravdess_lines = []
    with open(ravdess_path, 'r') as f:
        for line in f:
            # paths
            ravdess_lines.append(line.strip())
    # sample 15 lines
    import random
    random_texts = random_sample_text(15)
    random_ravdess_paths = random.sample(ravdess_lines, 15)
    random_text_lines = []
    for target_speaker_id in speaker_ids:
        spkr_random_texts = random_texts + random_sample_text(15)
        spkr_random_refs = random_ravdess_paths + random.sample(ravdess_lines, 15)
        for i in range(30):
            ravdess_path = spkr_random_refs[i]
            ref_path = ravdess_path
            # generate target speaker and text
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            (ref_text, ref_speaker_id) = ravdess_path_to_text_and_speaker(ravdess_path)
            target_text = spkr_random_texts[i]
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            random_text_lines.append(line)
    with open(out_random_path, 'w') as f:
        for line in random_text_lines:
            f.write(line)
    # now generate lines with actual text matches
    same_text_lines = []
    random_lines = random.sample(ravdess_lines, 30)
    # randomly sample 30 lines from the seen list
    for target_speaker_id in speaker_ids:
        for i in range(30):
            line = random_lines[i]
            ref_path = line
            assert os.path.isfile(ref_path), f"file not found {ref_path}"
            (ref_text, ref_speaker_id) = ravdess_path_to_text_and_speaker(ref_path)
            target_text = ref_text
            line = f'{ref_path}|{ref_speaker_id}|{ref_text}|{target_speaker_id}|{target_text}\n'
            same_text_lines.append(line)
    with open(out_path, 'w') as f:
        for line in same_text_lines:
            f.write(line)


def pabc_line_to_path(line):
    '''
    the speaker is always index 2 and
    the path is always /home/co-sigu1/rds/hpc-work/pabc_daft_out/<speaker_name>/wavs/<utt_id>.wav
    '''
    items = line.split('|')
    return os.path.join('/home/co-sigu1/rds/hpc-work/pabc_daft_out/', ids2spkrs[int(items[2])], 'wavs', f'{items[1]}.wav')


def gen_mini_golden_list(
    out_path='./golden_lists/mini_v3.txt',
    num_pabc_fixed=0,
    num_ravdess_fixed=0,
    num_pabc_seen=0,
    num_pabc_seen_similar=10,
    num_pabc_seen_random=0,
    num_pabc_seen_similar_random=20,
    num_pabc_unseen=0,
    num_pabc_unseen_similar=10,
    num_pabc_unseen_random=0,
    num_pabc_unseen_similar_random=20,
    num_ravdess=0,
    num_ravdess_random=0):

    import random
    golden_pt = './golden_lists'
    pabc_fixed_path = os.path.join(golden_pt, 'pabc_fixed.txt')
    ravdess_fixed_path = os.path.join(golden_pt, 'ravdess_fixed.txt')
    pabc_seen = os.path.join(golden_pt, 'seen.txt')
    pabc_seen_similar = os.path.join(golden_pt, 'seen_similar.txt')
    pabc_seen_random = os.path.join(golden_pt, 'seen_random.txt')
    pabc_seen_similar_random = os.path.join(golden_pt, 'seen_similar_random.txt')
    pabc_unseen = os.path.join(golden_pt, 'unseen_pabc.txt')
    pabc_unseen_similar = os.path.join(golden_pt, 'unseen_similar_pabc.txt')
    pabc_unseen_random = os.path.join(golden_pt, 'unseen_pabc_random.txt')
    pabc_unseen_similar_random = os.path.join(golden_pt, 'unseen_similar_pabc_random.txt')

    ravdess = os.path.join(golden_pt, 'unseen_ravdess.txt')
    ravdess_random = os.path.join(golden_pt, 'unseen_ravdess_random.txt')

    paths = [
        pabc_fixed_path,
        ravdess_fixed_path,
        pabc_seen,
        pabc_seen_similar,
        pabc_seen_random,
        pabc_seen_similar_random,
        pabc_unseen,
        pabc_unseen_similar,
        pabc_unseen_random,
        pabc_unseen_similar_random,
        ravdess,
        ravdess_random]
    nums = [
        num_pabc_fixed,
        num_ravdess_fixed,
        num_pabc_seen,
        num_pabc_seen_similar,
        num_pabc_seen_random,
        num_pabc_seen_similar_random,
        num_pabc_unseen,
        num_pabc_unseen_similar,
        num_pabc_unseen_random,
        num_pabc_unseen_similar_random,
        num_ravdess,
        num_ravdess_random]


    out_lines = []

    for (path, num) in zip(paths, nums):
        lines = []
        with open(path, 'r') as f:
            for line in f:
                lines.append(line)
        out_lines += random.sample(lines, num)

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(line)


def get_speaker_info(speaker_info_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/text_results/speaker_info.txt'):
    info = {}
    with open(speaker_info_path, 'r') as f:
        for line in f:
            items = line.strip().split('|')
            info[items[0]] = {
                'name': items[1],
                'gender': items[2],
                'duration': items[3]
            }
    return info


def gen_speaker_similarity_list(
    golden_list_path='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/golden_lists/mini_v2.txt',
    out_path='./speaker_similarity_list.txt',
    num_samples=30):
    """
    write out in:
    synth_path | real_speaker_sample | other_speaker_sample
    """
    spk_info = get_speaker_info()

    golden_lines = []
    with open(golden_list_path, 'r') as f:
        for line in f:
            # append only lines where gender matches
            _, ref_spkr, _, target_spkr, _ = line.strip().split('|')
            if ref_spkr != target_spkr:
                if spk_info[ref_spkr]['gender'] == spk_info[target_spkr]['gender']:
                    golden_lines.append(line)
    import random
    out_lines = random.sample(golden_lines, num_samples)

    with open(out_path, 'w') as f:
        for l in out_lines:
            f.write(l)


def mos_list(
    in_unseen_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/golden_lists/unseen_pabc.txt',
    in_unseen_random_path:str='/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/golden_lists/unseen_pabc_random.txt',
    out_path='./mos_list.txt'):
    out_lines = []
    in_lines = []
    with open(in_unseen_path, 'r') as f:
        for line in f:
            in_lines.append(line)

    spkr_offset = 30
    for i in range(15):
        out_lines.append(in_lines[i+spkr_offset*i])

    in_lines = []
    with open(in_unseen_random_path, 'r') as f:
        for line in f:
            in_lines.append(line)

    spkr_offset = 30
    for i in range(15):
        out_lines.append(in_lines[i+spkr_offset*i])

    with open(out_path, 'w') as f:
        for line in out_lines:
            f.write(line)


if __name__ == "__main__":
    #gen_utt_id_to_text()
    #gen_hifi_lists()
    #gen_random_pairs()
    #remove_validation_from_list(using_ref=True)
    #gen_seen_golden()
    #gen_unseen_golden()
    #gen_unseen_ravdess()
    #gen_unseen_ravdess()
    #gen_seen_similar_golden()
    #gen_unseen_similar_golden()
    mos_list()
    '''
    for speaker_id in speaker_ids:
        all_lines += gen_ravdess_fixed_list(speaker_id, utt_id_to_text)
        with open('./golden_lists/ravdess_fixed.txt', 'w') as f:
            for line in all_lines:
                assert os.path.isfile(line.split('|')[0]), f"file not found {line.split('|')[0]}"
                f.write(line)
    all_lines = []
    for speaker_id in speaker_ids:
        all_lines += gen_ravdess_fixed_list(speaker_id)
        with open('./golden_lists/ravdess_fixed.txt', 'w') as f:
            for line in all_lines:
                assert os.path.isfile(line.split('|')[0]), f"file not found {line.split('|')[0]}"
                f.write(line)
    '''
