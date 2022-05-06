import argparse
import logging
import os
import random
import sys
import time

import torch

from shutil import copyfile

FILE_ROOT = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_ROOT)
os.environ['PYTHONPATH'] = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from daft_exprt.generate import extract_reference_parameters, generate_mel_specs, phonemize_list
from daft_exprt.hparams import HyperParams
from daft_exprt.model import DaftExprt
from daft_exprt.utils import get_nb_jobs


_logger = logging.getLogger(__name__)
random.seed(1234)


'''
    Script example that showcases how to generate with Daft-Exprt
    using a target sentence, a target speaker, and a target prosody
'''


def synthesize(args, dur_factor=None, energy_factor=None, pitch_factor=None,
               pitch_transform=None, use_griffin_lim=False, get_time_perf=False):
    ''' Generate with DaftExprt
    '''
    # get hyper-parameters that were used to create the checkpoint
    checkpoint_dict = torch.load(args.checkpoint, map_location=f'cuda:{0}')
    hparams = HyperParams(verbose=False, **checkpoint_dict['config_params'])
    # load model
    torch.cuda.set_device(0)
    model = DaftExprt(hparams).cuda(0)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
    model.load_state_dict(state_dict)

    # prepare sentences
    n_jobs = get_nb_jobs('max')

    # parse the input file and prepare for synthesis
    os.makedirs(args.output_dir, exist_ok=False)
    data = []
    with open(args.list_file, 'r') as f:
        for line in f:
            (ref_wav_path, ref_speaker_id, ref_text, target_speaker_id, target_text) = line.strip().split('|')
            data.append({
                'ref_wav_path': ref_wav_path,
                'ref_speaker_id': ref_speaker_id,
                'ref_text': ref_text,
                'target_speaker_id': int(target_speaker_id),
                'target_text': target_text + "."}) # have to append punctuation

    sentences = phonemize_list([d['target_text'] for d in data], hparams, n_jobs)
    speaker_ids = [d['target_speaker_id'] for d in data]
    audio_refs = [d['ref_wav_path'] for d in data]

    for i, ref_path in enumerate(audio_refs):
        new_path = os.path.join(args.output_dir, f'ref_{i}.wav')
        copyfile(ref_path, new_path)
        audio_refs[i] = new_path

    file_names = []
    for i in range(len(sentences)):
        fname = f'synth_{i}'
        file_names.append(fname)
    for audio_ref in audio_refs:
        extract_reference_parameters(audio_ref, args.output_dir, hparams)
    refs = []
    for i in range(len(sentences)):
        refs.append(f'{os.path.splitext(audio_refs[i])[0]}.npz')

    # add duration factors for each symbol in the sentence
    dur_factors = [] if dur_factor is not None else None
    energy_factors = [] if energy_factor is not None else None
    pitch_factors = [pitch_transform, []] if pitch_factor is not None else None
    for sentence in sentences:
        # count number of symbols in the sentence
        nb_symbols = 0
        for item in sentence:
            if isinstance(item, list):  # correspond to phonemes of a word
                nb_symbols += len(item)
            else:  # correspond to word boundaries
                nb_symbols += 1
        # append to lists
        if dur_factors is not None:
            dur_factors.append([dur_factor for _ in range(nb_symbols)])
        if energy_factors is not None:
            energy_factors.append([energy_factor for _ in range(nb_symbols)])
        if pitch_factors is not None:
            pitch_factors[1].append([pitch_factor for _ in range(nb_symbols)])

    # generate mel-specs and synthesize audios with Griffin-Lim
    generate_mel_specs(model, sentences, file_names, speaker_ids, refs, args.output_dir,
                       hparams, dur_factors, energy_factors, pitch_factors, args.batch_size,
                       n_jobs, use_griffin_lim, get_time_perf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to synthesize sentences with Daft-Exprt')

    parser.add_argument('-out', '--output_dir', type=str,
                        help='output dir to store synthesis outputs')
    parser.add_argument('-chk', '--checkpoint', type=str,
                        help='checkpoint path to use for synthesis')
    parser.add_argument('-lf', '--list_file', type=str, default='./golden_lists/mini.txt',
                        help='text file to use for synthesis')
    parser.add_argument('-bs', '--batch_size', type=int, default=50,
                        help='batch of sentences to process in parallel')
    parser.add_argument('-rtf', '--real_time_factor', action='store_true',
                        help='get Daft-Exprt real time factor performance given the batch size')
    parser.add_argument('-ctrl', '--control', action='store_true',
                        help='perform local prosody control during synthesis')

    args = parser.parse_args()

    # set logger config
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )

    if args.real_time_factor:
        synthesize(args, get_time_perf=True)
        time.sleep(5)
        _logger.info('')
    if args.control:
        # small hard-coded example that showcases duration and pitch control
        # control is performed on the sentence level in this example
        # however, the code also supports control on the word/phoneme level
        dur_factor = 1.25  # decrease speed
        pitch_transform = 'add'  # pitch shift
        pitch_factor = 50  # 50Hz
        synthesize(args, dur_factor=dur_factor, pitch_factor=pitch_factor,
                   pitch_transform=pitch_transform, use_griffin_lim=True)
    else:
        synthesize(args, use_griffin_lim=True)
