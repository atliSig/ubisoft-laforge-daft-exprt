import argparse
import logging
import os
import random
import sys
import time
import torch

from shutil import copyfile, rmtree

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

    # create output directory or delete everything if it already exists
    if os.path.exists(args.output_dir):
        rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=False)
    os.makedirs(os.path.join(args.output_dir, 'refs'), exist_ok=False)
    sentences, speaker_ids, ref_paths = [], [], []
    with open(args.list_path, 'r') as f:
        for line in f:
            text, spkr_id, ref_path = line.strip().split('|')
            sentences.append(text)
            speaker_ids.append(int(spkr_id))
            ref_paths.append(ref_path)
    sentences = phonemize_list(sentences, hparams, n_jobs)

    # store the references in the output directory
    audio_refs = []
    for ref_path in ref_paths:
        new_path = os.path.join(args.output_dir, 'refs', os.path.split(ref_path)[1])
        copyfile(ref_path, new_path)
        audio_refs.append(new_path)

    file_names = []
    for i in range(len(sentences)):
        fname = f'text_{i}_'
        file_names.append(fname)
    for audio_ref in audio_refs:
        extract_reference_parameters(audio_ref, os.path.join(args.output_dir, 'refs'), hparams)

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

    #return file_names, refs, speaker_ids


def pair_ref_and_generated(args, file_names, refs, speaker_ids):
    ''' Simplify prosody transfer evaluation by matching generated audio with its reference
    '''
    # save references to output dir to make prosody transfer evaluation easier
    for idx, (file_name, ref, speaker_id) in enumerate(zip(file_names, refs, speaker_ids)):
        # extract reference audio
        ref_file_name = os.path.basename(ref).replace('.npz', '')
        audio_ref = os.path.join(args.style_bank, f'{ref_file_name}.wav')
        # check correponding synthesized audio exists
        synthesized_file_name = f'{file_name}_spk_{speaker_id}_ref_{ref_file_name}'
        synthesized_audio = os.path.join(args.output_dir, f'{synthesized_file_name}.wav')
        assert(os.path.isfile(synthesized_audio)), _logger.error(f'There is no such file {synthesized_audio}')
        # rename files
        os.rename(synthesized_audio, f'{os.path.join(args.output_dir, f"{idx}_{synthesized_file_name}.wav")}')
        copyfile(audio_ref, f'{os.path.join(args.output_dir, f"{idx}_ref.wav")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to synthesize sentences with Daft-Exprt')

    parser.add_argument('-out', '--output_dir', type=str,
                        help='output dir to store synthesis outputs')
    parser.add_argument('-chk', '--checkpoint', type=str,
                        help='checkpoint path to use for synthesis')
    parser.add_argument('-lp', '--list_path', type=str,
                        help='path to the list to synthesise')
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
        #pair_ref_and_generated(args, file_names, refs, speaker_ids)
