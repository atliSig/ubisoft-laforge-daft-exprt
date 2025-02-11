import argparse
import json
import logging
import os
import sys

from shutil import copyfile
from subprocess import call

# ROOT directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.environ['PYTHONPATH'] = os.path.join(PROJECT_ROOT, 'src')
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from daft_exprt.create_sets import create_sets
from daft_exprt.extract_features import check_features_config_used, extract_features, extract_mel_specs
from daft_exprt.features_stats import extract_features_stats
from daft_exprt.hparams import HyperParams
from daft_exprt.mfa import mfa
from daft_exprt.utils import get_nb_jobs
from daft_exprt.extract_features import extract_pitch

import librosa
import json
import numpy as np
import matplotlib.pyplot as plt

_logger = logging.getLogger(__name__)


def list_all_speakers(data_set_dir):
    ''' List all speakers contained in data_set_dir
    '''
    # initialize variables
    speakers = []
    data_set_dir = os.path.normpath(data_set_dir)
    # walk into data_set_dir
    for root, directories, files in os.walk(data_set_dir):
        if 'wavs' in directories and 'metadata.csv' in files:
            # extract speaker data set relative path
            spk_relative_path = os.path.relpath(root, data_set_dir)
            spk_relative_path = os.path.normpath(spk_relative_path)
            speakers.append(f'{spk_relative_path}')

    return speakers


def extract_mels(extract_mel_args):
    if extract_mel_args.out_dir == '':
        out_dir = None
    else:
        out_dir = extract_mel_args.out_dir
        os.makedirs(extract_mel_args.out_dir, exist_ok=True)
    nb_jobs = get_nb_jobs(extract_mel_args.nb_jobs)
    extract_mel_specs(
        extract_mel_args.file_list, out_dir, hparams, nb_jobs)


def extract_feats(extract_pitch_args):
    # /home/co-sigu1/projects/ubisoft-laforge-daft-exprt/mini_v3_qualtreats/ground_truth
    stats = json.load(open('/home/co-sigu1/projects/ubisoft-laforge-daft-exprt/trainings/cleaned_pabc/stats.json'))
    lines = []
    diffs = []
    with open(extract_pitch_args.list_path, 'r') as f:
        for line in f:
            lines.append(line.strip().split('|'))
    out_lines = []
    # ref speaker ind = 1, target speaker ind = 3
    for i in range(len(list(os.listdir(extract_pitch_args.dir)))):
        path = os.path.join(extract_pitch_args.dir, f'sample_{i}.wav')
        wav, fs = librosa.load(path, sr=hparams.sampling_rate)
        pitch_frames = extract_pitch(wav, fs, None, hparams, log=False)
        plt.plot(pitch_frames[pitch_frames!=0])

        sample_f0 = np.mean(pitch_frames[pitch_frames!=0])
        ref_f0 = np.exp(stats[f"spk {lines[i][1]}"]["pitch"]['mean'])
        tar_f0 = np.exp(stats[f"spk {lines[i][3]}"]["pitch"]['mean'])
        diffs.append(np.abs(sample_f0-tar_f0))
        out_lines.append(f'{sample_f0}, {ref_f0}, {tar_f0}')

    for line in out_lines:
        print(line)
    print(np.mean(diffs))


def plot_f0s(extract_pitch_args):
    exp_names = ['Daft-exprt', 'shuffle', 'text-based', 'F0-based']
    for i in range(60):
        fig, axs = plt.subplots(4, 1)
        ind = 0
        path = os.path.join(extract_pitch_args.main_dir, 'ground_truth', f'sample_{i}.wav')
        wav, fs = librosa.load(path, sr=hparams.sampling_rate)
        gt_pitch_frames = extract_pitch(wav, fs, None, hparams, log=False)
        for dir_name in ['pabc_cleaned_mini', 'pabc_cleaned_mini_v2', 'pabc_cleaned_mini_v4', 'pabc_cleaned_mini_v5']:
            path = os.path.join(extract_pitch_args.main_dir, dir_name, f'sample_{i}.wav')
            wav, fs = librosa.load(path, sr=hparams.sampling_rate)
            pitch_frames = extract_pitch(wav, fs, None, hparams, log=False)
            axs[ind].plot(gt_pitch_frames, label=exp_names[ind], color='blue')
            axs[ind].plot(pitch_frames, label=exp_names[ind], color='red')
            axs[ind].set_ylabel(f"F0 (Hz)\n{exp_names[ind]}")
            ind += 1
        fig.tight_layout()
        plt.savefig(f'./plots_2/s_{i}.png')

def pre_process(pre_process_args):
    ''' Pre-process speakers data sets for training
    '''
    # check experiment folder is new
    checkpoint_dir = os.path.join(output_directory, 'checkpoints')
    if os.path.isdir(checkpoint_dir):
        print(f'"{output_directory}" has already been used for a previous training experiment')
        print(f'Cannot perform pre-processing')
        print(f'Please change the "experiment_name" script argument\n')
        sys.exit(1)

    # set logger config
    log_dir = os.path.join(output_directory, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'pre_processing.log')
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w')
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    # create default location for features dir if not specified by the user
    features_dir = os.path.join(pre_process_args.features_dir, pre_process_args.language, f'{hparams.sampling_rate}Hz') \
        if pre_process_args.features_dir == os.path.join(PROJECT_ROOT, "datasets") else pre_process_args.features_dir
    # check current config is the same than the one used in features dir
    if os.path.isdir(features_dir):
        same_config = check_features_config_used(features_dir, hparams)
        assert(same_config), _logger.error(f'"{features_dir}" contains data that were extracted using a different set '
                                           f'of hyper-parameters. Please change the "features_dir" script argument')
    # set number of parallel jobs
    nb_jobs = get_nb_jobs(pre_process_args.nb_jobs)
    # perform alignment using MFA
    mfa(data_set_dir, hparams, nb_jobs)

    # copy metadata.csv
    for speaker in hparams.speakers:
        spk_features_dir = os.path.join(features_dir, speaker)
        os.makedirs(spk_features_dir, exist_ok=True)
        metadata_src = os.path.join(data_set_dir, speaker, 'metadata.csv')
        metadata_dst = os.path.join(features_dir, speaker, 'metadata.csv')
        assert(os.path.isfile(metadata_src)), _logger.error(f'There is no such file: {metadata_src}')
        copyfile(metadata_src, metadata_dst)
    # extract features
    extract_features(data_set_dir, features_dir, hparams, nb_jobs)
    # create train and valid sets
    create_sets(features_dir, hparams, pre_process_args.proportion_validation)
    # extract features stats on the training set
    stats = extract_features_stats(hparams, nb_jobs)
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4, sort_keys=True)


def train(train_args):
    ''' Train Daft-Exprt on the pre-processed data sets
    '''
    # launch training in distributed mode or not
    training_script = os.path.join(PROJECT_ROOT, 'src', 'daft_exprt', 'train.py')
    process = ['python', f'{training_script}',
               '--data_set_dir', f'{data_set_dir}',
               '--config_file', f'{config_file}',
               '--benchmark_dir', f'{benchmark_dir}',
               '--log_file', f"{os.path.join(output_directory, 'logs', 'training.log')}",
               '--world_size', f'{train_args.world_size}',
               '--rank', f'{train_args.rank}',
               '--master', f'{train_args.master}']
    if not train_args.no_multiprocessing_distributed:
        process.append('--multiprocessing_distributed')
    call(process)


def fine_tune(fine_tune_args):
    ''' Generate data sets with the Daft-Exprt trained model for vocoder fine-tuning
    '''
    # launch fine-tuning
    fine_tune_script = os.path.join(PROJECT_ROOT, 'src', 'daft_exprt', 'fine_tune.py')
    process = ['python', f'{fine_tune_script}',
               '--data_set_dir', f'{data_set_dir}',
               '--config_file', f'{config_file}',
               '--log_file', f"{os.path.join(output_directory, 'logs', 'fine_tuning.log')}",
               '--ft_dir', f'{fine_tune_args.ft_directory}',
               '--custom_path', f'{fine_tune_args.custom_path}']
    call(process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to pre-process speakers data sets and train with Daft-Exprt')
    subparsers = parser.add_subparsers(help='commands for pre-processing, training and generating data for vocoder fine-tuning')

    parser.add_argument('-en', '--experiment_name', type=str,
                        help='directory name where all pre-process, training and fine-tuning outputs will be stored')
    parser.add_argument('-dd', '--data_set_dir', type=str,
                        help='path to the directory containing speakers data sets')
    parser.add_argument('-spks', '--speakers', nargs='*', default=[],
                        help='speakers to use for training. '
                             'If [], finds all speakers contained in data_set_dir')
    parser.add_argument('-lg', '--language', type=str, default='english',
                        help='spoken language of the speakers that are stored in data_set_dir')

    parser_pre_process = subparsers.add_parser('pre_process', help='pre-process speakers data sets for training')
    parser_pre_process.set_defaults(func=pre_process)
    parser_pre_process.add_argument('-fd', '--features_dir', type=str, default=f'{os.path.join(PROJECT_ROOT, "datasets")}',
                                    help='path to the directory where pre-processed data sets will be stored')
    parser_pre_process.add_argument('-pv', '--proportion_validation', type=float, default=0.1,
                                    help='for each speaker, proportion of examples (%) that will be in the validation set')
    parser_pre_process.add_argument('-nj', '--nb_jobs', type=str, default='6',
                                    help='number of cores to use for python multi-processing')

    parser_extract_mels = subparsers.add_parser('extract_mels', help='Extract mels ')
    parser_extract_mels.set_defaults(func=extract_mels)
    parser_extract_mels.add_argument('--file_list', type=str)
    parser_extract_mels.add_argument('--out_dir', type=str, default='')
    parser_extract_mels.add_argument('-nj', '--nb_jobs', type=str, default='6',
                                    help='number of cores to use for python multi-processing')

    parser_extract_pitch = subparsers.add_parser('extract_feats', help='Extract mels ')
    parser_extract_pitch.set_defaults(func=extract_feats)
    parser_extract_pitch.add_argument('--dir', type=str)
    parser_extract_pitch.add_argument('--list_path', type=str)

    parser_extract_pitch = subparsers.add_parser('plot_pitch', help='Extract mels ')
    parser_extract_pitch.set_defaults(func=plot_f0s)
    parser_extract_pitch.add_argument('--main_dir', type=str)

    parser_train = subparsers.add_parser('train', help='train Daft-Exprt on the pre-processed data sets')
    parser_train.set_defaults(func=train)
    parser_train.add_argument('-chk', '--checkpoint', type=str, default='',
                              help='checkpoint path to use to restart training at a specific iteration. '
                                   'If empty, starts training at iteration 0')
    parser_train.add_argument('-nmpd', '--no_multiprocessing_distributed', action='store_true',
                              help='disable PyTorch multi-processing distributed training')
    parser_train.add_argument('-ws', '--world_size', type=int, default=1,
                              help='number of nodes for distributed training')
    parser_train.add_argument('-r', '--rank', type=int, default=0,
                              help='node rank for distributed training')
    parser_train.add_argument('-m', '--master', type=str, default='tcp://localhost:54321',
                              help='url used to set up distributed training')
    parser_train.add_argument('--using_pabc', type=bool, default=False,
                              help='Set this to True if reference does not match target')

    parser_fine_tune = subparsers.add_parser('fine_tune', help='generate data sets with the Daft-Exprt trained model for vocoder fine-tuning')
    parser_fine_tune.set_defaults(func=fine_tune)
    parser_fine_tune.add_argument('-chk', '--checkpoint', type=str,
                                  help='checkpoint path to use for creating the data set for fine-tuning')
    parser_fine_tune.add_argument('-ftdir', '--ft_directory', type=str, default='',
                                  help='Path to directory where the fine tuning dataset will be stored.')
    parser_fine_tune.add_argument('-cp', '--custom_path', type=str, default=None,
                                  help='Use this to generate custom fine tuning samples')
    parser_fine_tune.add_argument('--using_pabc', type=bool, default=False,
                              help='Set this to True if reference does not match target')
    args = parser.parse_args()

    # create path variables
    data_set_dir = args.data_set_dir
    output_directory = os.path.join(PROJECT_ROOT, 'trainings', args.experiment_name)
    training_files = os.path.join(output_directory, f'train_{args.language}.txt')
    validation_files = os.path.join(output_directory, f'validation_{args.language}.txt')
    config_file = os.path.join(output_directory, 'config.json')
    stats_file = os.path.join(output_directory, 'stats.json')
    benchmark_dir = os.path.join(PROJECT_ROOT, 'scripts', 'benchmarks')

    # find all speakers in data_set_dir if not specified in the args
    args.speakers = list_all_speakers(data_set_dir) if len(args.speakers) == 0 else args.speakers

    if 'using_pabc' in args:
        using_pabc = args.using_pabc
    else:
        using_pabc = False
    # fill hparams dictionary with mandatory keyword arguments
    hparams_kwargs = {
        'training_files': training_files,
        'validation_files': validation_files,
        'output_directory': output_directory,
        'language': args.language,
        'speakers': args.speakers,
        'using_pabc': using_pabc
    }
    # fill hparams dictionary to overwrite default hyper-param values
    hparams_kwargs['checkpoint'] = args.checkpoint if hasattr(args, 'checkpoint') else ''

    # create hyper-params object and save config parameters
    hparams = HyperParams(**hparams_kwargs)
    hparams.save_hyper_params(config_file)

    # run args
    args.func(args)
