import argparse
import json
import logging
import os
import time

import librosa
import numpy as np
import torch

from scipy.io.wavfile import write

from daft_exprt.data_loader import prepare_data_loaders, prepare_custom_data_loader
from daft_exprt.extract_features import mel_spectrogram_HiFi, rescale_wav_to_float32
from daft_exprt.hparams import HyperParams
from daft_exprt.model import DaftExprt
from daft_exprt.utils import estimate_required_time


_logger = logging.getLogger(__name__)


def fine_tuning(hparams, ft_dir, custom_path=None):
    ''' Extract mel-specs and audio files for Vocoder fine-tuning

    :param hparams:     hyper-params used for pre-processing and training
    '''
    # ---------------------------------------------------------
    # create model
    # ---------------------------------------------------------
    # load model on GPU
    torch.cuda.set_device(0)
    model = DaftExprt(hparams).cuda(0)

    # ---------------------------------------------------------
    # load checkpoint
    # ---------------------------------------------------------
    assert(hparams.checkpoint != ""), _logger.error(f'No checkpoint specified -- {hparams.checkpoint}')
    checkpoint_dict = torch.load(hparams.checkpoint, map_location=f'cuda:{0}')
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint_dict['state_dict'].items()}
    model.load_state_dict(state_dict)

    # ---------------------------------------------------------
    # prepare Data Loaders
    # ---------------------------------------------------------
    hparams.multiprocessing_distributed = False
    if custom_path is not None:
        data_loader = prepare_custom_data_loader(
            hparams, custom_path, num_workers=0, drop_last=False)
    else:
        data_loader, _, _, _ = \
            prepare_data_loaders(hparams, num_workers=0, drop_last=False)

    # ---------------------------------------------------------
    # create folders to store fine-tuning data set
    # ---------------------------------------------------------
    experiment_root = os.path.dirname(hparams.training_files)
    if ft_dir is not None:
        ft_data_set = ft_dir
    else:
        ft_data_set = os.path.join(experiment_root, 'fine_tuning_dataset')
    hparams.ft_data_set = ft_data_set
    if hparams.using_pabc:
        os.makedirs(ft_data_set, exist_ok=True)
    else:
        for speaker in hparams.speakers:
            os.makedirs(os.path.join(ft_data_set, speaker), exist_ok=True)

    # ==============================================
    #                   MAIN LOOP
    # ==============================================
    model.eval()  # set eval mode
    start = time.time()
    with torch.no_grad():
        # iterate over examples of train set
        for idx, batch in enumerate(data_loader):
            estimate_required_time(nb_items_in_list=len(data_loader), current_index=idx,
                                   time_elapsed=time.time() - start, interval=1)
            if hparams.using_pabc:
                (gt_inputs, _, gt_file_ids) = model.parse_batch(0, batch[0])
                (ref_inputs, _, ref_file_ids) = model.parse_batch(0, batch[1])
                inputs = (gt_inputs, ref_inputs)
                outputs = model(inputs)
                _, _, _, decoder_preds, _ = outputs
                mel_spec_preds, output_lengths = decoder_preds
                mel_spec_preds = mel_spec_preds.detach().cpu().numpy()  # (B, nb_mels, T_max)
                output_lengths = output_lengths.detach().cpu().numpy()  # (B, )
                # we will be using
                ref_feature_dirs, ref_feature_files = ref_file_ids
                feature_dirs, feature_files = gt_file_ids  # (B, ) and (B, )

                gt_speaker_ids = gt_inputs[-1].detach().cpu().numpy()
                ref_speaker_ids = ref_inputs[-1].detach().cpu().numpy()
            else:
                inputs, _, file_ids = model.parse_batch(0, batch)
                feature_dirs, feature_files = file_ids  # (B, ) and (B, )

                outputs = model(inputs)
                _, _, _, decoder_preds, _ = outputs
                mel_spec_preds, output_lengths = decoder_preds
                mel_spec_preds = mel_spec_preds.detach().cpu().numpy()  # (B, nb_mels, T_max)
                output_lengths = output_lengths.detach().cpu().numpy()  # (B, )

            # iterate over examples in the batch
            for idx in range(mel_spec_preds.shape[0]):
                mel_spec_pred = mel_spec_preds[idx]  # (nb_mels, T_max)
                output_length = output_lengths[idx]
                feature_dir = feature_dirs[idx]
                feature_file = feature_files[idx]
                # crop mel-spec prediction to the correct size
                mel_spec_pred = mel_spec_pred[:, :output_length]  # (nb_mels, T)
                # extract speaker name
                speaker_name = [speaker for speaker in hparams.speakers if feature_dir.endswith(speaker)]
                assert(len(speaker_name) == 1), _logger.error(f'{feature_dir} -- {feature_file} -- {speaker_name}')
                speaker_name = speaker_name[0]
                # read wav file to range [-1, 1] in np.float32
                wav_file = os.path.join(hparams.data_set_dir, speaker_name, 'wavs', f'{feature_file}.wav')
                wav, fs = librosa.load(wav_file, sr=hparams.sampling_rate)
                wav = rescale_wav_to_float32(wav)
                # crop audio to remove tailing silences based on markers file
                markers_file = os.path.join(hparams.data_set_dir, speaker_name, 'align', f'{feature_file}.markers')
                with open(markers_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                sent_begin = float(lines[0].strip().split(sep='\t')[0])
                sent_end = float(lines[-1].strip().split(sep='\t')[1])
                wav = wav[int(sent_begin * fs): int(sent_end * fs)]
                # check target and predicted mel-spec have the same size
                mel_spec_tgt = mel_spectrogram_HiFi(wav, hparams)
                assert(mel_spec_tgt.shape == mel_spec_pred.shape), \
                    _logger.error(f'{feature_dir} -- {feature_file} -- {mel_spec_tgt.shape} -- {mel_spec_pred.shape}')
                # save audio and mel-spec if they have the correct size (superior to 1s)
                if len(wav) >= fs:
                    # convert to int16
                    wav = wav * 32768.0
                    wav = wav.astype('int16')
                    # store files in fine-tuning data set
                    if hparams.using_pabc:
                        # save wav as is, save spectrogram as a combination
                        #ref_feature_dir = ref_feature_dirs[idx]
                        #ref_feature_file = ref_feature_files[idx]
                        #ref_speaker_name = [speaker for speaker in hparams.speakers if ref_feature_dir.endswith(speaker)][0]
                        #assert(len(ref_speaker_name) == 1), _logger.error(f'{ref_feature_dir} -- {ref_feature_file} -- {ref_speaker_name}')
                        #ref_speaker_name = speaker_name[0]
                        mel_spec_file = os.path.join(hparams.ft_data_set, f'{gt_speaker_ids[idx]}_{ref_speaker_ids[idx]}_{feature_file}.npy')
                        wav_file = os.path.join(hparams.ft_data_set, f'gt_{gt_speaker_ids[idx]}_{feature_file}.wav')
                    else:
                        mel_spec_file = os.path.join(hparams.ft_data_set, speaker_name, f'{feature_file}.npy')
                        wav_file = os.path.join(hparams.ft_data_set, speaker_name, f'{feature_file}.wav')
                    try:
                        np.save(mel_spec_file, mel_spec_pred)
                        write(wav_file, fs, wav)
                    except Exception as e:
                        _logger.error(f'{feature_dir} -- {feature_file} -- {e}')
                        if os.path.isfile(mel_spec_file):
                            os.remove(mel_spec_file)
                        if os.path.isfile(wav_file):
                            os.remove(wav_file)
                else:
                    _logger.warning(f'{feature_dir} -- {feature_file} -- Ignoring because audio is < 1s')


def launch_fine_tuning(data_set_dir, config_file, log_file, ft_dir, custom_path=None):
    ''' Launch fine-tuning
    '''
    # set logger config
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ],
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )

    # get hyper-parameters
    with open(config_file) as f:
        data = f.read()
    config = json.loads(data)
    hparams = HyperParams(verbose=False, **config)

    # update hparams
    hparams.data_set_dir = data_set_dir
    hparams.config_file = config_file

    # save hyper-params to config.json
    hparams.save_hyper_params(hparams.config_file)

    # define cudnn variables
    torch.manual_seed(0)
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
    torch.backends.cudnn.deterministic = hparams.cudnn_deterministic

    # display fine-tuning setup info
    _logger.info(f'PyTorch version -- {torch.__version__}')
    _logger.info(f'CUDA version -- {torch.version.cuda}')
    _logger.info(f'CUDNN version -- {torch.backends.cudnn.version()}')
    _logger.info(f'CUDNN enabled = {torch.backends.cudnn.enabled}')
    _logger.info(f'CUDNN deterministic = {torch.backends.cudnn.deterministic}')
    _logger.info(f'CUDNN benchmark = {torch.backends.cudnn.benchmark}\n')

    # create fine-tuning data set
    fine_tuning(hparams, ft_dir, custom_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_set_dir', type=str, required=True,
                        help='Data set containing .wav files')
    parser.add_argument('--config_file', type=str, required=True,
                        help='JSON configuration file to initialize hyper-parameters for fine-tuning')
    parser.add_argument('--log_file', type=str, required=True,
                        help='path to save logger outputs')
    parser.add_argument('--ft_dir', type=str, required=False,
                        help='Directory where ft dataset will be stored')
    parser.add_argument('--custom_path', type=str, required=False, default="",
                        help='Use this to generate a fine-tune dataset using custom utterances')

    args = parser.parse_args()

    if args.custom_path == "None":
        args.custom_path = None
    # launch fine-tuning
    launch_fine_tuning(args.data_set_dir, args.config_file, args.log_file, args.ft_dir, args.custom_path)
