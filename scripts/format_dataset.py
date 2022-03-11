import argparse
import logging
import os

from shutil import copyfile
from collections import defaultdict

_logger = logging.getLogger(__name__)


'''
    This script modifies speakers data sets to match the required format
    Each speaker data set must be of the following format:

    /speaker_name
        metadata.csv
        /wavs
            wav_file_name_1.wav
            wav_file_name_2.wav
            ...

    metadata.csv must be formatted as follows (pipe "|" separator):
        wav_file_name_1|text_1
        wav_file_name_2|text_2
        ...
'''


def format_LJ_speech(lj_args):
    ''' Format LJ data set
        Only metadata.csv needs to be modified
    '''
    # read metadata lines
    _logger.info('Formatting LJ Speech')
    metadata = os.path.join(lj_args.data_set_dir, 'metadata.csv')
    assert(os.path.isfile(metadata)), _logger.error(f'There is no such file {metadata}')
    with open(metadata, 'r', encoding='utf-8') as f:
        metadata_lines = f.readlines()
    # create new metadata.csv
    metadata_lines = [line.strip().split(sep='|') for line in metadata_lines]
    metadata_lines = [f'{line[0]}|{line[2]}\n' for line in metadata_lines]
    with open(os.path.join(lj_args.data_set_dir, 'daft_metadata.csv'), 'w', encoding='utf-8') as f:
        f.writelines(metadata_lines)
    _logger.info('Done!')


def format_ESD(esd_args):
    ''' Format ESD data set
    '''
    # extract speaker dirs depending on the language
    _logger.info(f'Formatting ESD -- Language = {esd_args.language}')
    speakers = [x for x in os.listdir(esd_args.data_set_dir) if
                os.path.isdir(os.path.join(esd_args.data_set_dir, x))]
    speakers.sort()
    if esd_args.language == 'english':
        for speaker in speakers[10:]:
            _logger.info(f'Speaker -- {speaker}')
            speaker_dir = os.path.join(esd_args.data_set_dir, speaker)
            spk_out_dir = os.path.join(esd_args.data_set_dir, esd_args.language, speaker)
            os.makedirs(spk_out_dir, exist_ok=True)
            # read metadata lines
            if speaker == speakers[10]:
                metadata = os.path.join(speaker_dir,f'{speaker}.txt')
                assert(os.path.isfile(metadata)), _logger.error(f'There is no such file {metadata}')
                with open(metadata, 'r', encoding='utf-8') as f:
                    metadata_lines = f.readlines()
                metadata_lines = [line.strip().split(sep='\t') for line in metadata_lines]
            # create new metadata.csv
            # each line has format [filename(without extension)]\t[text]
            spk_metadata_lines = [f'{speaker}_{line[0].strip().split(sep="_")[1]}|{line[1]}\n'
                                  for line in metadata_lines]
            with open(os.path.join(spk_out_dir, 'metadata.csv'), 'w', encoding='utf-8') as f:
                f.writelines(spk_metadata_lines)
            # copy all audio files to /wavs directory
            wavs_dir = os.path.join(spk_out_dir, 'wavs')
            os.makedirs(wavs_dir, exist_ok=True)
            for root, _, files in os.walk(speaker_dir):
                wav_files = [x for x in files if x.endswith('.wav')]
                for wav_file in wav_files:
                    src = os.path.join(root, wav_file)
                    dst = os.path.join(wavs_dir, wav_file)
                    copyfile(src, dst)
    elif esd_args.language == 'mandarin':
        _logger.error(f'"mandarin" not implemented')
    else:
        _logger.error(f'"language" must be either "english" or "mandarin", not "{esd_args.language}"')
    _logger.info('Done!')


def format_PABC(pabc_args):
    _logger.info('Formatting PABC')
    utt_dict = defaultdict(list)
    books = [x for x in os.listdir(pabc_args.data_set_dir) if
                os.path.isdir(os.path.join(pabc_args.data_set_dir, x))]
    for book in books:
        print("checking book: ", book)
        book_dir = os.path.join(pabc_args.data_set_dir, book)

        # read in text
        utt2text = read_PABC_text(os.path.join(book_dir, 'txt.clean'))
        print("Number of texts: ", len(utt2text))

        # read in spk2utt
        spk2utt = read_PABC_spk2utt(os.path.join(book_dir, 'spk2utt'))
        print("Number of speakers: ", len(spk2utt))

        for speaker, utts in spk2utt.items():
            for utt_id in utts:
                new_utt_id = f'{book}_{utt_id}'
                utt_dict[speaker].append([book, utt_id, new_utt_id, utt2text[utt_id]])

    # each item in a speaker utt list is now of format similar to:
    # ['huck', 'chp06_utt138', 'huck_chp06_utt138', 'By and by I got the old split...down the gun']]

    # Create file structure and copy, avoiding any fname collisions.
    for speaker, utts in utt_dict.items():
        speaker_out_dir = os.path.join(pabc_args.data_set_out_dir, speaker)
        speaker_wav_out_dir = os.path.join(speaker_out_dir, 'wavs')
        os.makedirs(speaker_wav_out_dir, exist_ok=True)

        metadata_file = open(os.path.join(speaker_out_dir, 'metadata.csv'), 'w')
        for utt in utts:
            book, utt_id, new_utt_id, utt_text = utt[0], utt[1], utt[2], utt[3]
            # copy wav and change file name
            wav_in_path = os.path.join(pabc_args.data_set_dir, book, 'wav', speaker, f'{utt_id}.wav')
            wav_out_path = os.path.join(speaker_wav_out_dir, f'{new_utt_id}.wav')
            copyfile(wav_in_path, wav_out_path)
            # add info to index
            metadata_file.write(f'{new_utt_id}|{utt_text}\n')


def read_PABC_spk2utt(path:str):
    spk2utt = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            spk, *utts = line.strip().split()
            spk2utt[spk] = utts
    return spk2utt


def read_PABC_text(path:str):
    """Read in the standard txt.clean file used in PABC.
    Returns a dictionary with key=utt_id, value=text"""
    with open(path, 'r', encoding='utf-8') as f:
        metadata_lines = f.readlines()
    metadata_lines = [line.strip().split(sep='\t') for line in metadata_lines]
    return {m[0]: m[1] for m in metadata_lines}


def format_PABC_1(pabc_args):
    _logger.info('Formatting PABC')
    books = [x for x in os.listdir(pabc_args.data_set_dir) if
                os.path.isdir(os.path.join(pabc_args.data_set_dir, x))]
    speaker_utts = {}
    for book in books:
        book_dir = os.path.join(pabc_args.data_set_dir, book)
        book_wav_dir = os.path.join(book_dir, 'wav')
        speakers  = [x for x in os.listdir(book_wav_dir) if
                os.path.isdir(os.path.join(book_wav_dir, x))]
        # book_dir/spk2utt contains a mapping from each speaker to
        # each utterance (each line is [speaker]\s[utt_1]\s[utt_2]\s...)
        spk2utt = {}
        with open(os.path.join(book_dir, 'spk2utt'), 'r', encoding='utf-8') as f:
            for line in f:
                spk, *utts = line.strip().split()
                spk2utt[spk] = [f'{book}_{utt}' for utt in utts]
        # book_dir/txt.clean is a tsv file where each line is
        # <wav_fname>\t<text>.
        utt2text = {}
        with open(os.path.join(book_dir, 'txt.clean'), 'r', encoding='utf-8') as f:
                metadata_lines = f.readlines()
        metadata_lines = [line.strip().split(sep='\t') for line in metadata_lines]
        # add book ID to file identifiers
        for line in metadata_lines:
            line[0] = f'{book}_{line[0]}'
        utt2text = {m[0]:m[1] for m in metadata_lines}
        for speaker in speakers:
            # add new speakers to speaker_utts and required
            # file structure
            speaker_wav_in_dir = os.path.join(book_wav_dir, speaker)
            speaker_out_dir = os.path.join(pabc_args.data_set_dir, 'english', speaker)
            speaker_wav_out_dir = os.path.join(speaker_out_dir, 'wavs')
            if speaker not in speaker_utts:
                speaker_utts[speaker] = []
                os.makedirs(speaker_wav_out_dir, exist_ok=True)
            # iterate metadata
            book_utts_by_speaker = spk2utt[speaker]
            for fname in book_utts_by_speaker:
                # add metadata
                speaker_utts[speaker].append([fname, utt2text[fname]])
                # copy the wav
                wav_in_path = os.path.join(speaker_wav_in_dir, f'{fname}.wav')
                wav_out_path = os.path.join(speaker_wav_out_dir, f'{fname}.wav')
                copyfile(wav_in_path, wav_out_path)

            ## PROBLEM: THIS ASSUMES THAT BOOK ID HAS BEEN APPENDED.
            ## DONT CHANGE THE METDATA STUFF UNTIL COPYING.
    # write the metdata
    for speaker, metadata_lines in speaker_utts.items():
        with open(os.path.join(pabc_args.data_set_dir, 'english', speaker), 'w') as f:
            for line in metadata_lines:
                f.write(f'{metadata_lines[0]}|{metadata_lines[1]}\n')
    _logger.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to format speakers data sets')
    subparsers = parser.add_subparsers(help='commands for targeting a specific data set')

    parser.add_argument('-dd', '--data_set_dir', type=str,
                        help='path to the directory containing speakers data sets to format')

    parser_LJ = subparsers.add_parser('LJ', help='format LJ data set')
    parser_LJ.set_defaults(func=format_LJ_speech)

    parser_ESD = subparsers.add_parser('ESD', help='format emotional speech dataset from Zhou et al.')
    parser_ESD.set_defaults(func=format_ESD)
    parser_ESD.add_argument('-lg', '--language', type=str,
                            help='either english or mandarin')

    parser_PABC = subparsers.add_parser('PABC', help='format Parallel AudioBook Corpus')
    parser_PABC.set_defaults(func=format_PABC)
    parser_PABC.add_argument('--data_set_out_dir', type=str,
                             help="Path to the directory where the formatted dataset will be stored")

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

    # run args
    args.func(args)
