import librosa
import os
import numpy as np
import matplotlib.pyplot as plt

from daft_exprt.extract_features import extract_mel_specs
from scipy.io.wavfile import write


def rescale_wav_to_float32(x):
    ''' Rescale audio array between -1.f and 1.f based on the current format
    '''
    # convert
    if x.dtype == 'int16':
        y = x / 32768.0
    elif x.dtype == 'int32':
        y = x / 2147483648.0
    elif x.dtype == 'uint8':
        y = ((x / 255.0) - 0.5)*2
    elif x.dtype == 'float32' or x.dtype == 'float64':
        y = x
    else:
        raise TypeError(f"could not normalize wav, unsupported sample type {x.dtype}")
    # check amplitude is correct
    y = y.astype('float32')
    max_ampl = np.max(np.abs(y))
    if max_ampl > 1.0:
        print('max amplitude')
        pass  # the error should be raised but librosa returns values bigger than 1 sometimes
        # raise ValueError(f'float32 wav contains samples not in the range [-1., 1.] -- '
        #                  f'max amplitude: {max_ampl}')

    return y


def iterate_directory(directory):
    """
    Return a list of file paths for the typical
    directory structures used in this work
    """
    paths = []
    for speaker in os.listdir(directory):
        for fname in os.listdir(os.path.join(directory, speaker)):
            if fname.endswith('.wav'):
                #print(fname)
                paths.append(os.path.join(directory, speaker, fname))
    return paths

def trim_wav(in_path, target_sr=22050):
    wav, _ = librosa.load(in_path, sr=target_sr)
    wav = rescale_wav_to_float32(wav)
    wav_trimmed, _ = librosa.effects.trim(wav, top_db=35)
    write(in_path, target_sr, wav_trimmed)
    #fig = plt.plot(wav)
    #plt.savefig(f'{os.path.split(in_path)[1]}.png')
    #plt.clf()
    #fig = plt.plot(wav_trimmed)
    #plt.savefig(f'{os.path.split(in_path)[1]}_trimmed.png')
    #plt.clf()

if __name__ == "__main__":
    import random
    paths = iterate_directory('/home/co-sigu1/rds/hpc-work/ravdess')
    #random.shuffle(paths)
    for i in range(len(paths)):
        p = paths[i]
        print(p)
        trim_wav(p)