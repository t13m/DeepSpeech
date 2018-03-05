from __future__ import absolute_import, print_function

import scipy.io.wavfile as wav
import sys
import math

try:
    from deepspeech import audioToInputVector
except ImportError:
    import numpy as np
    from python_speech_features import mfcc
    from six.moves import range

    class DeprecationWarning:
        displayed = False

    def audioToInputVector(audio, fs, numcep, numcontext):
        if DeprecationWarning.displayed is not True:
            DeprecationWarning.displayed = True
            print('------------------------------------------------------------------------', file=sys.stderr)
            print('WARNING: libdeepspeech failed to load, resorting to deprecated code',      file=sys.stderr)
            print('         Refer to README.md for instructions on installing libdeepspeech', file=sys.stderr)
            print('------------------------------------------------------------------------', file=sys.stderr)

        # Get mfcc coefficients
        features = np.empty([0, numcep])

        win_len = 0.025
        win_step = 0.010

        win_len_samples = int(math.ceil(win_len * fs))
        win_step_samples = int(math.ceil(win_step * fs)) * 2 # stride = 2

        features = np.empty([0, numcep])
        for i in range(0, len(audio), win_step_samples):
            frame = audio[i:i + win_len_samples]
            if len(frame) < win_step_samples:
                break
            framefeats = mfcc(frame,
                              samplerate=fs,
                              numcep=numcep,
                              winlen=win_len,
                              winstep=win_step)
            features = np.concatenate((features, framefeats))

        # One stride per time step in the input
        num_strides = len(features)

        # Add empty initial and final contexts
        empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2*numcontext+1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, numcep),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)

        # Flatten the second and third dimensions
        train_inputs = np.reshape(train_inputs, [num_strides, -1])

        # Return results
        return train_inputs


def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    r"""
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    """
    # Load wav files
    fs, audio = wav.read(audio_filename)

    return audioToInputVector(audio, fs, numcep, numcontext)
