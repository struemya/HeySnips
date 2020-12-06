import json
import numpy as np
from scipy.io import wavfile
from scipy.signal import medfilt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tqdm import tqdm
from vad import VoiceActivityDetector
import matplotlib.pyplot as plt
DataSetPath = "/Users/yannickstrumpler/Documents/Studium/MLoMCU/Lab2/hey_snips_research_6k_en_train_eval_clean_ter/"

with open(DataSetPath+"train.json") as jsonfile:
    traindata = json.load(jsonfile)

with open(DataSetPath+"test.json") as jsonfile:
    testdata = json.load(jsonfile)

fs = 16000  # Sampling rate of the samples

def detect_audio(tensor):
    energy_sqrt = abs(tensor)
    filtered_energy = medfilt(energy_sqrt, 151)
    max_filtered = max(filtered_energy)
    is_audio = filtered_energy > (max_filtered * 5e-2)
    is_audio_index = np.nonzero(is_audio)
    #beginning = min(is_audio_index[0])
    beginning = max(is_audio_index[0]) - 2560
    ending = max(is_audio_index[0]) + 2560
    mask = np.zeros_like(is_audio)
    mask[beginning:ending+1] = True
    #plt.figure()
    #plt.plot(data_speech, label='Speech')
    # plt.plot(tensor, label='data', alpha=0.7)
    # plt.plot(is_audio * 10000, label='bool_audio')
    # plt.plot(filtered_energy, label='filtered')
    # plt.legend()
    # plt.show()
    return mask

def load_data_time_series():
    x_train_list = []
    y_train_list = []

    x_test_list = []
    y_test_list = []
    totalSliceLength = 10
    trainsize = 10000 #len(traindata)  # Number of loaded training samples
    testsize = 10000 #len(testdata)  # Number of loaded testing samples
    fs = 16000  # Sampling rate of the samples
    segmentLength = 160
    sliceLength = int(totalSliceLength * fs / segmentLength) * segmentLength

    for i in tqdm(range(trainsize)):
        filename = DataSetPath + traindata[i]['audio_file_path']
        #v = VoiceActivityDetector(filename)
        #v.plot_detected_speech_regions()

        fs, train_sound_data = wavfile.read(filename) # Read wavfile to extract amplitudes

        _x_train = train_sound_data#.copy() # Get a mutable copy of the wavfile

        label = traindata[i]['is_hotword']  # Read label
        label_vec = np.full(int(sliceLength / segmentLength), 0)
        if label:
            is_audio = detect_audio(train_sound_data)
            is_audio_subsampled = is_audio[0::segmentLength]
            num_segments = int(sliceLength / segmentLength)
            if len(is_audio_subsampled) >= num_segments:
                is_audio_subsampled = is_audio_subsampled[:num_segments]
            else:
                is_audio_subsampled = np.pad(is_audio_subsampled, (num_segments - len(is_audio_subsampled), 0))
            #is_audio_subsampled.resize(int(sliceLength / segmentLength), refcheck=False) #append zeros
            label_vec = label_vec + is_audio_subsampled #overlay frames where label is true

        if len(_x_train) >= sliceLength:
            _x_train = _x_train[:sliceLength]
        else:
            _x_train = np.pad(_x_train, (sliceLength - len(_x_train),0))
        # _x_train.resize(sliceLength, refcheck=False)
        #
        # _x_train = _x_train.reshape(-1, int(segmentLength)) # Split slice into Segments with 0 overlap

        x_train_list.append(_x_train.astype(np.float32)) # Add segmented slice to training sample list, cast to float so librosa doesn't complain
        y_train_list.append(label_vec)

    for i in tqdm(range(testsize)):
        fs, test_sound_data = wavfile.read(DataSetPath + traindata[i]['audio_file_path'])  # Read wavfile to extract amplitudes

        _x_test = test_sound_data  # Get a mutable copy of the wavfile

        label_vec = np.full(int(sliceLength / segmentLength), 0)
        label = traindata[i]['is_hotword']  # Read label
        if label:
            is_audio = detect_audio(test_sound_data)
            is_audio_subsampled = is_audio[0::segmentLength]
            num_segments = int(sliceLength / segmentLength)
            if len(is_audio_subsampled) >= num_segments:
                is_audio_subsampled = is_audio_subsampled[:num_segments]
            else:
                is_audio_subsampled = np.pad(is_audio_subsampled, (num_segments - len(is_audio_subsampled), 0))
            #is_audio_subsampled.resize(int(sliceLength / segmentLength), refcheck=False)  # append zeros
            label_vec = label_vec + is_audio_subsampled  # overlay frames where label is true

        #_x_test.resize(sliceLength, refcheck=False)

        #_x_test = _x_test.reshape(-1, int(segmentLength))  # Split slice into Segments with 0 overlap
        if len(_x_test) >= sliceLength:
            _x_test = _x_test[:sliceLength]
        else:
            _x_test = np.pad(_x_test, (sliceLength - len(_x_test),0))
        x_test_list.append(list(_x_test.astype(
            np.float32)))  # Add segmented slice to training sample list, cast to float so librosa doesn't complain

        y_test_list.append(label_vec)


    x_train = tf.convert_to_tensor(np.asarray(x_train_list))
    y_train = tf.convert_to_tensor(np.asarray(y_train_list))

    x_test = tf.convert_to_tensor(np.asarray(x_test_list))
    y_test = tf.convert_to_tensor(np.asarray(y_test_list))
    return x_train, y_train, x_test, y_test
def load_data_fixed_length():
    x_train_list = []
    y_train_list = []

    x_test_list = []
    y_test_list = []

    totalSliceLength = 5 # Length to stuff the signals to, given in seconds

    trainsize = 10000#len(traindata) # Number of loaded training samples
    testsize = 10000#len(testdata) # Number of loaded testing samples



    fs = 16000 # Sampling rate of the samples
    segmentLength = 160 # Number of samples to use per segment

    sliceLength = int(totalSliceLength * fs / segmentLength)*segmentLength

    for i in tqdm(range(trainsize)):
        fs, train_sound_data = wavfile.read(DataSetPath+traindata[i]['audio_file_path']) # Read wavfile to extract amplitudes

        _x_train = train_sound_data.copy() # Get a mutable copy of the wavfile
        #_x_train.resize(sliceLength, refcheck=False) # Zero stuff the single to a length of sliceLength
        if len(_x_train) >= sliceLength:
            _x_train = _x_train[:sliceLength]
        else:
            _x_train = np.pad(_x_train, (sliceLength - len(_x_train),0))
     #   _x_train = _x_train.reshape(-1,int(segmentLength)) # Split slice into Segments with 0 overlap
        x_train_list.append(_x_train.astype(np.float32)) # Add segmented slice to training sample list, cast to float so librosa doesn't complain
        y_train_list.append(traindata[i]['is_hotword']) # Read label

    for i in tqdm(range(testsize)):
        fs, test_sound_data = wavfile.read(DataSetPath+testdata[i]['audio_file_path'])
        _x_test = test_sound_data.copy()
        #_x_test.resize(sliceLength, refcheck=False)
        if len(_x_test) >= sliceLength:
            _x_test = _x_test[:sliceLength]
        else:
            _x_test = np.pad(_x_test, (sliceLength - len(_x_test),0))
      #  _x_test = _x_test.reshape((-1,int(segmentLength)))
        x_test_list.append(_x_test.astype(np.float32))
        y_test_list.append(testdata[i]['is_hotword'])

    x_train = tf.convert_to_tensor(np.asarray(x_train_list))
    y_train = tf.expand_dims(tf.convert_to_tensor(np.asarray(y_train_list)), axis=-1)

    x_test = tf.convert_to_tensor(np.asarray(x_test_list))
    y_test = tf.expand_dims(tf.convert_to_tensor(np.asarray(y_test_list)), axis=-1)

    return x_train, y_train, x_test, y_test

def compute_mfccs(tensor):
    sample_rate = 16000.0
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    frame_length = 1024
    num_mfcc = 13

    stfts = tf.signal.stft(tensor, frame_length=frame_length, frame_step=frame_length, fft_length=frame_length)
    spectrograms = tf.abs(stfts)
    spectrograms = tf.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],-1))
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
      upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfcc]
    return tf.reshape(mfccs, (mfccs.shape[0],mfccs.shape[1],mfccs.shape[2],-1))