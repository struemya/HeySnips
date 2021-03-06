import json

import numpy as np
from python_speech_features.base import logfbank
from scipy.io import wavfile
from scipy.signal import medfilt
from tqdm import tqdm


class DatasetGenerator:
    """Dataset Generator for the HeySnips Dataset
    Args:
      dataset_path: path to the dataset (.wav files and .json)
      dataset_size: optional parameter to limit the number of samples, defaults to the full dataset
      sampling_frequency: sampling frequency of the incoming audio files
      type: Type of dataset: either train, test or dev
      num_feautures: number of features to be extracted from the audio
      time_step: offset between two frames
      window_size: window around time step that is used for the extraction of the features
      total_slice_length: length of a slice in seconds, one slice corresponds to one wav file, trancated or zero padded in front if not matching
      on_key_length: number of frames at the end of the keyword that are labelled as 1 (symmetrical around the end)
      """
    def __init__(self,
                 dataset_path,
                 dataset_size=None,
                 sampling_frequency=16000,
                 type='train',  # may also be 'dev, 'test'
                 num_features=20,
                 time_step=0.01,
                 window_size=0.025,
                 total_slice_length=3,
                 on_key_length=5120,
                 ):

      self.dataset_path = dataset_path
      self.dataset_size = dataset_size
      self.sampling_frequency = sampling_frequency
      self.type = type
      self.num_features = num_features
      self.time_step = time_step
      self.window_size = window_size
      self.total_slice_length = total_slice_length
      self.on_key_length = on_key_length

    def compute_dataset(self):
      """computes and saves the demanded dataset"""
      with open(self.dataset_path + self.type + ".json") as jsonfile:
        data_files = json.load(jsonfile)

      segmentLength = int(self.sampling_frequency * self.time_step)
      sliceLength = int(self.total_slice_length * self.sampling_frequency / segmentLength) * segmentLength
      if self.dataset_size is not None:
        num_samples = self.dataset_size
      else:
        num_samples = len(data_files)
      num_time_steps = int(sliceLength / (self.time_step * self.sampling_frequency) - 1)
      data = np.zeros((num_samples, num_time_steps, self.num_features), dtype='float32')
      labels = np.zeros((num_samples, num_time_steps), dtype='float32')

      for i in tqdm(range(num_samples)):
        data_sample, data_label = self._compute_sample(data_files[i], sliceLength, segmentLength)
        data[i] = data_sample
        labels[i] = data_label

      name = self.type + str(num_samples) + '_feat' + str(self.num_features) + '_slicelen' + str(self.total_slice_length) + '.npz'

      np.savez(name, data=data, labels=labels)
      return data, labels

    def _compute_sample(self, file, sliceLength, segmentLength):
      """computes the feature and label vector for single audio slice"""
      filename = self.dataset_path + file['audio_file_path']
      fs, sound_data = wavfile.read(filename)
      sound_data = sound_data.astype('float32')
      label = file['is_hotword']
      label_vec = np.full(int(sliceLength / segmentLength -1), 0, dtype='int8')
      if label:
        is_audio_subsampled = self._detect_audio(sound_data)[0::segmentLength] #detect audio and subsample to single label per segment
        num_segments = int(sliceLength / segmentLength)
        if len(is_audio_subsampled) >= num_segments:
          is_audio_subsampled = is_audio_subsampled[:num_segments] #truncate if longer than target segments
        else:
          is_audio_subsampled = np.pad(is_audio_subsampled, (num_segments - len(is_audio_subsampled), 0)) #pad if shorter than target segments
        label_vec = label_vec + is_audio_subsampled[1:]  # overlay frames where label is true

      if len(sound_data) >= sliceLength:
        sound_data = sound_data[:sliceLength]
      else:
        sound_data = np.pad(sound_data, (sliceLength - len(sound_data), 0))

      #compute log mel filterbank energies
      feature_vec = logfbank(sound_data, samplerate=self.sampling_frequency, winlen=self.window_size, winstep=self.time_step, nfilt=self.num_features)
      return feature_vec, label_vec

    def _detect_audio(self, input_tensor):
      """Detects the active audio region and labels the end of keyword with ones for self.on_key_length frames"""
      #constants
      median_filter_window = 151
      threshold = 5e-2

      energy_sqrt = abs(input_tensor)
      filtered_energy = medfilt(energy_sqrt, median_filter_window)
      max_filtered = max(filtered_energy)

      is_audio = filtered_energy > (max_filtered * threshold) #binary vector that says whether audio is considered active or off
      is_audio_index = np.nonzero(is_audio) #indices of active
      #last index that is considered active is max(is_audio_index[0]) -> add 1 label around this "end of keyword"
      beginning = max(is_audio_index[0]) - self.on_key_length // 2
      ending = max(is_audio_index[0]) + self.on_key_length // 2
      mask = np.zeros_like(is_audio, dtype='float32')
      mask[beginning:ending + 1] = True
      return mask

#set the path to the audio dataset (wav files)
DataSetPath = "/Users/yannickstrumpler/Documents/Studium/MLoMCU/Lab2/hey_snips_research_6k_en_train_eval_clean_ter/"

#set demanded parameters of the dataset generator
gen = DatasetGenerator(DataSetPath, num_features=26, type='dev')

#compute and save dataset
data, labels = gen.compute_dataset()