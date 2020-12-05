import json
import numpy as np
from scipy.io import wavfile
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tqdm import tqdm
from get_dataset import load_data_fixed_length, compute_mfccs

x_train, y_train, x_test, y_test = load_data_fixed_length()

x_train_mfcc = compute_mfccs(x_train)
x_test_mfcc = compute_mfccs(x_test)

print(x_train_mfcc.shape)
print(x_test_mfcc.shape)
