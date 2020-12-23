import os
import numpy as np

"""Loads the demanded precomputed dataset"""

def get_dataset(data_root, num_feat, slice_length, type='train', return_sequences=True):
    if type == 'train':
        name = "train50659_feat{}_slicelen{}.npz".format(num_feat, slice_length)
    elif type == 'dev':
        name = "dev22665_feat{}_slicelen{}.npz".format(num_feat, slice_length)
    elif type == 'test':
        name = "test23072_feat{}_slicelen{}.npz".format(num_feat, slice_length)
    else:
        raise NotImplementedError(
            "Dataset with the demanded parameters is not available, use compute_dataset.py to precompute it")

    dict = np.load(os.path.join(data_root, name))
    data = dict['data']
    labels = dict['labels']

    if not return_sequences:
        # if return_sequences is False we just want a single label per audio slice,
        # hence we just return the max over all labels in the time series
        labels = np.max(labels, axis=-1)
    return data, labels
