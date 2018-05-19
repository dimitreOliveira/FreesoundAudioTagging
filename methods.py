import numpy as np
from random import shuffle
from dataset import load_audio_file
from test import file_to_int


def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def train_generator(list_files, batch_size=32):
    while True:
        shuffle(list_files)
        for batch_files in chunker(list_files, size=batch_size):
            batch_data = [load_audio_file(fpath) for fpath in batch_files]
            batch_data = np.array(batch_data)[:, :, np.newaxis]
            batch_labels = [file_to_int[fpath] for fpath in batch_files]
            batch_labels = np.array(batch_labels)

            yield batch_data, batch_labels