import csv
import re
import os
import glob
import librosa
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import wavfile


def load_data(train_path, test_path, train_labels, test_labels):
    """
    method for data loading
    :param train_path: path for the train set file
    :param test_path: path for the test set file
    :return: a 'pandas' array for each set
    """

    train_path_h5 = 'data/train.h5'
    test_path_h5 = 'data/test.h5'








    if os.path.exists(train_path_h5):
        with h5py.File(train_path_h5, 'r') as hf:
            train_data = hf['train'][:]

    else:
        train_data = []
        # for file, label, input_type in tqdm(train_labels.as_matrix()):
        #     fs, x = wavfile.read('{}/{}'.format(train_path, file))
        #     times = np.arange(len(x)) / float(fs)
        #     clip = []
        #     for (start, end) in windows(times, 512*127):
        #         if len(times[start:end]) == 512*127:
        #             signal = times[start:end]
        #             clip.append(signal)
        #     train_data.append(clip)
        window_size_sec = 0.025
        window_shift_sec = 0.0125
        sample_rate = 44100
        for file, label, input_type in tqdm(train_labels.as_matrix()):
            data, sampling_rate = librosa.core.load('{}/{}'.format(train_path, file), sr=sample_rate, mono=True)
            # data = load_audio_file('{}/{}'.format(train_path, file))
            win_length = int(sample_rate * window_size_sec)
            hop_length = int(sample_rate * window_shift_sec)
            n_fft = win_length  # must be >= win_length
            spectrogram = librosa.core.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            # train_data.append(spectrogram)
            # print(data)
            print(spectrogram.shape)
            print(data.shape)
            # data = np.array(data)[:, :, np.newaxis]
            # print(data)
            train_data.append(spectrogram)

        # with h5py.File(train_path_h5, 'w') as hf:
        #     hf.create_dataset("train", data=train_data)


    test_data = []
    # if os.path.exists(test_path_h5):
    #     with h5py.File(test_path_h5, 'r') as hf:
    #         test_data = hf['x_test'][:]
    #
    # else:
    #     test_data = []
    #     for file, label in tqdm(test_labels.as_matrix()):
    #         fs, x = wavfile.read('{}/{}'.format(test_path, file))
    #         times = np.arange(len(x)) / float(fs)
    #         train_data.append(times)
    #
    #     with h5py.File(test_path_h5, 'w') as hf:
    #         hf.create_dataset("test", data=test_data)

    # print("number of training examples = " + str(train_data.shape[0]))
    # print("number of test examples = " + str(test_data.shape[0]))
    # print("train shape: " + str(train_data.shape))
    # print("test shape: " + str(test_data.shape))
    print(len(train_data))
    print(len(test_data))

    return train_data, test_data
    return np.asarray(train_data), np.asarray(test_data)


def output_submission(test_ids, predictions, id_column, predction_column, file_name):
    """
    :param test_ids: vector with test dataset ids
    :param predictions: vector with test dataset predictions
    :param id_column: name of the output id column
    :param predction_column: name of the output predction column
    :param file_name: string for the output file name
    :return: output a csv with ids ands predictions
    """

    print('Outputting submission...')
    with open('submissions/' + file_name, 'w') as submission:
        writer = csv.writer(submission)
        writer.writerow([id_column, predction_column])
        for test_id, test_prediction in zip(test_ids, np.argmax(predictions, 1)):
            writer.writerow([test_id, test_prediction])
    print('Output complete')


def pre_process_data(df):
    """
    Perform a number of pre process functions on the data set
    :param df: pandas data frame
    :return: updated data frame
    """
    # setting `passengerID` as Index since it wont be necessary for the analysis
    df = df.set_index("PassengerId")

    # convert 'Sex' values
    df['gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    # We see that 2 passengers embarked data is missing, we fill those in as the most common Embarked value
    df.loc[df.Embarked.isnull(), 'Embarked'] = df['Embarked'].mode()[0]

    # Replace missing age values with median ages by gender
    for gender in df['gender'].unique():
        median_age = df[(df['gender'] == gender)].Age.median()
        df.loc[(df['Age'].isnull()) & (df['gender'] == gender), 'Age'] = median_age

    # convert 'gender' values to new columns
    df = pd.get_dummies(df, columns=['gender'])

    # convert 'Embarked' values to new columns
    df = pd.get_dummies(df, columns=['Embarked'])

    # bin Fare into five intervals with equal amount of values
    df['Fare-bin'] = pd.qcut(df['Fare'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # bin Age into seven intervals with equal amount of values
    # ('baby','child','teenager','young','mid-age','over-50','senior')
    bins = [0, 4, 12, 18, 30, 50, 65, 100]
    age_index = (1, 2, 3, 4, 5, 6, 7)
    df['Age-bin'] = pd.cut(df['Age'], bins, labels=age_index).astype(int)

    # create a new column 'family' as a sum of 'SibSp' and 'Parch'
    df['family'] = df['SibSp'] + df['Parch'] + 1
    df['family'] = df['family'].map(lambda x: 4 if x > 4 else x)

    # create a new column 'FTicket' as the first character of the 'Ticket'
    df['FTicket'] = df['Ticket'].map(lambda x: x[0])
    # combine smaller categories into one
    df['FTicket'] = df['FTicket'].replace(['W', 'F', 'L', '5', '6', '7', '8', '9'], '4')
    # convert 'FTicket' values to new columns
    df = pd.get_dummies(df, columns=['FTicket'])

    # get titles from the name
    df['title'] = df.apply(lambda row: re.split('[,.]+', row['Name'])[1], axis=1)

    # convert titles to values
    df['title'] = df['title'].map({' Capt': 'Other', ' Master': 'Master', ' Mr': 'Mr', ' Don': 'Other',
                                   ' Dona': 'Other', ' Lady': 'Other', ' Col': 'Other', ' Miss': 'Miss',
                                   ' the Countess': 'Other', ' Dr': 'Other', ' Jonkheer': 'Other', ' Mlle': 'Other',
                                   ' Sir': 'Other', ' Rev': 'Other', ' Ms': 'Other', ' Mme': 'Other', ' Major': 'Other',
                                   ' Mrs': 'Mrs'})
    # convert 'title' values to new columns
    df = pd.get_dummies(df, columns=['title'])

    df = df.drop(['Name', 'Ticket', 'Cabin', 'Sex', 'Fare', 'Age'], axis=1)

    return df


def mini_batches(train_set, train_labels, mini_batch_size):
    """
    Generate mini batches from the data set (data and labels)
    :param train_set: data set with the examples
    :param train_labels: data set with the labels
    :param mini_batch_size: mini batch size
    :return: mini batches
    """
    set_size = train_set.shape[0]
    batches = []
    num_complete_minibatches = set_size // mini_batch_size

    for k in range(0, num_complete_minibatches):
        mini_batch_x = train_set[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_y = train_labels[k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if set_size % mini_batch_size != 0:
        mini_batch_x = train_set[(set_size - (set_size % mini_batch_size)):]
        mini_batch_y = train_labels[(set_size - (set_size % mini_batch_size)):]
        mini_batch = (mini_batch_x, mini_batch_y)
        batches.append(mini_batch)

    return batches


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size // 2)


def extract_features(parent_dir, sub_dirs, file_ext="*wav", bands=128, frames=128):
    window_size = 512*127
    log_specgrams = []
    labels = []
    ITJ = 0
    for l, sub_dir in enumerate(sub_dirs):
        PTJ = 1
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            label = ITJ
            for (start, end) in windows(sound_clip, window_size):
                if len(sound_clip[start:end]) == window_size:
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                    logspec = librosa.amplitude_to_db(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)
                    labels.append(label)
            print(PTJ+ITJ*270, ITJ)
            PTJ = PTJ+1
        ITJ = ITJ+1
    log_specgrams = np.array(log_specgrams)
    print(log_specgrams.shape)
    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames)
    features = log_specgrams
    return np.array(features)


import wave
import struct


def parse_wave_python(filename):
    with wave.open(filename, 'rb') as wave_file:
        sample_rate = wave_file.getframerate()
        length_in_seconds = wave_file.getnframes() / sample_rate

        first_sample = struct.unpack(
            '<h', wave_file.readframes(1))[0]
        second_sample = struct.unpack(
            '<h', wave_file.readframes(1))[0]
    print('''
        Parsed {filename}
        -----------------------------------------------
        Channels: {num_channels}
        Sample Rate: {sample_rate}
        First Sample: {first_sample}
        Second Sample: {second_sample}
        Length in Seconds: {length_in_seconds}'''.format(
        filename=filename,
        num_channels=wave_file.getnchannels(),
        sample_rate=wave_file.getframerate(),
        first_sample=first_sample,
        second_sample=second_sample,
        length_in_seconds=length_in_seconds))


# parse_wave_python('data/audio_train/00ad7068.wav')

import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

def parse_wave_tf(filename):
    audio_binary = tf.read_file(filename)
    desired_channels = 1
    wav_decoder = contrib_audio.decode_wav(
        audio_binary,
        desired_channels=desired_channels)

    with tf.Session() as sess:
        sample_rate, audio = sess.run([
            wav_decoder.sample_rate,
            wav_decoder.audio])
        first_sample = audio[0][0] * (1 << 15)
        second_sample = audio[1][0] * (1 << 15)
        print('''
        Parsed {filename}
        -----------------------------------------------
        Channels: {desired_channels}
        Sample Rate: {sample_rate}
        First Sample: {first_sample}
        Second Sample: {second_sample}
        Length in Seconds: {length_in_seconds}'''.format(
        filename=filename,
        desired_channels=desired_channels,
        sample_rate=sample_rate,
        first_sample=first_sample,
        second_sample=second_sample,
        length_in_seconds=len(audio) / sample_rate))



def read_wav_file(fname):
    _, wav = wavfile.read(fname)
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav



window_size_sec = 0.025
window_shift_sec = 0.0125
sample_rate = 8000
data, sampling_rate = librosa.core.load('data/audio_train/00ad7068.wav', sr=sample_rate, mono=True)
win_length = int(sample_rate * window_size_sec)
hop_length = int(sample_rate * window_shift_sec)
n_fft = win_length  # must be >= win_length
spectrogram = librosa.core.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

input_length = 16000 * 2

batch_size = 32


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 0.0001)
    return data - 0.5


def load_audio_file(file_path, input_length=input_length):
    data = librosa.core.load(file_path, sr=16000)[0]  # , sr=16000
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

    data = audio_norm(data)
    return data





from scipy import signal
import os

RATE = 44100

MAX_FRAME = int(RATE * 30)
MIN_FRAME = int(RATE * 0.3)
NORM_FACTOR = 1.0 / 2 ** 16.0

MAX_INPUT = int(MAX_FRAME / MIN_FRAME)
FREQUENCY_BINS = int(MIN_FRAME / 2) + 1


def make_tensor(fname):
    """
    Brief
    -----
    Creates a 3D tensor from an audio file

    Params
    ------
    fname: name of the file to pre-process

    Returns
    -------
    A 3D tensor of the audio file as an np.array
    """
    rate, data = wavfile.read(fname)
    data = np.array([(e * NORM_FACTOR) * 2 for e in data])
    # output = np.zeros((FREQUENCY_BINS, MAX_INPUT, 2))
    output = np.zeros((FREQUENCY_BINS, MAX_INPUT))
    freqs, times, specs = signal.spectrogram(data,
                                             fs=RATE,
                                             window="boxcar",
                                             nperseg=MIN_FRAME,
                                             noverlap=0,
                                             detrend=False,
                                             mode='complex')
    output[:, :specs.shape[1]] = np.real(specs)
    # output[:, :specs.shape[1], 0] = np.real(specs)
    # output[:, :specs.shape[1], 1] = np.imag(specs)
    return output


def make_input_data(audio_dir, fnames=None):
    """
    Brief
    -----
    Pre-process a list of file or a full directory.

    Params
    ------
    audio_dir: str
        Directory where files are stored
    fnames: str or None
        List of filenames to preprocess. If None: pre-process the full directory.

    Returns
    -------
    A 4D tensor (last dimension refers to observations) as an np.array
    """
    if fnames is None:
        fnames = os.listdir(audio_dir)
    else:
        fnames = [fname + '.wav' for fname in fnames]
    print(FREQUENCY_BINS)
    print(MAX_INPUT)
    print(len(fnames))
    # output = np.zeros((FREQUENCY_BINS, MAX_INPUT, 2, len(fnames)))
    output = np.zeros((FREQUENCY_BINS, MAX_INPUT, len(fnames)))
    print('************')
    i = 0
    for fname in fnames:
        full_path = os.path.join(audio_dir, fname)
        # output[:, :, :, i] = make_tensor(full_path)
        output[:, :, i] = make_tensor(full_path)
        i = i + 1
    return output












import glob
import os
import librosa
import numpy as np


def windows2(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


def extract_features2(parent_dir, sub_dirs, file_ext="*.wav", bands=60, frames=41):
    window_size = 512 * (frames - 1)
    log_specgrams = []
    for l, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            sound_clip, s = librosa.load(fn)
            for (start, end) in windows2(sound_clip, window_size):
                if len(sound_clip[start:end]) == window_size:
                    signal = sound_clip[start:end]
                    melspec = librosa.feature.melspectrogram(signal, n_mels=bands)
                    logspec = librosa.amplitude_to_db(melspec)
                    logspec = logspec.T.flatten()[:, np.newaxis].T
                    log_specgrams.append(logspec)

    log_specgrams = np.asarray(log_specgrams).reshape(len(log_specgrams), bands, frames, 1)
    features = np.concatenate((log_specgrams, np.zeros(np.shape(log_specgrams))), axis=3)
    for i in range(len(features)):
        features[i, :, :, 1] = librosa.feature.delta(features[i, :, :, 0])

    return np.array(features)



import keras
from keras.layers import Activation, Dense, Dropout, Conv2D, \
                         Flatten, MaxPooling2D, LSTM
from keras.models import Sequential
import librosa
import librosa.display
import numpy as np
import pandas as pd
import random


y, sr = librosa.load('data/audio_train/00ad7068.wav', duration=2.97)
ps = librosa.feature.melspectrogram(y=y, sr=sr)
print(ps.shape)


D = []

for file, label, input_type in pd.read_csv('data/train2.csv').as_matrix():
    y, sr = librosa.load('{}/{}'.format('data/audio_train', file), duration=2.97)
    ps = librosa.feature.melspectrogram(y=y, sr=sr)
    if ps.shape != (128, 128):
        continue
    D.append((ps, label))


print("Number of samples: ", len(D))


dataset = D
random.shuffle(dataset)

train = dataset[:8]
test = dataset[8:]

X_train, y_train = zip(*train)
X_test, y_test = zip(*test)

# Reshape for CNN input
X_train = np.array([x.reshape((128, 128, 1)) for x in X_train])
X_test = np.array([x.reshape((128, 128, 1)) for x in X_test])

X_train = np.array([x for x in X_train])
X_test = np.array([x for x in X_test])

# One-Hot encoding for classes
y_train = pd.get_dummies(y_train).as_matrix()
y_test = pd.get_dummies(y_test).as_matrix()
y_train = np.array(keras.utils.to_categorical(y_train, 41))
y_test = np.array(keras.utils.to_categorical(y_test, 41))


model = Sequential()
# input_shape = (128, 128, 1)
input_shape = (128, 128)
model.add(Conv2D(24, (5, 5), strides=(1, 1), input_shape=input_shape))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))
model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(MaxPooling2D((4, 2), strides=(4, 2)))
model.add(Activation('relu'))
model.add(Conv2D(48, (5, 5), padding="valid"))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(rate=0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(41))
model.add(Activation('softmax'))
print(model.summary())
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(x=X_train, y=y_train, epochs=12, batch_size=1, validation_data=(X_test, y_test))
# model.fit(x=X_train, y=y_train, batch_size=2, epochs=2, validation_split=0.2)


model = Sequential()
model.add(LSTM(60,  dropout=0.5, recurrent_dropout=0.5, input_shape=(128, 128)))
model.add(Dense(41, activation='softmax'))
print(model.summary())
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=['accuracy'])
model.fit(x=X_train, y=y_train, epochs=12, batch_size=1, validation_data=(X_test, y_test))

score = model.evaluate(x=X_test, y=y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

