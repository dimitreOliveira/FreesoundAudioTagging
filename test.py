from keras import optimizers, losses, activations, models
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, Convolution2D, MaxPool2D, \
    Flatten, regularizers, LSTM, BatchNormalization, TimeDistributed, GRU, Activation
import librosa
import glob
import pandas as pd
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data - min_data) / (max_data - min_data + 0.0001)
    return data - 0.5


def load_audio_file(file_path, input_length=32000):
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


def get_model():
    nclass = len(list_labels)
    inp = Input(shape=(input_length, 1))
    img_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=16)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu)(img_1)
    dense_1 = Dense(1028, activation=activations.relu)(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()

    return model


def get_model2():
    nclass = len(list_labels)
    inp = Input(shape=(input_length, 1))

    img_1 = Convolution1D(16, kernel_size=2, activation=activations.relu, padding="valid")(inp)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Convolution1D(32, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Convolution1D(32, kernel_size=9, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=4)(img_1)
    img_1 = Flatten()(img_1)
    dense_1 = Dense(500, activation=activations.relu)(img_1)
    dense_1 = Dropout(rate=0.5)(dense_1)
    dense_1 = Dense(nclass, activation=activations.softmax)(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()

    return model


def get_model3():
    nclass = len(list_labels)
    inp = Input(shape=(input_length, 1))

    # X = Convolution1D(196, kernel_size=15, strides=4)(inp)  # CONV1D
    x = Convolution1D(64, kernel_size=15, strides=4)(inp)  # CONV1D
    x = BatchNormalization()(x)  # Batch normalization
    x = Activation('relu')(x)  # ReLu activation
    x = Dropout(0.5)(x)  # dropout (use 0.8)
    # X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    x = GRU(units=16, return_sequences=True)(x)  # GRU (use 128 units and return the sequences)
    x = Dropout(0.8)(x)  # dropout (use 0.8)
    x = BatchNormalization()(x)  # Batch normalization
    # X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    x = GRU(units=16, return_sequences=True)(x)  # GRU (use 128 units and return the sequences)
    x = Dropout(0.5)(x)  # dropout (use 0.8)
    x = BatchNormalization()(x)  # Batch normalization
    x = Dropout(0.5)(x)  # dropout (use 0.8)
    x = Flatten()(x)
    x = Dense(nclass, activation=activations.softmax)(x)
    # X = TimeDistributed(Dense(nclass, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    model = models.Model(inputs=inp, outputs=x)
    opt = optimizers.Adam(0.0001)

    model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    model.summary()

    return model


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


input_length = 16000 * 2
batch_size = 64

train_files = glob.glob("data/audio_train/*.wav")
test_files = glob.glob("data/audio_test/*.wav")
train_labels = pd.read_csv("data/train.csv")

train_files = [x.replace('\\', '/') for x in train_files]
test_files = [x.replace('\\', '/') for x in test_files]

file_to_label = {"data/audio_train/"+k: v for k, v in zip(train_labels.fname.values, train_labels.label.values)}

list_labels = sorted(list(set(train_labels.label.values)))
label_to_int = {k: v for v, k in enumerate(list_labels)}
int_to_label = {v: k for k, v in label_to_int.items()}
file_to_int = {k: label_to_int[v] for k, v in file_to_label.items()}


tr_files, val_files = train_test_split(train_files, test_size=0.1)

model = get_model2()

model.fit_generator(train_generator(tr_files, batch_size), steps_per_epoch=len(tr_files)//batch_size, epochs=1,
                    validation_data=train_generator(val_files, batch_size), validation_steps=len(val_files)//batch_size)


model.save_weights("baseline_cnn.h5")


list_preds = []


for batch_files in tqdm(chunker(test_files, size=batch_size), total=len(test_files)//batch_size):
    batch_data = [load_audio_file(fpath) for fpath in batch_files]
    batch_data = np.array(batch_data)[:, :, np.newaxis]
    preds = model.predict(batch_data).tolist()
    list_preds += preds


array_preds = np.array(list_preds)
list_labels = np.array(list_labels)
top_3 = list_labels[np.argsort(-array_preds, axis=1)[:, :3]]
pred_labels = [' '.join(list(x)) for x in top_3]
df = pd.DataFrame(test_files, columns=["fname"])
df['label'] = pred_labels
df['fname'] = df.fname.apply(lambda x: x.split("/")[-1])
df.to_csv("baseline.csv", index=False)
