import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model import *
from methods import train_generator, chunker
from dataset import load_audio_file


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

input_shape = (input_length, 1)
nclasses = len(list_labels)

model = get_model2(input_shape, nclasses)

model.fit_generator(train_generator(tr_files, batch_size), steps_per_epoch=len(tr_files)//batch_size, epochs=1,
                    validation_data=train_generator(val_files, batch_size), validation_steps=len(val_files)//batch_size)

model.save_weights("model2.h5")

list_preds = []

for batch_files in tqdm(chunker(test_files, size=batch_size), total=len(test_files)//batch_size):
    batch_data = [load_audio_file(fpath) for fpath in batch_files]
    batch_data = np.array(batch_data)[:, :, np.newaxis]
    preds = model.predict(batch_data).tolist()
    list_preds += preds

score = model.evaluate(x=train_generator(tr_files, batch_size), y=train_generator(val_files, batch_size))
print('Test loss:', score[0])
print('Test accuracy:', score[1])

array_preds = np.array(list_preds)
list_labels = np.array(list_labels)
top_3 = list_labels[np.argsort(-array_preds, axis=1)[:, :3]]
pred_labels = [' '.join(list(x)) for x in top_3]
df = pd.DataFrame(test_files, columns=["fname"])
df['label'] = pred_labels
df['fname'] = df.fname.apply(lambda x: x.split("/")[-1])
df.to_csv("baseline.csv", index=False)
