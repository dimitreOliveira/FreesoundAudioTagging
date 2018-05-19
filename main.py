import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from dataset import load_data, pre_process_data, output_submission, extract_features, make_input_data
import matplotlib.pyplot as plt


TRAIN_PATH = 'data/audio_train'
TEST_PATH = 'data/audio_test'
TRAIN_LABELS_PATH = 'data/train2.csv'
TEST_LABELS_PATH = 'data/sample_submission.csv'

train_labels_raw = pd.read_csv(TRAIN_LABELS_PATH)
test_labels = pd.read_csv(TEST_LABELS_PATH)
# train, test = load_data(TRAIN_PATH, TEST_PATH, train_labels_raw, test_labels)

train = make_input_data('data/train')
test = make_input_data('data/test')


train_dataset_size = len(train)
# The labels need to be one-hot encoded
train_labels = pd.get_dummies(train_labels_raw.label).as_matrix()
CLASSES = train_labels.shape[1]

# train = pre_process_data(train)
# test = pre_process_data(test)

# drop unwanted columns
# train_pre = train.drop(['Survived'], axis=1).as_matrix().astype(np.float)
# test_pre = test.as_matrix().astype(np.float)


# scale values
# standard_scaler = preprocessing.StandardScaler()
# train_pre = standard_scaler.fit_transform(train_pre)
# test_pre = standard_scaler.fit_transform(test_pre)

# data split
print(train.shape)
print(train_labels.shape)
X_train, X_valid, Y_train, Y_valid = train_test_split(train, train_labels, test_size=0.2, random_state=1)

# hyperparameters
input_layer = len(train)*2
output_layer = 2
num_epochs = 100
learning_rate = 0.01
train_size = 0.8
layers_dims = [input_layer, 500, 500, output_layer]


# print(submission_name)
