from sklearn.model_selection import train_test_split
import tensorflow as tf

import numpy as np
import pandas as pd


BATCH_SIZE = 8
SHUFFLE_BUFFER_SIZE = 100


def dataset(data):
    if data == 'DNA':
        x_train, y_train = np.load("D:/DL/UiProject/data/DNA/X_train.npy"), \
                           np.load("D:/DL/UiProject/data/DNA/Y_train.npy")
        x_test, y_test = np.load("D:/DL/UiProject/data/DNA/X_test.npy"), np.load("D:/DL/UiProject/data/DNA/Y_test.npy")
    elif data == 'covid':
        file = pd.read_csv("D:/DL/UiProject/data/covid/covid_and_healthy_spectra.csv")
        x_train = []
        y_train = []
        for i in range(0, file.shape[0]):
            x_train.append(file.iloc[i][:-1])
            if file.iloc[i][-1] == 'Healthy':
                y_train.append(0)
            else:
                y_train.append(1)
    elif data == 'cell':
        1
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    if data == 'DNA':
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(BATCH_SIZE)
    else:
        test_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        test_dataset = test_dataset.batch(BATCH_SIZE)
    return train_dataset, val_dataset, test_dataset, len(train_dataset)
