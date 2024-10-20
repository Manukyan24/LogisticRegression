import numpy as np
import h5py
from model import LogisticRegression


#### This function was provided with dataset
def load_data():
    train_dataset = h5py.File("data/catvnocat_dataset/train_catvnoncat.h5", "r")
    train_set_x_orig = np.array(
        train_dataset["train_set_x"][:]
    )  # your train set features
    train_set_y_orig = np.array(
        train_dataset["train_set_y"][:]
    )  # your train set labels

    test_dataset = h5py.File("data/catvnocat_dataset/test_catvnoncat.h5", "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


####### -------------------------------- #######


X_train, Y_train, X_test, Y_test, classes = load_data()

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
Y_train = Y_train.T
Y_test = Y_test.T


model = LogisticRegression(learning_rate=0.001, epochs=1000)
model.fit(X_train, Y_train, X_test, Y_test)
