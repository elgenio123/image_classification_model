import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical

# Function to unpickle the CIFAR-10 dataset files
def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data

def train_test_data(cifar_10_folder_path):
    # Load training data and labels
    train_data = []
    train_labels = []
    for i in range(1, 6):  # CIFAR-10 is split into 5 training batches
        file_path = f'{cifar_10_folder_path}/data_batch_{i}'
        batch_data = unpickle(file_path)
        train_data.append(batch_data[b'data'])
        train_labels += batch_data[b'labels']

    # Convert the list of arrays to a single NumPy array
    x_train = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.array(train_labels)

    # Load test data and labels
    test_data = unpickle(f'{cifar_10_folder_path}/test_batch')
    x_test = test_data[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_data[b'labels'])

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # One-hot encode the labels
    y_train = to_categorical(train_labels, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test
