import pandas as pd
import numpy as np

class Utilities(object):

    # Reads csv and returns data and labels
    def get_data_from_csv(self, csv_file, label_header_name):
        data_frame = pd.read_csv(csv_file)
        X = data_frame.drop(label_header_name, axis = 1)
        Y = data_frame[label_header_name]
        X, Y = np.array(X), np.array(Y)
        return X.T, Y.reshape(1, -1)

    def get_one_hot_encoding(self, array, num_classes):
        return np.squeeze(np.eye(num_classes)[array]).T

    def normalize(self, array):
        return array / 255.0

utils = Utilities()
X, Y = utils.get_data_from_csv('fashion-mnist_test.csv', 'label')
X = utils.normalize(X)
print (X)