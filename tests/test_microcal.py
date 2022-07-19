"""CMEPDA Project: Unittest"""
## pylint: disable=C0103
## pylint: disable=C0413

import sys
import os
from pathlib import Path
import warnings
import unittest

import numpy as np

# Locally: setting path
#sys.path.insert(0, str(Path(os.getcwd()).parent)) # Get the absolute path to the parent dir.
#PACKAGE_NAME = "../microcal_classifier/"
#sys.path.insert(0, PACKAGE_NAME)

package_name = "microcal_classifier"
sys.path.insert(0, package_name)

from microcal_classifier.wavelethelper import read_img
from microcal_classifier.cnnhelper import read_imgs, count_labels, split_dataset
from microcal_classifier.wavelet_coeff import dwt_coeff_array


class TestMicroClassifier(unittest.TestCase):
    """Unit test for the microcal_classifier project."""

    def setUp(self):
        """
        Initialize the class.
        """
        warnings.simplefilter('ignore', ResourceWarning)

        # Locally: setting path
        #self.PATH = '/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/'
        #self.TRAIN_PATH = '/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/Train'
        #self.TEST_PATH = '/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/Test'

        self.PATH = '/dataset/IMAGES/Mammography_micro/'
        self.TRAIN_PATH = '/dataset/IMAGES/Mammography_micro/Train'
        self.TEST_PATH = '/dataset/IMAGES/Mammography_micro/Test'

    def test_read_img_mp(self):
        """Unit test for read img function with multiprocessing."""

        error_message_len = "Wrong lenght"

        train_path_0 = os.path.join(self.PATH, 'Train/0')
        _, y0_train = read_img(train_path_0)
        train_path_1 = os.path.join(self.PATH, 'Train/1')
        _, y1_train = read_img(train_path_1)
        test_path_0 = os.path.join(self.PATH, 'Test/0')
        _, y0_test = read_img(test_path_0)
        test_path_1 = os.path.join(self.PATH, 'Test/1')
        _, y1_test = read_img(test_path_1)

        self.assertEqual(len(y0_train), 330, error_message_len)
        self.assertEqual(len(y1_train), 306, error_message_len)
        self.assertEqual(len(y0_test), 84, error_message_len)
        self.assertEqual(len(y1_test), 77, error_message_len)


    def test_read_imgs(self):
        """Unit test for read img function (no multiprocessing)."""
        error_message_shape = "Wrong shape"

        X_train, y_train = read_imgs(self.TRAIN_PATH, [0, 1])
        X_test, y_test = read_imgs(self.TEST_PATH, [0, 1])

        self.assertEqual(X_train.shape, (636, 60, 60, 1), error_message_shape)
        self.assertEqual(X_test.shape, (161, 60, 60, 1), error_message_shape)
        self.assertEqual(y_train.shape, (636,), error_message_shape)
        self.assertEqual(y_test.shape, (161,) , error_message_shape)


    def test_count_labels(self):
        """Unittest for count the number of labels in the data set."""

        _, y_train = read_imgs(self.TRAIN_PATH, [0, 1])
        _, y_test = read_imgs(self.TEST_PATH, [0, 1])
        data = count_labels(y_train, y_test, verbose=False)

        self.assertIn('Train', data.keys(), 'Train not in dataframe')
        self.assertIn('Test', data.keys(), 'Test not in dataframe')
        self.assertEqual(data.shape, (2, 3), 'Shape not correct')


    def test_split_dataset(self):
        """Unit test for split the train dataset for train and validation set."""

        x_train_raw, y_train_raw = read_imgs(self.TRAIN_PATH, [0, 1])
        x_test_raw, y_test_raw = read_imgs(self.TEST_PATH, [0, 1])
        perc = 0.2
        message = "Split not valid."
        delta = 0.09

        _, _, y_train, y_val = split_dataset(X_train=x_train_raw,
                                             y_train=y_train_raw,
                                             X_test=x_test_raw,
                                             y_test=y_test_raw,
                                             perc=perc,
                                             verbose=False
                                             )

        self.assertAlmostEqual(perc, len(y_val)/len(y_train), None, message, delta)


    def test_dwt_coeff_array(self):
        """Unit test for calculation of weavelet coefficients of the 2DWT."""

        train_path_0 = os.path.join(self.PATH, 'Train/0')
        X0_train, _ = read_img(train_path_0)
        train_path_1 = os.path.join(self.PATH, 'Train/1')
        X1_train, _ = read_img(train_path_1)
        test_path_0 = os.path.join(self.PATH, 'Test/0')
        X0_test, _ = read_img(test_path_0)
        test_path_1 = os.path.join(self.PATH, 'Test/1')
        X1_test, _ = read_img(test_path_1)

        X_train = X0_train + X1_train
        X_test = X0_test + X1_test
        X_tot = X_test + X_train

        wavelet = 'db5'
        level = 4
        partial = True
        coefficients = []

        for i, _ in enumerate(X_tot):
            array = dwt_coeff_array(X_tot[i],
                                    wavelet=wavelet,
                                    level=level,
                                    partial=partial
                                    )
            coefficients.append(array)
        coefficients = np.array(coefficients)

        self.assertEqual(coefficients.shape, (797, 2430), "Wrong shape in wavelet coefficient.")


if __name__ == "__main__":
    unittest.main()
