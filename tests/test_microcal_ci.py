"""CMEPDA Project: Unittest"""
## pylint: disable=C0103
## pylint: disable=C0413

import sys
import os
from pathlib import Path
import warnings
import unittest

import numpy as np

sys.path.insert(0, str(Path(os.getcwd()).parent)) # Get the absolute path to the parent dir.
package_name = "../microcal_classifier/"
sys.path.insert(0, package_name)


from microcal_classifier.cnnhelper import count_labels, split_dataset
from microcal_classifier.wavelet_coeff import dwt_coeff_array


class TestMicroClassifier(unittest.TestCase):
    """Unit test for the microcal_classifier project."""


    def setUp(self):
        """
        Initialize the class.
        """
        warnings.simplefilter('ignore', ResourceWarning)

        self.x_train = np.random.rand(6, 6)
        self.x_test = np.random.rand(6, 6)
        self.y_train = np.array([0,0,1,1,0,0])
        self.y_test = np.array([1,1,1,0,0,0])
        self.x_train_wavelet = np.random.rand(60, 60)
        self.x_test_wavelet = np.random.rand(60, 60)


    def test_count_labels(self):
        """Unittest for count the number of labels in the data set."""

        data = count_labels(self.y_train, self.y_test, verbose=False)

        self.assertIn('Train', data.keys(), 'Train not in dataframe')
        self.assertIn('Test', data.keys(), 'Test not in dataframe')
        self.assertEqual(data['Train'][0], 4 , 'Number of 0 in train not correct')
        self.assertEqual(data['Test'][0], 3 , 'Number of 0 in test not correct')
        self.assertEqual(data['Train'][1], 2 , 'Number of 1 in train not correct')
        self.assertEqual(data['Test'][1], 3 , 'Number of 1 in test not correct')
        self.assertEqual(data.shape, (2, 3), 'Shape not correct')


    def test_split_dataset(self):
        """Unit test for split the train dataset for train and validation set."""

        perc = 0.2
        message = "Split not valid."
        delta = 0.3

        _, _, y_train, y_val = split_dataset(X_train=self.x_train,
                                             y_train=self.y_train,
                                             X_test=self.x_test,
                                             y_test=self.y_test,
                                             perc=perc,
                                             verbose=False
                                             )

        self.assertAlmostEqual(perc, len(y_val)/len(y_train), None, message, delta)


    def test_dwt_coeff_array(self):
        """Unit test for calculation of weavelet coefficients of the 2DWT."""

        x_tot = [self.x_train_wavelet, self.x_test_wavelet]

        wavelet = 'db5'
        level = 4
        partial = True
        coefficients = []

        for i, _ in enumerate(x_tot):
            array = dwt_coeff_array(x_tot[i],
                                    wavelet=wavelet,
                                    level=level,
                                    partial=partial
                                    )
            coefficients.append(array)
        coefficients = np.array(coefficients)

        self.assertEqual(coefficients.shape, (2, 2430), "Wrong shape in wavelet coefficient.")


if __name__ == "__main__":
    unittest.main()
