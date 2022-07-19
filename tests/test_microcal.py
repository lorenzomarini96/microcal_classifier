"""CMEPDA Project: Unittest"""
import sys
import os
from pathlib import Path
import warnings
import unittest

from sklearn.model_selection import train_test_split

# Get the absolute path to the parent dir.
sys.path.insert(0, str(Path(os.getcwd()).parent))

package_name = "../microcal_classifier/"

# setting path
#sys.path.append('../microcal_classifier')

sys.path.insert(0, package_name)

from microcal_classifier.wavelethelper import read_img
from microcal_classifier.cnnhelper import read_imgs, count_labels, split_dataset


class TestMicroClassifier(unittest.TestCase):
    """Unittest for the microcal_classiofier project."""

    def setUp(self):
        """
        Initialize the class.
        """
        warnings.simplefilter('ignore', ResourceWarning)
        
        self.PATH = '/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/'
        self.TRAIN_PATH = '/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/Train'
        self.TEST_PATH = '/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/Test'


    def test_read_img_mp(self):
        """Unittest for read img function with multiprocessing."""
        
        # error message in case if test case got failed
        error_message_len = "Wrong lenght"
        TRAIN_PATH_0 = os.path.join(self.PATH, 'Train/0')
        X0_train, y0_train = read_img(TRAIN_PATH_0)
        TRAIN_PATH_1 = os.path.join(self.PATH, 'Train/1')
        X1_train, y1_train = read_img(TRAIN_PATH_1)
        TEST_PATH_0 = os.path.join(self.PATH, 'Test/0')
        X0_test, y0_test = read_img(TEST_PATH_0)
        TEST_PATH_1 = os.path.join(self.PATH, 'Test/1')
        X1_test, y1_test = read_img(TEST_PATH_1)

        self.assertEqual(len(y0_train), 330, error_message_len)
        self.assertEqual(len(y1_train), 306, error_message_len)
        self.assertEqual(len(y0_test), 84, error_message_len)
        self.assertEqual(len(y1_test), 77, error_message_len)
    

    def test_read_imgs(self):
        """Unittest for read img function (no multiprocessing)."""
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
        """Unittest for split the train dataset for train and validation set."""

        X_TRAIN_raw, y_TRAIN_raw = read_imgs(self.TRAIN_PATH, [0, 1])
        X_TEST_raw, y_TEST_raw = read_imgs(self.TEST_PATH, [0, 1])
        PERC = 0.2
        message = "Split not valid."
        delta = 0.09
    
        _, _, y_TRAIN, y_VAL = split_dataset(X_train=X_TRAIN_raw,
                                             y_train=y_TRAIN_raw,
                                             X_test=X_TEST_raw,
                                             y_test=y_TEST_raw,
                                             perc=PERC,
                                             verbose=False
                                             )

        self.assertAlmostEqual(PERC, len(y_VAL)/len(y_TRAIN), None, message, delta)





if __name__ == "__main__":
    unittest.main()
