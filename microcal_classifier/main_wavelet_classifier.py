"""CMEPDA Project: Image binary classification using wavelet transform.

This python script evaluates the performance of the most common methods
utilized in machine learning to binary classification on a
microcalcification image dataset.
The mammographic images containing either microcalcifications
or normal tissue are represented in terms of wavelet coefficients.

The most common methods in ML to binary classification:

- Support Vector Machines
- Naive Bayes
- Nearest Neighbor
- Decision Trees
- Logistic Regression
- Neural Networks
- Random Forest
- Multi-layer Perceptron

The binary classifiers are evaluate by means the following parameters:

- Accuracy
- Precision
- Recall
- AUC

"""

import os
import glob
import multiprocessing as mp
from pathlib import Path
import sys

from PIL import Image
import skimage
from skimage import img_as_float
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from wavelet_coeff import dwt_coeff_array
from wavelethelper import read_img

sys.path.insert(0, str(Path(os.getcwd()).parent))


if __name__ == '__main__':

    #==================================================
    # STEP 1: LOAD IMAGE DATA SET
    #==================================================
    PATH = '/home/lorenzomarini/Desktop/DATASETS_new/IMAGES/Mammography_micro/' # Image path

    TRAIN_PATH_0 = os.path.join(PATH, 'Train/0')
    X0_train, y0_train = read_img(TRAIN_PATH_0)

    TRAIN_PATH_1 = os.path.join(PATH, 'Train/1')
    X1_train, y1_train = read_img(TRAIN_PATH_1)

    TEST_PATH_0 = os.path.join(PATH, 'Test/0')
    X0_test, y0_test = read_img(TEST_PATH_0)

    TEST_PATH_1 = os.path.join(PATH, 'Test/1')
    X1_test, y1_test = read_img(TEST_PATH_1)

    # Creates a list containing both 0 and 1 labelled images and
    # an array containing both 0 and 1 labels associated to the images.
    X_train = X0_train + X1_train
    y_train = np.concatenate((y0_train, y1_train))

    X_test = X0_test + X1_test
    y_test = np.concatenate((y0_test, y1_test))

    # NB: To concatenate two lists, I have to add to each other.
    X_tot = X_test + X_train # Images
    labels = np.concatenate((y_test, y_train)) # Labels

    #==================================================
    # STEP 2: COMPUTE THE WAVELET COEFFICIENTS
    #==================================================
    WAVELET = 'db5'
    LEVEL = 4
    partial = True

    # Defines a list to put all coefficients in
    coefficients = []

    # Appends to the previous list the coefficients obtained from wavelet decomposition
    for i, image in enumerate(X_tot):
        array = dwt_coeff_array(X_tot[i],
                                wavelet=WAVELET,
                                level=LEVEL,
                                partial=partial
                                )
        coefficients.append(array)

    # Converts the list to a numpy array
    coefficients = np.array(coefficients)
    print(coefficients.shape)
    
    #==================================================
    # STEP 3: BINARY CLASSIFICATION
    #==================================================

    # Step 3.1: Define X (wavelet coefficients) and y (labels) variables
    #--------------------------------------------------------------------
    X = coefficients
    y = labels


    # Step 3.2: Split the dataset into training and testing sets
    #-----------------------------------------------------------
    # 75% of data for training and 25% for testing.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    
    # Step 3.3: Normalize the data for numerical stability
    #-----------------------------------------------------
    ss_train = StandardScaler()
    X_train = ss_train.fit_transform(X_train)

    ss_test = StandardScaler()
    X_test = ss_test.fit_transform(X_test)


    # STEP 3.4: Initializing each binary classifier
    #----------------------------------------------
    models = {}

    # Logistic Regression
    models['Logistic Regression'] = LogisticRegression()

    # Support Vector Machines
    models['Support Vector Machines'] = LinearSVC()

    # Decision Trees   
    models['Decision Trees'] = DecisionTreeClassifier()

    # Random Forest
    models['Random Forest'] = RandomForestClassifier()

    # Naive Bayes
    models['Naive Bayes'] = GaussianNB()

    # K-Nearest Neighbors
    models['K-Nearest Neighbor'] = KNeighborsClassifier()

    # Multi-layer Perceptron classifier
    models['MLPClassifier'] = MLPClassifier()


    # STEP 3.5: Performance evaluation of each binary classifier
    #-----------------------------------------------------------
    # Once the models have been initialized, loop over each one,
    # train it, make predictions, calculate metrics,
    # and store each result in a dictionary.

    #accuracy, precision, recall, roc_auc = {}, {}, {}, {}
    accuracy, precision, recall = {}, {}, {}

    for key in models.keys():

        # Fit the classifier
        models[key].fit(X_train, y_train)

        # Make predictions
        predictions = models[key].predict(X_test)

        # Calculate metrics
        accuracy[key] = accuracy_score(predictions, y_test)
        precision[key] = precision_score(predictions, y_test)    
        recall[key] = recall_score(predictions, y_test)
        #roc_auc[key] = roc_auc_score(predictions, y_test)


    # Use pandas to view all the stored metrics as a table
    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy',
                                                          'Precision',
                                                          'Recall',
                                                          #'AUC'
                                                          ])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()
    #df_model['AUC'] = roc_auc.values()

    print(df_model)

    # LaTeX
    df = pd.DataFrame(dict(df_model))
    print(df.to_latex(index=False))

    # Plot a bar chart to compare the classifiers' performance:
    ax = df_model.plot.barh(figsize=(8, 5))
    ax.legend(
        ncol=len(models.keys()), 
        bbox_to_anchor=(0, 1), 
        loc='lower left', 
        prop={'size': 15}
    )
    plt.tight_layout()
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.show()
    
    #==================================================
    # STEP 4: K-FOLD CROSS VALIDATION 
    #==================================================
