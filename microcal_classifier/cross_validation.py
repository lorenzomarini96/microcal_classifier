"""K-Cross validation procedure.
 The StratifiedKFold cross-validation technique is a powerful tool for
 optimizing machine learning models. It is a way of dividing data into train
 and test sets while accounting for the class imbalance in the data.
 It works by dividing the data into k-folds, and then training the model
 on k-1 folds while testing it on the remaining fold.
 This process is repeated k times, with each fold serving as the test set once.
 The StratifiedKFold technique is particularly useful for imbalanced datasets,
 where there is a significant difference in the class proportions between train and test sets.
 By stratifying the folds, we can ensure that the class proportions are similar
 between train and test sets, which helps to avoid overfitting."""

import numpy as np
from math import *
from numpy.random import seed
import matplotlib.pyplot as plt
seed(1)
import tensorflow
tensorflow.random.set_seed(1)
from keras.utils.vis_utils import plot_model
from IPython.display import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import itertools
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Load model
from keras.models import load_model
# Cross validation
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from numpy import interp
from sklearn.model_selection import KFold


def plot_cv_roc(X, y, X_test, y_test, model, n_splits=5, epochs=25):
    """
    Implement the K-cross validation procedure using StratifiedKFold
    with a model supplied by the user.
    It is a variation of k-fold which returns stratified folds: each set contains
    approximately the same percentage of samples of each target class as the complete set.
    Thus, plot the roc curves obtained from each configuration and the average roc curve.
    Finally, print the average scores of loss and accuracy on the train, validation and test set.

    Parameters
    ----------
    X : numpy array
        Array of train images.
    y : numpy array
        Array of train labels.
    X_test : numpy array
        Array of test images.
    y_test : numpy array
        Array of test labels.
    model : keras model
        Model to train for cross validation.
    n_splits : int
        Number of folders.
    epochs : int
        Number of epochs for train.

    Returns
    -------
    None
    
    """

    try:
      y = y.to_numpy()
      X = X.to_numpy() 
    except AttributeError:
      pass

    # Creates per-fold accuracy and loss lists
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    test_acc = []
    test_loss = []

    cv = StratifiedKFold(n_splits, shuffle=True)
    #https://scikit-learn.org/stable/modules/cross_validation.html

    tprs = []
    aucs = []
    interp_fpr = np.linspace(0, 1, 100)

    # Compiles the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Saving the weights before training as reset before each training
    model.save_weights('reset_model.h5')

    fig1 = plt.figure(1)

    i = 0
    for train, val in cv.split(X, y):
        # Reset the untrained model weights saved before
        model.load_weights('reset_model.h5')
        prediction = model.fit(X[train],
                               y[train],
                               batch_size=32,
                               epochs=epochs,
                               verbose=0
                               )
    
        # Evaluates the efficiency of the model
        scores_train = model.evaluate(X[train], y[train], verbose=0)
        scores_val = model.evaluate(X[val], y[val], verbose=0)
        scores_test = model.evaluate(X_test, y_test, verbose=0)

        print(f'Folder {i}')

        y_val_pred = model.predict(X[val])
        fpr, tpr, thresholds = roc_curve(y[val], y_val_pred)
        interp_tpr = interp(interp_fpr, fpr, tpr)
        tprs.append(interp_tpr)

        # Compute ROC curve and area under the curve    
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr,
                lw=1,
                label=f'ROC fold {i} (AUC = {roc_auc:.2f})'
        )
        # Train
        train_loss.append(scores_train[0])
        train_acc.append(scores_train[1])

        # Validation
        val_loss.append(scores_val[0])
        val_acc.append(scores_val[1])
    
        # Test
        test_loss.append(scores_test[0])
        test_acc.append(scores_test[1])
    
        i += 1

    plt.legend()
    plt.xlabel('False Positive Rate (FPR)', fontsize=18)
    plt.ylabel('True Positive Rate (TPR)', fontsize=18)
    plt.show()

    fig2 = plt.figure(2)
    plt.plot([0, 1], [0, 1],
            linestyle='--',
            lw=2,
            color='r',
            label='Chance',
    )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(interp_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(interp_fpr, mean_tpr,
            color='b',
            label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})',
            lw=2,
            )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(interp_fpr, tprs_lower, tprs_upper,
                    color='grey',
                    alpha=.2,
                    label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Cross-Validation ROC', fontsize=18)
    plt.legend(loc="lower right", prop={'size': 15})
    plt.show()

    # Computing and printing average scores
    print('\nTRAIN - Average scores:')
    print(f'Train accuracy: {np.mean(train_acc):.2f} +/- {np.std(train_acc):.2f}')
    # Calculare loss error as maximum value error
    err_train_loss = (max(train_loss)-min(train_loss))/2
    print(f'Train loss: {np.mean(train_loss):.2f} +/- {err_train_loss:.2f}')

    print('\nVAL - Average scores:')
    print(f'Val accuracy: {np.mean(val_acc):.2f} +/- {np.std(val_acc):.2f}')
    # Calculare loss error as maximum value error
    err_val_loss = (max(val_loss)-min(val_loss))/2
    print(f'Val loss: {np.mean(val_loss):.2f} +/- {err_val_loss:.2f}')

    print('\nTEST - Average scores:')
    print(f'Test accuracy: {np.mean(test_acc):.2f} +/- {np.std(test_acc):.2f}')
    # Calculare loss error as maximum value error
    err_test_loss = (max(test_loss)-min(test_loss))/2
    print(f'Test loss: {np.mean(test_loss):.2f} +/- {err_test_loss:.2f}')

    return fig1, fig2