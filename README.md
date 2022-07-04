# microcal_classifier

[![CircleCI](https://circleci.com/gh/lorenzomarini96/microcal-classifier.svg?style=shield)](https://app.circleci.com/pipelines/github/lorenzomarini96/microcal_classifier?filter=all)

[![Documentation Status](https://readthedocs.org/projects/microcal-classifier/badge/?version=latest)](https://microcal-classifier.readthedocs.io/en/latest/?badge=latest)

Convolutional neural networks for the computing methods for experimental physics and data analysis (CMEPDA) course. This project Compare the performance of a convolutional neural networks classification on a microcalcification image dataset, with the performance obtained in an analysis pipeline where the mammographic images containing either microcalcifications or normal tissue are represented in terms of wavelet coefficients.

# Motivations

Cluster of microcalcifications can be an early sign of breast cancer. In this python package, an approach based on convolutional neural networks (CNN) for the classification of microcalcification clusters is proposed.

# Materials and methods

## Train, test, validation sets

The provided dataset contains 396 images of 60$\times$60 pixels representing portions of mammogram either containing microcalcification clusters (label=1) or  normal breast tissue (label=0).
This dataset contains many images representing portions of mammogram either containing microcalcification clusters (label=1) or  normal breast tissue (label=0).
The available images are already partitioned in a train and a test samples, containing, respectively:

| Sets      | Normal tissue | Microcalcification clusters|
| ---       |     ---       |         ---                |
| Train     |      209      |    187                     |
| Test      |      205      |    196                     |

All dataset details are provided in the reference paper.

The datasets are partitionated according as follows:




## CNN Architecture

## Cross validation procedures

Train and test sets can be swapped in a cross validation procedure.

# Results

## Performance evaluation

$Accuracy = \frac{TP + TN}{TP + FN + FP + TN}$

### Precision, Recall and F1-Score

### Confusion Matrix

### ROC Curves


# Conclusions


