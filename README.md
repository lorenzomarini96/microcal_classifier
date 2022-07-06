# microcal_classifier

[![CircleCI](https://circleci.com/gh/lorenzomarini96/microcal_classifier.svg?style=shield)](https://app.circleci.com/pipelines/github/lorenzomarini96/microcal_classifier?filter=all)
[![Documentation Status](https://readthedocs.org/projects/microcal-classifier/badge/?version=latest)](https://microcal-classifier.readthedocs.io/en/latest/?badge=latest)

Convolutional neural networks for the computing methods for experimental physics and data analysis (CMEPDA) course. This project Compare the performance of a convolutional neural networks classification on a microcalcification image dataset, with the performance obtained in an analysis pipeline where the mammographic images containing either microcalcifications or normal tissue are represented in terms of wavelet coefficients.

# Motivations

Cluster of microcalcifications can be an early sign of breast cancer. In this python package, an approach based on convolutional neural networks (CNN) for the classification of microcalcification clusters is proposed.

# Materials and methods

## Train, test, validation sets

The provided dataset contains 797 images of 60 $\times$ 60 pixels representing portions of mammogram either containing microcalcification clusters (label=1) or  normal breast tissue (label=0). The available images are already partitioned in a train and a test samples, containing, respectively:

| Sets      | Normal tissue | Microcalcification clusters|
| --------- | ------------- | -------------------------- |
| Train     |      209      |    187                     |
| Test      |      205      |    196                     |

All dataset details are provided in the reference paper (add references).

The datasets are partitionated according the following hierarchy:

```
DATASETS
└── IMAGES
    └── Mammography_micro
        ├── Test
        │   ├── 0
        │   └── 1
        └── Train
            ├── 0
            └── 1
```

<img src="docs/images/mammo_images.png" width="500"> 

The DATASETS folder can be downloaded from 

To define the train and validation set we can use the function **train_test_split**, which split the labels into random train and validation subsets by proportions.

In this case, the first dataset contains 80% of the total number of 396 images, randomly selected within the sample (randomly assigns the specified proportion of files from each label to the new datastores).


## CNN Architecture

### Model

## Cross validation procedures

Train and test sets can be swapped in a cross validation procedure.

# Results

## Performance evaluation

The performance of CNN prediction model was evaluated by computing the area under the receiver-operating characteristic curve (AUC), specificity, sensitivity, F1-score, and accuracy.
Before exploring the metrics, we need to define True Positive (TP), False Positive (FP), True Negative (TN) and False Negative (FN).

### Precision, Recall and F1-Score

$Accuracy = \frac{TP + TN}{TP + FN + FP + TN}$

$Specificity = \frac{TN}{TN + FP}$

$Sensitivity = \frac{TP}{TP + FN}$

$F1 = \frac{2 \times PR \times Recall}{PR + Recall}$

INSERIRE TABELLA

## Loss/Accuracy vs Epoch

Inserire plot


### Confusion Matrix

Inserire plot

### ROC Curves

Inserire plot

### Correct/Incorrect classification samples

Inserire plot

# Conclusions

| Model | Class      | Precision     | Recall|  F1-score | Support | Accuracy |
| ----- | ---------- | ------------- | ----- | --------- | ------- | -------- |
| CNN   | Train      |      209      |   187 |           |  205    |          |
|       | Test       |      205      |   196 |           |  196    |          |
|       |            |               |       |           |         |          |
| ML    | Train      |      209      |   187 |           |  205    |          |
|       | Test       |      205      |   196 |           |  196    |          |


# References

- Retico: Inserire Link
- 
