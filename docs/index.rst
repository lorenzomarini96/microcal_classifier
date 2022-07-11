.. microcal_classifier documentation master file, created by
   sphinx-quickstart on Fri Jul  8 11:07:48 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to microcal_classifier's documentation!
===============================================

The microcal_classifier is a python package for analyzing digital mammograms images.
It allows to apply a convolutional neural network on a set of images - click here to download the images - for the classification of mammograms. 
In particular, it is able to identify or not the presence of microcalcifications.
The performance of the classifier is quantified in terms of area under the ROC cure and accuracy.
It is also possible to implement a cross validation on the data, as well as a data augmentation procedure.


.. toctree::
   :maxdepth: 2
   
   api
   
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
