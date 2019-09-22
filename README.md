# Supervised learning
*Assignment #1 - CS 7641 Machine Learning course - Charles Isbell & Michael Littman - Georgia Tech*

Please clone this git to a local project if you want to replicate the experiments reported in the assignment paper.

Virtual Environment
----
This project contains a virtual environment folder [```venv```](placeholder). This folder contains all the files needed to create a virtual environment in which the project is supposed to run.

requirements.txt
----
This file contains all the necessary packages for this project. (Running ```pip install -r requirements.txt``` will install all the packages in your project's environment - should not be necessary if you are using the given ```venv```folder here)

The datasets
----
These datasets (```train_32x32.mat```and ```tumor_classification_data.csv```) are the datasets described in the assignment paper. They can also be downloaded from their original sources:
* http://ufldl.stanford.edu/housenumbers/train_32x32.mat
* https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

digit_recognition.py and tumor_classification.py
----
These Python files are the implementations of the 5 Machine Learning algorithms studied in this assignment over the two datasets that we use for this project. They are almost identical but the way the data is prepared for the algorithms and the hyperparameters of each algorithm differ from one script to the other because of the differences in the datasets.
