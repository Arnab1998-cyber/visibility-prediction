# visibility-prediction

## Overview
This is a flask based project, which predicts visibility in a perticular day

## Problem Statement
This is a project for ATC. Based on various factors on a perticular day, they want to build a system which predicts visibility on that day.

## Tecchnologies
Pandas, Flask, Sklearn

## Data Visualization
Visualize the data pairwise
#### Tools
Seaborn

## Database
MySql

## Algorithhms
#### Clustering:
K-Means Clustering
#### Predictive Model:
Random Forest Regressor

# Description
we are given a set of csv files with vaarious visibility data for training purpose. But here is a problem. In that data many descrepencies can be found. For that reason we first created a training pipeline which includes data validation process. In validation stage we check the naming convension of each files, check whether there is any null column, number of columns. Based on these, the given training data goes to good or bad csv files.

The Good files are integrated and we dump that data into mysql database. Now this data is ready for training our machine learning model.

First we clustered the data and for each cluster we use a machine learning model. As we know model performs well when varience in the dataset is less.

Now we have Prediction files also. We first send those files for validation, and divide them into good and bad csv files. The good data is dumped into database and from there we create a csv file. The csv file is goes into the cluster model, we saved after training. After dividing the data into clusters, each cluster goes into the specific model, we specified for that cluster. We create output files with prediction.on
