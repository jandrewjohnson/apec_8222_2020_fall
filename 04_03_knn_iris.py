# Author: Justin A Johnson. Adapted from sklearn documentation and original content. License: Modified BSD License.

import numpy as np
from scipy import sparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Here we will  use the built-in dataset load_iris to make a K-Nearest-Neightbors model
# that predicts which species of Iris an observation is based on 4 input variables.

from sklearn.datasets import load_iris
iris_dataset = load_iris()

# print('iris_dataset', iris_dataset)
#
# print("Keys of iris_dataset:", (iris_dataset.keys()))
# print("Target names:", iris_dataset['target_names'])
# print("Shape of data:", iris_dataset['data'].shape)



X_train, X_test, y_train, y_test = train_test_split(
 iris_dataset['data'], iris_dataset['target'], random_state=0)


# print("X_train shape: {}".format(X_train.shape))
# print("y_train shape: {}".format(y_train.shape))
#
# print("X_test shape: {}".format(X_test.shape))
# print("y_test shape: {}".format(y_test.shape))

# create dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
 marker='o', hist_kwds={'bins': 20}, s=60,
 alpha=.8)
import matplotlib.pyplot as plt
# plt.show()

# Here we will use a k-nearest
# neighbors classifier, which is easy to understand. Building this model only consists of
# storing the training set. To make a prediction for a new data point, the algorithm
# finds the point in the training set that is closest to the new point. Then it assigns the
# label of this training point to the new data point.
# The k in k-nearest neighbors signifies that instead of using only the closest neighbor
# to the new data point, we can consider any fixed number k of neighbors in the train‐
# ing (for example, the closest three or five neighbors). Then, we can make a prediction
# using the majority class among these neighbors. We will go into more detail about
# this in Chapter 2; for now, we’ll use only a single neighbor.
# All machine learning models in scikit-learn are implemented in their own classes,
# which are called Estimator classes. The k-nearest neighbors classification algorithm
# is implemented in the KNeighborsClassifier class in the neighbors module. Before
# we can use the model, we need to instantiate the class into an object. This is when we
# will set any parameters of the model. The most important parameter of KNeighbor
# sClassifier is the number of neighbors, which we will set to 1:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train, y_train)


# Making Predictions
# We can now make predictions using this model on new data for which we might not
# know the correct labels. Imagine we found an iris in the wild with a sepal length of
# 5 cm, a sepal width of 2.9 cm, a petal length of 1 cm, and a petal width of 0.2 cm.
# What species of iris would this be? We can put this data into a NumPy array, again by
# calculating the s

X_new = np.array([[5, 2.9, 1, 0.2]])
# print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)

# print("Prediction: {}".format(prediction))
# print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# Evaluating the Model
# This is where the test set that we created earlier comes in. This data was not used to
# build the model, but we do know what the correct species is for each iris in the test
# set.

# Therefore, we can make a prediction for each iris in the test data and compare it
# against its label (the known species). We can measure how well the model works by
# computing the accuracy, which is the fraction of flowers for which the right species
# was predicted:
y_pred = knn.predict(X_test)

# print("Test set predictions:\n {}".format(y_pred))
#
# print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

