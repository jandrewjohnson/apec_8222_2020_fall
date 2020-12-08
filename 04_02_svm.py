# Author: Justin A Johnson. Adapted from sklearn documentation and original content. License: Modified BSD License.

# SciPy
import numpy as np
import scipy
import sklearn
from sklearn import datasets
import pandas as pd
import os

# One of the canonical datasets in sklearn is a series of images of handwritten digits.
# We've imported the datasets above, but now lets load it.
digits = datasets.load_digits()

# First, take a look at the raw python object:
# print('digits\n', digits)

# Not super helpful unless you're very good at reading python dictionary notation
# Fortunately, one of the entries in this dataset is a description. Let's read that.
# print('DESCR\n', digits['DESCR'])

# Now that we're oriented, also look at one particular image of a digit, just so you know what
# it actually looks like. Below, we print just the first (index = 0) numeral of the 5620 they provide.
# print('digits.images[0]\n', digits.images[0])

# If you squint, maybe you can tel what image it is, but let's plot it to be sure.
import matplotlib
from matplotlib import pyplot as plt
# plt.imshow(digits.images[0])
# plt.show()

# Notice also in the dataset that there is a 'targets' attribute in the dataset.
# This is the correct numeral that we are trying to make the model predict.
# print('target', digits.target)

# Our task now is to train a model that inputs the digit images and predicts the digit numeral.
# For this, we're going to use SVM, as discussed in lecture.
from sklearn import svm

# For now, the parameters are going to be manually set but we'll address how to choose them later.
# Here, I want to illustrate the basic approach used in sklearn to Load, train, fit and predict the model
classifier = svm.SVC(gamma=0.001)

# At this point, clf (the classifier) is not yet "trained", ie. not yet fit to the model.
# All ML algorithms in SKLEARN have a .fit() method, which we will use here, passing it the images and the targets

# Before we train it, we want to split the data into testing and training splits
# Class question: Remind me WHY are we splitting it here? What is the bad thing
# that would happen if we just trained it on all of them?

# Before we can even split the data, however, we need to reshape it to be
# in the way the regression model expects.

# In particular, the SVM model needs a 1-dimensional, 64 element array. BUT, the
# input digits we saw were 2-dimensional, 8 by 8 arrays.

# This actually leads to a somewhat mind-blown example of how computers "think" differently than we do.
# We clearly think about a numeral in 2 dimensional space, but here we see that the computer doesn't
# care about the spatial relationship ship at all. It sees each individual pixel as it's own
# "Feature" to use the classification parlance. You could even reshuffle the order of those 64 digits
# and as long as you kept it consistent across the data, it would result in identical predictions.
# Later on, we will talk about machine learning techniques that leverage rather than ignore this
# 2 dimensional, spatial nature of the data.

# For now, let's just look at the data again. Rather than print it out, I really just want the shape
# so that i don't get inundated with text.
# print('digits.images shape', digits.images.shape)

# So we need to get it into a shape of n "samples" by 64 "features"
n_samples = len(digits.images)
n_features = digits.images[0].size

# print('n_samples', n_samples)
# print('n_features', n_features)

data = digits.images.reshape((n_samples, n_features))

# Now check the shame again to see that it's right.
# print('data shape', data.shape)

# Now that we've arranged our data in this shape, we can split it into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False)

# print('X_train', X_train)
# print('y_train', y_train)


# Finally, now that we've split it, we can call the classifier's fit method which takes the TRAINING data as input.
classifier.fit(X_train, y_train)

# Now, our classifier object has it's internal parameters fit so that when we give it new input, it predicts
# what it thinks the correct classification is.

predicted = classifier.predict(X_test)

# Looking at the predicted won't be very intuitive, but you could glance.
# print('predicted', predicted)

# Let's plot a few of them in nicer format. Don't worry about learning the plotting code
# but it's a useful example to show the power.
_, axes = plt.subplots(2, 4)
images_and_labels = list(zip(digits.images, digits.target))
for ax, (image, label) in zip(axes[0, :], images_and_labels[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for ax, (image, prediction) in zip(axes[1, :], images_and_predictions[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Prediction: %i' % prediction)
# plt.show()

from sklearn import metrics

# print("Classification report:\n", metrics.classification_report(y_test, predicted))

# Also, let's look at the confusion matrix. Here we use some convenient built in string-formatting options
# in sklearn.

disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
disp.figure_.suptitle("Confusion Matrix")

# print("Confusion matrix:\n", disp.confusion_matrix)

# Finally, show it so that you can look at it and see how good we did.
# plt.show()








