from keras.datasets import fashion_mnist

## Dataset
# 28x28 grayscale images of 70,000 fashion products from 10 categories,
# and 7,000 images per category. The training set has 60,000 images,
# and the test set has 10,000 images

# The load_data() function provided automatically will provide the training and testing
# split as two python "tuples" (similar to lists). The following line
# calls the function and assigns the returned data to 4 different variables
(train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()

# Process the data

import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

print('Training data shape : ', train_X.shape, train_Y.shape)

print('Testing data shape : ', test_X.shape, test_Y.shape)

