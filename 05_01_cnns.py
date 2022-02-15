from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Today we will work with a dataset on breast cancer, also built into the scikit-learn datasets.
cancer = load_breast_cancer()

# print('Dataset raw object', cancer)
print('Dataset description', cancer['DESCR'])

# Split into our training and testing XY sets.
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# However, the MLP method doesn't automatically scale the data, so let's do that.
# Here I show how to do it manually with Numpy functions, though there are alternative
# built-in methods within scikit-learn.

# Using numpy functions, compute the mean value per feature on the training set and the STD.
# May want to remind ourselves what the X_train looks like.
print('X_train', X_train)

# The power of Numpy starts to be evident here.
# We can calculate the mean of each column (we specify we want sums each column, which means
# summing down the first axis). The .mean() function from Numpy as insanely efficient
# for processing very big data and is ready for, e.g., super computers.
mean_on_train = X_train.mean(axis=0)

# print('mean_on_train', mean_on_train)

# the .std() function is similarily powerful/fast.
std_on_train = X_train.std(axis=0)

# print('std_on_train', std_on_train)

# Still using the Numpy awesomeness,
# subtract the mean, and scale by inverse standard deviation,
# making it  mean=0 and std=1

X_train_scaled = (X_train - mean_on_train) / std_on_train
X_test_scaled = (X_test - mean_on_train) / std_on_train

# Using this new scaled training data, we are ready to define a Neural Net,
# Known here as a Multi-Layer-Perceptron (MLP) classifier.
# Because this next line hides away millions of other lines of code, you may want
# to explore it. In Pycharm, you can navigate to a function's definition by placing
# your cursor in the function and press Ctrl-b. Try it! The  best documentation
# is often the code itself.
mlp = MLPClassifier(random_state=0)

# Now fit it with the scaled X and y TRAINING data.
mlp.fit(X_train_scaled, y_train)

# Assess its accuracy on the TRAINING and the TESTING data.
# Notice here also I'm introducing another convenient way of combining strings
# and numbers. The {:.2f} specifies a placeholder for a 2-digit representation
# of a floating point number. The Format method then places that floating point value
# into that placeholder.
score_train = mlp.score(X_train_scaled, y_train)
score_test = mlp.score(X_test_scaled, y_test)

print("Accuracy on training set: {:.3f}".format(score_train))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# Other concepts discussed earlier, such as regularization and Cross-Validation, also apply here.
# To illustrate, here we will set the alpha parameter to include a regulariazation term.
# The cross-validation method is often defined by the model itself and will be used automatically
# when you call .fit().
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)

# print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
# print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))

# But what does a MLP Neural Net actually LOOK like?
# Plot the coeffs_ array to find out:

plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()