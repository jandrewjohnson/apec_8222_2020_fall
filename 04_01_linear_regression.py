# Author: Justin A Johnson. Adapted from sklearn documentation and original content. License: Modified BSD License.

# As always, start with importing the libraries we will use.
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Here we will load the diabetes dataset, which is a built-in dataset from sklearn.
full_dataset = datasets.load_diabetes()
print('full_dataset', full_dataset)

# It's overwhelming and hard to read what is printed out, but let's dig into this notation
# because it's frequently used and will help us understand different Python
# datatypes.

# First, notice that the object starts with {, which indicates it is a python dicitonary.
# Dictionaries are standard ways of expressing key-value pairs.
# The standard notation for a dictionary is {key1: value1, key2: value2}

# Next, looking at the keys of the database lets us dig 1 level in. we can print out just the keys.
print('dictionary keys:', full_dataset.keys())

# If we want, we can access just one entry in the dictionary using the key. A useful one is the key DESCR.
# Print that out using the dictionary [] notation.
print(full_dataset['DESCR'])

# We could also extract the data and assign
# it to a data_array variable for inspection.
data_array = full_dataset['data']
print('data_array', data_array)

# Class exercise: Print out the mean BMI in the dataset, the sum BMI, and the sum of the squared BMI values.
# Explain why the sum of the squared BMI is what it is.

# HINT: You will need to read the DESCR to understand which column the BMI is stored in.
# HINT2: To create a new variable with just the desired column of the array, you can use
# Array slicing notation like a = data_array[:, n] where the : means you want ALL rows, and the n means
# You want just column n
# HINT3: You may want to use np.sum(), np.mean(), and the ** exponent operator.


# For conveinence, sklearn also just has an option to get the key parts for the regression ready to use.
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Look at diabetes_X and notice there are lots of independent variables. Rather than printing the whole
# Array, which would be messy, just look at the .shape attribute.l
# print('diabetes_X', diabetes_X.shape)

# For now, we're just going to use a single one for simplicity. The following line extracts just the second column and
# reshapes it to be the shape expected by the LinearRegression model. IGNORE UNDERSTANDING THIS FOR NOW if you want
# because we will dig in to reshaping later.

diabetes_X = diabetes_X[:, np.newaxis, 2]
# print('diabetes_X', diabetes_X.shape)

# Next we are going to do a very rudimentary split of the data into training and testing sets using
# array slice notation. The following lines assigns the last all but the last 20 lines to the TRAIN set
# and the remaining 20 to the test set.
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# The basic notation for sklearn is first to create a regression model object.
# this model is "empty" in the sense that it has no coefficients identified.
regression_object = linear_model.LinearRegression()
print('regression_object', regression_object)

# To set the coefficients, we call the regression_object's fit() method, calling the X and Y training
# data
regression_object.fit(diabetes_X_train, diabetes_y_train)

# Now the regression_object is "trained," which means we can also call it's predict() method
# which will take some other observations and (in the case of OLS), multiple the new observations
# against our trained coefficients to make a prediciton.
diabetes_y_pred = regression_object.predict(diabetes_X_test)

# The predict method returned an array of numerical predictions, which we can look at.
print('diabetes_y_pred', diabetes_y_pred)

# More interesting might be to look at the coefficients. Once the model has been fit, it has a new
# attribute .coef_ which stores an array of coefficients. In this case it will only be an array of length
# 1 because we just have one input.

print('Coefficients: \n', regression_object.coef_)

# We can also use sklearn's built in evaluation functions, such as for the mean squared error
mse = mean_squared_error(diabetes_y_test, diabetes_y_pred)
print('Mean squared error:',  mse)

# Or perhaps we want the r2 for the second independent variable (which is the only one we used)
r2_score_value = r2_score(diabetes_y_test, diabetes_y_pred)
print('r2 calculated on TEST data: ', r2_score_value)


# Finally, to prove to ourselves that we know what we are doing, let's plot this.
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

# CLASS EXERCISE:
# report the r2 for a LinearRegression model that uses all of the independent variables provided by the dataset.
# Is this new r2 higher or lower than what you calculated above. Why? Is this expected?



# Strangely (i think), sklearn doesn't report summary statistics in the classic, econometric
# sense because it focuses on the train, test paradigm, which is not equivilent to a model
# performance report (which in the classic case is only reporting performance of the TRAINING
# data.

# Nonetheless, Here's how I do it, using an alternative, more econometrics-focused package.
# You will need to conda install statsmodel if you want to uncomment this line and have it work.
import statsmodels
from statsmodels.api import OLS

result = OLS(full_dataset.target,full_dataset.data).fit().summary()
# print(result)


