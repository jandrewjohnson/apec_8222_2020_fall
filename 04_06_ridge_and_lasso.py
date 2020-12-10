# Author: Justin A Johnson. Adapted from sklearn documentation and original content. License: Modified BSD License.

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.feature_selection import SelectFromModel
from sklearn import datasets, linear_model
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score


# Again we're going to use our diabetes dataset. Inspect it again just to remind yourself
# what is in it.

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

feature_names = diabetes.feature_names

print(diabetes['DESCR'])
print(feature_names)

# To speed up calculation, we're going to just use the first 150 observations
# using numpy slice notation to grab them out of the X, y

X = X[:150]
y = y[:150]

# Class exercise: Review of OLS. Report back the MSE when of y versus the predicted y when you use the X and y variables above.
# You might want to utilize the scikit-learn linear regression tools, in particular the following functions:
#
# linear_model.LinearRegression()
# .fit(X, y)
# .predict(X)
# mean_squared_error(y, y_pred)


# In addition to the scikit-learn OLS, we also will be using today
# the statsmodels implementation. Let's go ahead an import it.
# I'm assuming you've already run conda install statsmodels -c conda-forge in your
# anaconda command line.
from statsmodels.api import OLS

# statsmodels uses a similar syntax to scikit learn of createing a model, fitting it, and
# returning the summary.
model = OLS(y, X)
fitted_model = model.fit()
result = fitted_model.summary()
# print(result)

# Today's goal, however, is to do Lasso this same dataset.
# To start, lets create a Lasso object. Notice that we are not
# setting the alpha/gamma value when we create it.

lasso = Lasso(random_state=0, max_iter=10000)

# Instead, we are going to test a variety of different alphas, as here:
alphas = np.logspace(-3, -0.5, 30)

# We are going to be passing this range of tuning parameters to a GridSearch function
# that will test which works best when cross-validation methods are applied.
# First though, we have to put the alphas into the form the GridSearchCV funciton
# Expects, which is a list of dictionaries.
tuning_parameters = [{'alpha': alphas}]

# Recall that CV works by calculating the fit quality of different folds of the
# TRAINING data. Here we will just use 5 folds.
n_folds = 5

# Finally, we have all our objects ready to pass to the GridSearchVC function which will
# Give us back a classifier object.

clf = GridSearchCV(lasso, tuning_parameters, cv=n_folds, refit=False)

# When we call the clf.fit() method, we will iteratively be calling the Lasso.fit() with different permutations of
# tuned parameters and then will return the classifier with the best CV fit.
clf.fit(X, y)

# The classifier object now has a variety of diagnostic metrics, reporting
# back on different folds within the Cross Validation
# print('clf keys returned:', clf.cv_results_.keys())

# Some relevant results are as below, which we'll extract and assign to lists.
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']

# CLASS Activity: break out into groups and explore the scores and alphas lists we've created.
# Identify which alpha is the best, based on the MSE score returned. One way to consider doing this
# would be to create a for loop to iterate through a range(len(scores)): object, reporting the
# alphas and scores. save the optimal alpha as chosen_alpha.


chosen_alpha = 0.5 # This will be set for real from class activity

# Now we can rerun a vanilla (no CV) version of Lasso with that specific alpha.
# This will return, for instance, a .coef_ list.
clf2 = Lasso(alpha=chosen_alpha, random_state=0, max_iter=10000).fit(X, y)
# print("coefficients", clf2.coef_)

# Simply looking at the coefficients tells us which are to be included.
# Question: How will we know just by looking?

# Extract the feature names and colum indices of the features that Lasso has selected.
selected_coefficient_labels = []
selected_coefficient_indices = []
for i in range(len(clf2.coef_)):
    # print('Coefficient', feature_names[i], 'was', clf2.coef_[i])
    if abs(clf2.coef_[i]) > 0:
        selected_coefficient_labels.append(feature_names[i])
        selected_coefficient_indices.append(i)

# This process led us to the following selected_coefficient_labels:
# print('selected_coefficient_labels', selected_coefficient_labels)

# For fun, let's plot the alphas, scores and a confidence range.
# What does this show us about the optimal alpha and how it varies with score?
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])

# plt.show()

# Finally, now that we have our selected labels, we can use them to select the numpy array
# columns that we want to use for a post-LASSO run.
new_x =X[:, selected_coefficient_indices]
# print('new_x', new_x)

# Plug this new x matrix into our statsmodels OLS function and print that out.
# How is this better than a vanilla OLS?
result = OLS(y, new_x).fit().summary()
# print(result)

