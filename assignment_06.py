## Assignment 6: Using post-LASSO on large spatial data
# This assignment will give you a real (active) research topic that I've discussed a little bit in class:
# predicting carbon storage as a function of high-resolution gridded data. In the class google drive for data
# you will see a new assignment_06 folder we will use.

# This assignment will have you use the automated variable selection approach within LASSO to deal with a common situation
# in regressions on raster-stacks: we have so much data everything is significant but will lead to massive overfitting.
# The basic approach used here will involve reading in 2d rasters, flattening them into a 1d column ready to add
# to a dataframe shaped object, which we will use as our X matrix.

# There will be two actual files you will submit: This python file with your code and comments and a word document
# containing key outputs from steps (noted specifically below). For convenience and to ensure you
# don't miss any outputs, you can search the phrase "WORD DOCUMENT" to highlight each place this is.

# PRELIMINARY STEP: Import these libraries.
# Load libraries
import numpy as np
import os
from osgeo import gdal
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt
from statsmodels.api import OLS


# Step 1: download the data and assign a relative path to the soyo_tile directory in that assignment directory.
# Here, I actually want you to use the exact path below so that it works on my machine too. It
# is your task to ensure your script runs in the right location and the data is stored in the right
# location that this relative path works.

data_dir = '../../Data/assignment_06/soyo_tile'


# Step 2: assign each of the raster paths in the directory to a dictionary for later use. I've included
# most of the code (so you don't have to waste your time typing), but add in the three missing
# paths.

raster_paths = {}
raster_paths['agb_observed_baccini_2000_30m'] = os.path.join(data_dir, "agb_observed_baccini_2000_30m.tif")

raster_paths['CRFVOL_M_sl1_250m'] = os.path.join(data_dir, "CRFVOL_M_sl1_250m.tif")
raster_paths['HISTPR_250m'] = os.path.join(data_dir, "HISTPR_250m.tif")
raster_paths['OCDENS_M_sl1_250m'] = os.path.join(data_dir, "OCDENS_M_sl1_250m.tif")
raster_paths['PHIHOX_M_sl1_250m'] = os.path.join(data_dir, "PHIHOX_M_sl1_250m.tif")
raster_paths['roughness_30s'] = os.path.join(data_dir, "roughness_30s.tif")
raster_paths['SLGWRB_250m'] = os.path.join(data_dir, "SLGWRB_250m.tif")
raster_paths['SLTPPT_M_sl1_250m'] = os.path.join(data_dir, "SLTPPT_M_sl1_250m.tif")
raster_paths['SNDPPT_M_sl1_250m'] = os.path.join(data_dir, "SNDPPT_M_sl1_250m.tif")
raster_paths['terrain_ruggedness_index_30s'] = os.path.join(data_dir, "terrain_ruggedness_index_30s.tif")
raster_paths['TEXMHT_M_sl1_250m'] = os.path.join(data_dir, "TEXMHT_M_sl1_250m.tif")
raster_paths['wc2.0_bio_30s_01'] = os.path.join(data_dir, "wc2.0_bio_30s_01.tif")
raster_paths['alt_30s'] = os.path.join(data_dir, "alt_30s.tif")
raster_paths['AWCh1_M_sl1_250m'] = os.path.join(data_dir, "AWCh1_M_sl1_250m.tif")
raster_paths['BDRICM_M_250m'] = os.path.join(data_dir, "BDRICM_M_250m.tif")
raster_paths['BDRLOG_M_250m'] = os.path.join(data_dir, "BDRLOG_M_250m.tif")
raster_paths['BLDFIE_M_sl1_250m'] = os.path.join(data_dir, "BLDFIE_M_sl1_250m.tif")


# Step 3: Our dependent variable will be 30 meter observations of carbon storage from
# Baccini et al. (unpublished, but soon to be published) data. The label I assigned in the dictionary
# above was agb_observed_baccini_2000_30m for this variable.
# Use gdal.Open, GetRasterBand(1) and ReadAsArray() to read this geotiff as a numpy file
# See lesson 03_07_spatial_data.py if you've forgotten how to use gdal.
# Side note:
# If you get an error like: "ERROR 4: This is a BigTIFF file.  BigTIFF is not supported by this version of GDAL and libtiff."
# make sure you have conda installed gdal from the CONDA FORGE  using the command "conda install gdal -c conda-forge" option.


# Step 4, Create an empty numpy array (or full of zeros) of the right shape to house all our raster data.
# A very CPU-efficient way of arranging a stack of 2d rasters (which would be 3d once stacked up), is
# to flatten each 2d raster into a longer 1d array. This will go into our X matrix.
# In order to create the right sized X matrix, first get the n_obs and n_vars by inspecting
# the dependent variable raster and the dictionary of inputs above.
# Note that the n_vars should be the number of INDEPENDENT and DEPENDENT variables
# report in your WORD DOCUMENT the size of the data_array you created.


# Step 5, Iterate through the dictionary and load each raster as a 2d array, flatten it to 1d
# using the .flatten() method in numpy. Assign this 1d array to the correct column
# of the data array. By convention, the depvar will be the first column.

# Hint, assuming you have arranged your X array in the correct way, it should have observations (pixels)
# as rows and variables as cols. Given that each flattened array is for one variable and is as long as there
# are rows, a convenient way of assigning it would be to use numpy slice notation, potentially similar to:
# data_array[:, column_index_integer].
# The first colon just denotes the whole row and the column index is an integer you could create pointing to the right row.
# Some incomplete code to get you started is below.

for name, path in raster_paths.items():
    print('Loading', path)
    flattened_raster_array = band.ReadAsArray().flatten()
    data_array[:, col_index] = flattened_raster_array
    feature_names.append(name)


# Step 6, extract the first array row of the data_array and assign it to y. Assign the rest to X.


# Step 7, split thre X and y into testing and training data such that
# the training data is the first million pixels and the testing data is the next 200,000
# Do this using numpy slice notaiton on the X and y variables you created.


# Step 8 (optional but useful). To make the code run faster, we are going to
# use every 10th pixel. We can easily get this via numpy slicing again, using
# x_train[::10] to get every 10th pixel.


# Step 9, create a Lasso object (using the default penalty term alpha)
# and fit it to the training data. Create
# and print out a vector of predicted carbon values. Also print out the score
# using the lasso object's .score() method on the TESTING data.
# Add the fitted lasso score to your WORD DOCUMENT.


# Step 10, optional and just for fun. To view how our projections LOOK, we can
# create a predicted matrix on the whole X, reshape it back into the
# original 2d shape and look at it. You can compare this to the input array
# to visualize how it looks. Note that this will only work if you name your objects
# like mine.

# full_prediction = fitted_lasso.predict(X)
# prediction_2d = full_prediction.reshape(2000, 2000)
# plt.imshow(prediction_2d)
# plt.show()
#
# plt.imshow(array)
# plt.show()


# Step 11, Create a list of 30 alphas using np.logspace(-1, 3, 30). Using a for loop
# iterate over those alphas and run the Lasso model like above, but using the alpha values
# in the loop. Print out the fit score at each step. Using matplotlib, plot how this
# value changes as alpha changes. Finally, extract the best alpha of the bunch.
# Put the plot of alphas and their scores in your WORD DOCUMENT along with the value
# of the optimal alpha.

alphas = np.logspace(-1, 3, 30)


# Step 12, rerun the lasso with that best value and identify all of the coefficiencts
# that were "selected" ie had non-zero values. Save these coefficient indices and labels
# to a list.


# Step 13, Using Statsmodels, run an OLS version on the selected variables.
# Copy and paste your resulting OLS.summary() table to your WORD DOCUMENT. In addition
# add to your WORD DOCUMENT a descripting of how this result is better than a vanilla OLS.




