# Author: Justin A Johnson. Adapted from sklearn documentation and original content. License: Modified BSD License.

# Python Programming and Big-Data for Sustainability Science
# This guide is a shortened version of what we will cover in section 3, intended for summary and/or quick start
# for those who know some python.


# PYTHON BASICS

# Comments: The hashtag makes the rest of the line a comment. The more programming you do, the more you focus on making good comments.

# Assign some text (a string) to a variable
some_text = 'This is the text.'

# Assign some numbers to variables
a = 5  # Here, we implicitly told python that a is an integer
b = 4.6  # Here, we told python that b is a floating point number (a decimal)

# Python as a calculator. Notice that it can add the integer and the float. Python "smartly" redefines variables so that they work together.
# This is different from other languages which require you to manually manage the "types" of your variables.
sum_of_two_numbers = a + b

# Printing output to the console
print('Our output was', sum_of_two_numbers) # You may see print output in some cases, but this is dated code.

# At this point, run the script (SHIFT F10).

# IDE Fun tricks (also below illustrates a multi-line string

"""1.) Move back and forth in your history of cursor positions (using your mouse forward and back buttons)
2.) Edit on multiple lines at the same time (hold alt and click new spots)
3.) Smartly paste DIFFERENT values
4.) Duplicate lines (ctrl-d)
5.) Introspection
6.) Debugging
"""




# Functions
def my_function(input_parameter_1, input_parameter_2):
    product = input_parameter_1 * input_parameter_2
    return product

# Importing packages

# Built-in packages via the Python Standard Library
import math
import os, sys, time, random

# Using imported modules
number_rounded_down = math.floor(sum_of_two_numbers)

# NOTE: From here on, output print statements will start with a comment (to deactivate them) so that your output isn't overwhelming.
# when you get to a line that you actually want to see, uncomment it by deleting the #.

# print(number_rounded_down)

# Using packages from elsewhere

# To get a new package from the internet, for example "numpy", simply go to
# the command line/terminal (Not this python editor, but the actual command line, which conveniently is
# provided in the "Terminal" tab in PyCharm below). In the command line, simply type
# "pip install numpy". Now numpy can be imported:

import numpy as np # The as just defines a shorter name

# Create an 2 by 3 array of random integer
low = 3
high = 8
size = (2, 3)
small_array = np.random.randint(low, high, size)
print('Here\'s a small numpy array\n', small_array)

# Sidenote: from above backspace \ put in front of a character is the
# "escapce character," which makes python interpret the next thing as a string or special text operator. \n makes a line break

# Load a geotiff as a numpy array using GDAL

from osgeo import gdal
geotiff_filename = 'carbon_conserved.tif'

# First, open the gdal dataset
carbon_conserved_dataset = gdal.Open(geotiff_filename)

# The dataset object holds information about the area and extent of the data, or the geotransform information
geotransform = carbon_conserved_dataset.GetGeoTransform()
projection = carbon_conserved_dataset.GetProjection()
# print('GDAL dataset geotransform', geotransform

# IMPORTANT ANNOYING NOTE: in programming, there are different conventions for identifying a place by rows, cols vs. x, y vs. upper-left, lower-right, etc.
# Numpy is denoted row, col but gdal is denoted X, Y (which flips the order). Just memorize that row = Y and col = X.

n_rows = carbon_conserved_dataset.RasterYSize
#print('Number of rows in a GDAL dataset', n_rows)

n_cols = carbon_conserved_dataset.RasterXSize
#print('Number of columns in a GDAL dataset', n_cols)

# Next, get the "band" of the dataset. Many datasets have multiple layers (e.g. NetCDFs).
# Geotiffs only have 1 band by default, so we just grab band 1
carbon_conserved_band = carbon_conserved_dataset.GetRasterBand(1)

# The band object has information too, like the datatype of the geotiff:
data_type = carbon_conserved_band.DataType
no_data_value = carbon_conserved_band.GetNoDataValue()

# Finally, we can get the array from the band as a numpy array:
c = carbon_conserved_band.ReadAsArray()
shape = c.shape

# print('Look at the array itself', c)
# print('Add up the array', np.sum(c))

# Make a copy in memory for us to play with. NOTE that if we just did c_view = c and then modified c_view, the c array would also be changed.
c_view = c # This only creates a new pointer to the same block of memory on your computer that holds the array. If we change c_view, c will also be changed.
c_calcs = c.copy() # This gives us a NEW array in a new block of memory, so changing c_calcs will not change c.

# Get specific elements in the array with [row, col]
specific_value = c_calcs[400, 500]

# Or you can get values between a range of rows and cols with :
chunk_of_array = c_calcs[1000:1100, 1600:1700] # This would give you a 100 by 100 subarray

# Or you can select out a subset of the array based on a logic conditional
conditional_subset = c_calcs[c_calcs>10000]

# Note that when we took the conditional subset, the array dimensions no longer made sense (there now are unspecified missing locations).
# Numpy deals with this by flattening the array to 1 dimension.
#print('conditional_subset shape', conditional_subset.shape)

# But, if we don't save it as a new array (and do something like reassigning values), it retains the array's shape.
# print('Sum of c_calcs before changing values', np.sum(c_calcs))

# Change all values in c_calcs that are > 10000 to 22 IN-PLACE (i.e., changes the underlying c_calcs array).
c_calcs[c_calcs>10000] = 22
# print('Sum of c_calcs after changing values', np.sum(c_calcs))

# Set c_calcs back to the original by taking a new copy
c_calcs = c.copy()

# If you dont want to overwrite c_calcs, the above method won't work unless you create another copy first.
d = c_calcs.copy()

d[(d > 200) & (d < 10000)] = 33 # Note, unlike vanilla python, Numpy conditionals here must use & and must be in parenthases.
# print('The sum of d after we messed with it', np.sum(d))

# Save the as a new geotiff to disk

# Create a new filename for our output file. The + concatenates things. Str() makes the number a string.
# This is one of those cases where python wouldn't correctly guess the data type
output_filename = 'gdal_created_array_' + str(random.randint(1, 1000000)) + '.tif'

# Create a new file at that filename location using the attributes we used above
# Notice that we flipped n_cols and n_rows from how numpy would have wanted it.
output_dataset = gdal.GetDriverByName('GTiff').Create(output_filename, n_cols, n_rows, 1, data_type)

# Set dataset-level information
output_dataset.SetGeoTransform(geotransform)
output_dataset.SetProjection(projection)

# Now get a band from our new dataset on which we'll write our array.
output_band = output_dataset.GetRasterBand(1)

# Do the array writing
output_band.WriteArray(d)

# Set any final band-level information
output_band.SetNoDataValue(no_data_value)

import pandas

food_prices = pandas.read_csv('world_monthly_food_prices.csv')
# print('Whole dataframe:', food_prices)
# print('List of column names:', food_prices.columns)
# print('Specific column:', food_prices['Value'])
# print('Specific value in that column:', food_prices['Value'][6])

import matplotlib
from matplotlib import pyplot as plt
# plt.plot(food_prices['Value'])
# plt.show()
#
# plt.imshow(c)
# plt.show()

# Sightly more complex example
# Create a new figure and axes.
fig, ax = plt.subplots()

# Make up some data
data = np.clip(np.random.randn(250, 250), -1, 1)

# Use the axes object to show the data with a coolwarm colorbar
cax = ax.imshow(data, interpolation='nearest', cmap=matplotlib.cm.coolwarm)

# Give a title to the axis.
ax.set_title('Gaussian noise with vertical colorbar')

# Add a colorbar to  the figure
cbar = fig.colorbar(cax, ticks=[-1, 0, 1]) # Add colorbar, make sure to specify tick locations to match desired ticklabels

# Modify the axes within the colorbar
cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

# Show it.
#plt.show()






























