# For spatial data, the amazing workhorse is GDAL. You might remember this name from RGDAL.
# The open-source scientific computing stack is all interlinked.

# We will need to start by ensuring you have gdal. So, open up that anaconda prompt/terminal
# and do the command "conda install gdal -c conda-forge"

# Now import gdal and load a geotiff as a numpy array using GDAL
from osgeo import gdal
import numpy as np
import os, random
geotiff_path= '../../Data/maize_Production.tif'

# First, open the gdal dataset
crop_production_tons_per_cell = gdal.Open(geotiff_path)

# The dataset object holds information about the area and extent of the data, or the geotransform information
geotransform = crop_production_tons_per_cell.GetGeoTransform()
projection = crop_production_tons_per_cell.GetProjection()

# print('GDAL dataset geotransform', geotransform)
# print('GDAL dataset projection', projection)

# IMPORTANT ANNOYING NOTE: in programming, there are different conventions for identifying a place by rows, cols vs. x, y vs. upper-left, lower-right, etc.
# Numpy is denoted row, col but gdal is denoted X, Y (which flips the order). Just memorize that row = Y and col = X.

n_rows = crop_production_tons_per_cell.RasterYSize
#print('Number of rows in a GDAL dataset', n_rows)

n_cols = crop_production_tons_per_cell.RasterXSize
#print('Number of columns in a GDAL dataset', n_cols)

# Next, get the "band" of the dataset. Many datasets have multiple layers (e.g. NetCDFs).
# Geotiffs only have 1 band by default, so we just grab band 1
carbon_conserved_band = crop_production_tons_per_cell.GetRasterBand(1)

# The band object has information too, like the datatype of the geotiff:
data_type = carbon_conserved_band.DataType
no_data_value = carbon_conserved_band.GetNoDataValue()

print('data_type', data_type)
print('no_data_value', no_data_value)

# Finally, we can get the array from the band as a numpy array:
array = carbon_conserved_band.ReadAsArray()
shape = array.shape



print('Look at the array itself', array)

import matplotlib.pyplot as plt
plt.imshow(array)
plt.show()

print('Add up the array', np.sum(array))
print('Add up the nan array', np.nansum(array))

# array = np.where(, 0, array)

mask = np.isnan(array)
array[mask] = 0.0
print('Add up the masked array', np.sum(array))

plt.imshow(array)
plt.show()

print(np.unique(array, return_counts=True))

# Make a copy in memory for us to play with. NOTE that if we just did c_view = c and then modified c_view, the c array would also be changed.
c_view = array # This only creates a new pointer to the same block of memory on your computer that holds the array. If we change c_view, c will also be changed.
c_calcs = array.copy() # This gives us a NEW array in a new block of memory, so changing c_calcs will not change c.

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
c_calcs = array.copy()

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

# Finally, and very importantly, clean up after yourself. It wont actually write until the resources in
# memory have been released.

d = None
output_band = None
output_dataset = None





 # Additional exercises for home.

import pandas

food_prices = pandas.read_csv('../../Data/world_monthly_food_prices.csv')
print('Whole dataframe:', food_prices)
print('List of column names:', food_prices.columns)
print('Specific column:', food_prices['Value'])
print('Specific value in that column:', food_prices['Value'][6])

plt.plot(food_prices['Value'])
plt.show()

# plt.imshow(array)
# plt.show()

# Sightly more complex example
# Create a new figure and axes.
fig, ax = plt.subplots()

# Make up some data
data = np.clip(np.random.randn(250, 250), -1, 1)

# Use the axes object to show the data with a coolwarm colorbar
import matplotlib
cax = ax.imshow(data, interpolation='nearest', cmap=matplotlib.cm.coolwarm)

# Give a title to the axis.
ax.set_title('Gaussian noise with vertical colorbar')

# Add a colorbar to  the figure
cbar = fig.colorbar(cax, ticks=[-1, 0, 1]) # Add colorbar, make sure to specify tick locations to match desired ticklabels

# Modify the axes within the colorbar
cbar.ax.set_yticklabels(['< -1', '0', '> 1'])  # vertically oriented colorbar

# Show it.
plt.show()









