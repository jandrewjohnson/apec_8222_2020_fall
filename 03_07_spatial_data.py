# Author: Justin A Johnson. Adapted from sklearn documentation and original content. License: Modified BSD License.

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
maize_production_tons_per_cell = gdal.Open(geotiff_path)

# The dataset object holds information about the area and extent of the data, or the geotransform information
geotransform = maize_production_tons_per_cell.GetGeoTransform()
projection = maize_production_tons_per_cell.GetProjection()

print('GDAL dataset geotransform', geotransform)
print('GDAL dataset projection', projection)

# IMPORTANT ANNOYING NOTE: in programming, there are different conventions for identifying a place by rows, cols vs. x, y vs. upper-left, lower-right, etc.
# Numpy is denoted row, col but gdal is denoted X, Y (which flips the order). Just memorize that row = Y and col = X.

n_rows = maize_production_tons_per_cell.RasterYSize
print('Number of rows in a GDAL dataset', n_rows)

n_cols = maize_production_tons_per_cell.RasterXSize
print('Number of columns in a GDAL dataset', n_cols)

# Next, get the "band" of the dataset. Many datasets have multiple layers (e.g. NetCDFs).
# Geotiffs can have multiple bands but often have just 1. For now, grab band 1
maize_production_tons_per_cell_band = maize_production_tons_per_cell.GetRasterBand(1)

# The band object has information too, like the datatype of the geotiff:
data_type = maize_production_tons_per_cell_band.DataType
no_data_value = maize_production_tons_per_cell_band.GetNoDataValue()

print('data_type', data_type)
print('no_data_value', no_data_value)

# Finally, we can get the array from the band as a numpy array:
array = maize_production_tons_per_cell_band.ReadAsArray()
shape = array.shape

print('Look at the array itself', array)

import matplotlib.pyplot as plt

plt.imshow(array)
plt.title('Maize production')
# plt.show()

# Depending on your setup and operating system, some people might not be having their plots
# show up like mine. A useful approach that gets around this is saving each plot as a png:
plt.savefig('maize_production.png')

# Other things you can do are use the numpy vectorized (fast) functions just like with Pandas.
# However, spatial data often have Not-a-Number NaN Values

# print('Add up the array', np.sum(array))

# We can fix this with np.nansum()

# print('Add up the nan array', np.nansum(array))

# Additionally, we could do it manually by
# creating a logical mask array, which would have a True False value for each pixel
# depending on nan status. Numpy of course has a builtin functino for this.
mask = np.isnan(array)

# You could also use this mask to assign a value. Here we replace all NaNs with zero
array[mask] = 0.0
print('Add up the masked array', np.sum(array))

# array = np.where(, 0, array)

plt.imshow(array)
plt.show()

# A common task is to want to see each unique value in an array. As below.
print(np.unique(array, return_counts=True))

# BIG DATA IMPLICATION: Copies versus Views in mumpy.
# Make a copy in memory for us to play with.
# NOTE that if we just did c_view = c and then modified c_view,
# the c array would also be changed.

# This only creates a new pointer to the same block of memory on your computer that holds the array. If we change c_view, c will also be changed.
c_view = array

# This gives us a NEW array in a new block of memory, so changing c_calcs will not change c.
d = array.copy()

# QUICK REVIEW ON ARRAY NOTATION, which is identical from before for 2-dim arrays.
# Get specific elements in the array with [row, col]
specific_value = d[400, 500]

# Or you can get values between a range of rows and cols with :
chunk_of_array = d[1000:1100, 1600:1700] # This would give you a 100 by 100 subarray

#  Note, unlike vanilla python, Numpy conditionals here must use & and must be in parenthases.
d[(d > 200) & (d < 10000)] = 33
print('The sum of d after we messed with it', np.sum(d))

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





