import os

import numpy as np
import pandas as pd

# Set a seed value for the random generator
np.random.seed(48151623)

# Creating a Series by passing a list of values, letting pandas create a default integer index:
s = pd.Series([1, 3, 5, np.nan, 6, 8])

# Pandas is very detailed in dealing with dates and all the quirks (leap year?) that this leads to.
dates = pd.date_range('20130101', periods=6)

# Creating a DataFrame by passing a NumPy array, with a datetime index and labeled columns:
df = pd.DataFrame(np.random.randn(6, 4), columns=list('ABCD'))
print('df:\n', df)


df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})

# df.head()
# print(df.index)
# print(df.columns)
df.describe()

# Also note that a dataframe is really just a numpy array dressed up with extra trappings. If you want you
# can get back the raw array (though this might lose a lot of functionality).
a = df.to_numpy()
print('a\n', a)

# Sorting Values:

# Also, I want to illustrate THE MOST COMMON MISTAKE people make with Pandas.

# The sort_values method (a method is just a function attached to an object) returns a NEW modified dataframe.
# Thus, in the line below, if you just printed df, it would not be sorted because we didn't use the returned value.
df.sort_values(by='B')
# print('Not sorted:\n', df)

# Easy way to get around this is just to assign the returned dataframe to a variable (even the input variable)
df = df.sort_values(by='B')
# print('Sorted with return:\n', df)

# Alternatively, if you hate returning things, there is the inplace=True command, which will modify the df ... inplace.
df.sort_values(by='B', inplace=True)
# print('Sorted inplace:\n', df)

## Selection/subsetting of data

# Selecting a single column, which yields a Series, equivalent to df.A
df['A']
df.A

# Selecting via [], which slices the rows.
df[0:3] # CAN BE SLOW

# Note, slicing above, which uses the
# standard Python / Numpy expressions for selecting and setting are intuitiveits best to use
# the optimized pandas data access methods, .at, .iat, .loc and .iloc.

## Selecting by LABELS, loc and iloc

r = df.loc[0] # 0-th row.

# print('r', r)

# Discuss difference between df['A'] and df.loc[0]
r = df.loc[0, 'A']

r = df.loc[:, 'A'] # Colon is a slice, an empty colon means ALL the values.

# OPTIMIZATION:
# for faster single point access, use:
r = df.at[0, 'A']

# SELECTING BY POSITION
r = df.iloc[3]

# Selecting with slices
r = df.iloc[3:5, 0:2]

# Slices again with an empty slice.
r = df.iloc[1:3, :]

r = df.iloc[:, 1:3]

# SIMILAR OPTIMIZATION:
r = df.iat[1, 1]

# Boolean indexing
# Using a single column’s values to select data.
r = df[df['A'] > 0]

# Make a copy (why?) and add a column
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
r = df2[df2['E'].isin(['two', 'four'])]


# Setting by assigning with a NumPy array:
df.loc[:, 'D'] = np.array([5] * len(df))

# Missing data

# First we're going to create a new df by "reindexing" the old one, which will shuffle the data into a new
# order according to the index provided. At the same time, we're going to add on a new, empty column
# EE, which we set as 1 for the first two obs.

df1 = df.reindex(index=[2, 0, 1, 3], columns=list(df.columns) + ['EE'])
df1.loc[0:1, 'EE'] = 1
# print(df1)

# Apply: Similar to R. Applies a function across many cells (fast because it's vectorized)
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())

# Concat
s = pd.Series(range(0, 6))
# print('s', s)

r = pd.concat([df, s]) # Concatenate it, default is by row, which just puts it on the bottom.

r = pd.concat([df, s], axis=1) # Concatenate as a new column

# print(r) # Result when concatenating a series of the same size.

s = pd.Series(range(0, 7))
r = pd.concat([df, s], axis=1) # Concatenate as a new column

s = pd.Series(range(0, 2))
r = pd.concat([df, s], axis=1) # Concatenate as a new column

# Join
# SQL style merges. See the Database style joining section.

left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})

# print(left)
# print(right)

df = pd.merge(left, right, on='key')

# print('df:\n', df)

# Stacking
stacked = df.stack()
# print('stacked:\n', stacked)


# Pivot Tables
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})

# print(df) # SPREADSHEET VIEW
df = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
# print(df) # Multiindexed (Pivot table) view.

# NOTICE that a pivot table is just the above date but where specific things have been made into multi-level
# indices.

# PLOTTING
ts = pd.Series(np.random.randn(1000),
            index=pd.date_range('1/1/2000', periods=1000))

ts = ts.cumsum()
ts.plot()
import matplotlib.pyplot as plt
# plt.show()


# Writing to files

df.to_csv('foo.csv')
# df.to_excel('foo.xlsx', sheet_name='Sheet1') # PROBABLY will get ModuleNotFoundError: No module named 'openpyxl'. Conda install it.

# Reading files:

# FIRST NOTE, here we are using relative paths (which you should almost always do too). the ../ means go up one level.
# this path works if you organized your data into the folder structure I suggested.
wdi_path = "../../Data/WDI_CO2_data.csv"
df = pd.read_csv(wdi_path)

print('csv read as a df\n', df)

# For reference, here's the Excel version
# df = pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])

cols = list(df.columns)

# Make a subset of only 2 cols
r = df[['Country Code', '1970 [YR1970]']]
# print(r)

r = df.loc[df['Country Code'] == 'CAN']
# print('r', r)

rr = r.loc[df['Series Name'] == 'Total greenhouse gas emissions (kt of CO2 equivalent)']
print(rr)

# Class exercise: Plot the emissions of CO2 for Canada (or whereever I don't care).