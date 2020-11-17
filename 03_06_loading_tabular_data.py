import os

import numpy as np
import pandas as pd

# Creating a Series by passing a list of values, letting pandas create a default integer index:
s = pd.Series([1, 3, 5, np.nan, 6, 8])

dates = pd.date_range('20130101', periods=6)

np.random.seed(48151623)






# Creating a DataFrame by passing a NumPy array, with a datetime index and labeled columns:
df = pd.DataFrame(np.random.randn(6, 4), columns=list('ABCD'))
print(df)


df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3] * 4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})


df.head()


print(df.index)

print(df.columns)



df.to_numpy()
df.describe()

"""
Note
While standard Python / Numpy expressions for selecting and setting are intuitive 
and come in handy for interactive work, for production code, we recommend the 
optimized pandas data access methods, .at, .iat, .loc and .iloc.
"""

# Different for index cause index is key
df.sort_index(axis=1, ascending=False)

df = df.sort_values(by='B')


## Selecting by LABELS
# Selecting a single column, which yields a Series, equivalent to df.A
df['A']
df.A
# Selecting via [], which slices the rows.
df[0:3] # CAN BE SLOW

r = df.loc[0]
# Discuss difference between df['A'] and df.loc[0]

r = df.loc[0, 'A']

r = df.loc[:, 'A']

# OPTIMIZATION:
# for faster single point access, use:
r = df.at[0, 'A']

# SELECTING BY POSITION
r = df.iloc[3]

r = df.iloc[3:5, 0:2]

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
print(df1)

df.mean()
df.mean(1) # other axis

# Apply
df.apply(np.cumsum)
df.apply(lambda x: x.max() - x.min())

# Concat
s = pd.Series(range(0, 6))
print('s', s)

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

print(left)
print(right)

df = pd.merge(left, right, on='key')


# # Grouping
# df.groupby('A').sum()
# df.groupby(['A', 'B']).sum()



# Stacking
print('df:\n', df)
stacked = df.stack()
print('stacked:\n', stacked)


# Pivot Tables
df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 3,
                   'B': ['A', 'B', 'C'] * 4,
                   'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,
                   'D': np.random.randn(12),
                   'E': np.random.randn(12)})
print(df) # SPREADSHEET VIEW
df = pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])
print(df) # Multiindexed (Pivot table) view.

# NOTICE that a pivot table is just the above date but where specific things have been made into multi-level
# indices.

# PLOTTING
ts = pd.Series(np.random.randn(1000),
            index=pd.date_range('1/1/2000', periods=1000))

ts = ts.cumsum()
ts.plot()
import matplotlib.pyplot as plt
plt.show()



df.to_csv('foo.csv')
df.to_excel('foo.xlsx', sheet_name='Sheet1')
df.to_hdf('foo.h5', 'df')

pd.read_excel('foo.xlsx', 'Sheet1', index_col=None, na_values=['NA'])


# path = "../../Data/WDI_CO2_data.csv"
#
# df = pd.read_csv(path)
# print(df)
#
# # Discuss copy vs view
# # df = df.dropna(inplace=True)
# # print(df) # NONE
#
# # Country Name	Country Code	Series Name	Series Code	1960 [YR1960]	1961 [YR1961]	1962 [YR1962]	1963 [YR1963]	1964 [YR1964]	1965 [YR1965]	1966 [YR1966]	1967 [YR1967]	1968 [YR1968]	1969 [YR1969]	1970 [YR1970]	1971 [YR1971]	1972 [YR1972]	1973 [YR1973]	1974 [YR1974]	1975 [YR1975]	1976 [YR1976]	1977 [YR1977]	1978 [YR1978]	1979 [YR1979]	1980 [YR1980]	1981 [YR1981]	1982 [YR1982]	1983 [YR1983]	1984 [YR1984]	1985 [YR1985]	1986 [YR1986]	1987 [YR1987]	1988 [YR1988]	1989 [YR1989]	1990 [YR1990]	1991 [YR1991]	1992 [YR1992]	1993 [YR1993]	1994 [YR1994]	1995 [YR1995]	1996 [YR1996]	1997 [YR1997]	1998 [YR1998]	1999 [YR1999]	2000 [YR2000]	2001 [YR2001]	2002 [YR2002]	2003 [YR2003]	2004 [YR2004]	2005 [YR2005]	2006 [YR2006]	2007 [YR2007]	2008 [YR2008]	2009 [YR2009]	2010 [YR2010]	2011 [YR2011]	2012 [YR2012]	2013 [YR2013]	2014 [YR2014]	2015 [YR2015]	2016 [YR2016]	2017 [YR2017]	2018 [YR2018]	2019 [YR2019]	2020 [YR2020]
# cols = list(df.columns)
# df = df[['Country Code', '1970 [YR1970]']]
#
# df = df.dropna()
# print(df)
#
# # df = df.iloc['Total greenhouse gas emissions (kt of CO2 equivalent)'
#
# # 1 country
# #
# import matplotlib.pyplot as plt
#
# # plt.plot(df['1970 [YR1970]']) # DONT DO IT!
# plt.ylabel('some numbers')
# plt.show()
#
# # fig, ax = plt.subplots()
# # plt.show()

