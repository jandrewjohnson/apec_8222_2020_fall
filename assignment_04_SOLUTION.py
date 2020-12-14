# Assignment 1

# This assignment will start with basic python tasks that a big-data economist might face. Questions will ask
# for you to print things to show that you got the answer right. For full credit, make sure your print
# statements are descriptive enough that I know which answer it was a part of, e.g.:
# print("Question 1a:", answer)
# Finally, also use comments to explain steps you are doing.
# no need to go overboard but it is always important in coding to be as descriptive as possible.

# Problem 1: Filesystems (30 pts)

# The python library named os is a built-in library for dealing with any file on your operating system.
# Often, research tasks involve LOTS of files and you need to iterate over them. To show you know how
# to do this, use the os.listdir function to answer:

# QUESTION 1a - 5 pts
# Print out a list of all the files in the class Data directory (which you have gotten from Google Drive)
# I don't care how many are actually there (in case you've added some yourself) but show me how.
# Reminder: Use a relative path so that when I run this script in the expected working directory, it is able to find
# the files on MY computer. This means your folders should be arranged like this (how I had said in lecture)

# - CLASS DIRECTORY
# -- Data
# -- Code
# --- python_for_big_data
# ---- assignment_4p.py

import os
data_dir = '../../Data'
files = os.listdir(data_dir)
print(files)

# Question 1b - 5 pts
# Using a for loop over that list, count how many times the character string "maize" occurs in the file NAMES
# (excluding the folders leading to the file)

counter = 0
for i in files:
    if "maize" in i:
        counter += 1
print('Maize occurs:', counter)


# Question 1c - 10 pts
# Using os.path.splitext(), count how many files have the ".tif" extension
# Note, obviously you could count it visually, but do it programatically.
# Note that os.path.splitext() returns list 2 elements long and only 1 element of those is
# what you want.

num_tifs = 0
for file in files:
    if os.path.splitext(file)[1] == '.tif':
        num_tifs += 1
print('num_tifs', num_tifs)

# Question 1d - 10 pts
# Using stackexchange (or wherever) to find the right functions, create a list
# of all the filesizes in the Data directory sorted largest to smallest. Note that to get the filesize, you need to point
# the funciton to the whole filepath (directory and filename). There is a handy os.path function that joins
# directories and filenames. If you want to impress me (no extra points assigned),
# Print out a sorted list of filename, filesize.

sizes = []
filenames_and_sizes = []
for file in files:
    path = os.path.join(data_dir, file)
    size = os.path.getsize(path)
    sizes.append(size)
    filenames_and_sizes.append((path, size))

sorted_sizes = sorted(sizes)

sorted_filenames = sorted(filenames_and_sizes, key=lambda x: x[1])

print('sorted_sizes', sorted_sizes)
print('sorted_filenames', sorted_filenames)

## Question 2: 70 pts

# Question 2a - 10 pts
# Using Pandas, read in the Production_Crops_E_All_Data_(Normalized).csv file. Note that
# When unzipped this file is 253 MB. This is not HUGE data but it's pretty big and would not be able to be
# Fully opened in Excel. I guess this means we are finally, officially, entering the territory of "BIG DATA"
# Insofar as some tools will now fail. Also note that I've given you a CSV exactly as it is provided by
# the FAO and that this file was "encoded" in a strange way. You will need to give an
# encoding='latin' argument to tell your computer how to read the file. This is a common problem
# with using external data so I thought I'd have you work through the solution.
# Save the column headers to a list and print it out. Also print out the number of rows.

import pandas as pd
import numpy as np

input_path = os.path.join(data_dir, 'Production_Crops_E_All_Data_(Normalized).csv')
df = pd.read_csv(input_path, encoding='latin')
columns = list(df.columns)
print('columns', columns)
print('df num_rows', len(df.index))

# Question 2b - 10 pts
# Pare down the dataframe to only have Production as the element. (Element is variable name in FAO
# lingo). In other words, reduce the size of the DF so that it only has Production statistics,
# rather than additional variables like Harvested Area.
# You can do this with the df.loc method

production_df = df.loc[df['Element'] == 'Production']
print('production_df num_rows', len(production_df.index))

# Question 2c - 10 pts
# Use pandas unique() function to get a list of all Area names used in this table and all the Item names used.
# print these both out.
country_names = pd.unique(production_df['Area'])
print('country_names', country_names)

item_names = pd.unique(production_df['Item'])
print('item_names', item_names)

# Question 2d - 10 pts
# Produce a line-graph of Production from 1961 to 2019 of Maize in Canada. To do this, you may want to first
# to use .loc again to pare down the dataframe to only have the required info. For plotting, use matplotlib.
# Hint: the import statement for matplotlib is
# import matplotlib.pyplot as plt
# don't forget to call plt.show() or otherwise it plots it but never actually displays it.

item_df = production_df.loc[production_df['Item'] == 'Maize']
print('item_df num_rows', len(item_df.index))

country_df = item_df.loc[item_df['Area'] == 'Canada']
print('country_df', country_df)

import matplotlib.pyplot as plt
plt.plot(country_df['Year'], country_df['Value'])
plt.title('Production of Maize in Canada')
plt.xlabel('Year')
plt.ylabel('Tonnes')
plt.show()

# Question 2e - 10 pts
# Notice that data are in vertical, stacked, etc. format. When dealing with multiple countries, it may be
# useful to put years into different column headers. This is referred to as unstacking
# the data. Modify one of your previous steps to create a dataframe that has each Area in a unique
# row and each year in a unique column. A useful command might be pd.pivot_table()
# Print out this new dataframe, but also save it to a file named "question_2e.csv" in the current
# working directory.

years = pd.unique(country_df['Year'])

item_pivot_df = pd.pivot_table(item_df, values=['Value'], index=['Area','Area Code', 'Item Code', 'Element', 'Unit', ], columns=['Year'])
print('item_pivot_df', item_pivot_df)

item_pivot_df.to_csv(os.path.join('question_2e.csv'))

# Question 2f - 10 pts
# Create an unstacked dataframe similar to above except with ALL of the different crops (Items) included.
# If you saved previous steps' dataframes, you might already have the one you need to pivot on, otherwise recreate it.
# Print and save this to question_2f.csv

fuller_pivot_df = pd.pivot_table(production_df,
                                 values=['Value'],
                                 index=['Area','Area Code', 'Item Code', 'Item', 'Element', 'Unit', ],
                                 columns=['Year'],
                                 aggfunc=np.sum) # NOTE: excluding aggfunc means that things with no entries will be excluded.

print('fuller_pivot_df', fuller_pivot_df)
print('fuller_pivot_df num_rows', len(fuller_pivot_df.index))
fuller_pivot_df.to_csv(os.path.join('question_2f.csv'))

# Question 2g - 10 pts
# Plot the total production tonnage of all crops for all countries over time. Save this figure as  question_2g.png
# Consider using plt.savefig(). Also note that if you call savefig after you called plt.show() on this
# plot, the saved file will be blank because .show() clears what you plotted.

aggregated_df = pd.pivot_table(production_df,
                                 values=[ 'Value'],
                                 index=['Year',],
                                 columns=[],
                                 aggfunc=np.sum)
print('aggregated_df', aggregated_df)
print('aggregated_df num_rows', len(aggregated_df.index))

plt.plot(aggregated_df['Value'])
plt.title('Production tonnage of all crops over time')
plt.xlabel('Year')
plt.ylabel('Tonnes')
# plt.show()
plt.savefig('question_2g.png')



