## Other types

# Reminder, this assumes you have setup an envioronment with conda using:
# conda create --name py38 python=3.8
# and that you have then activated it:
# conda activate py38
list_1 = [4, 5, 6]
print('list_1', list_1)

dictionary_1 = {23: "Favorite number", 24: "Second favorite number"}
print('dictionary_1', dictionary_1)

# Although it is not usually best, you can run specific lines by selecting them and using the shortcut Alt-Shift-E
# or right-click --> Run Selection.

# Here is a multi line string: (also discusses improved capabilities of an IDE editor)

ide_fun_tricks = """
1.) Move back and forth in your history of cursor positions (using your mouse forward and back buttons)
2.) Edit on multiple lines at the same time (hold alt and click new spots)
3.) Smartly paste DIFFERENT values
4.) Duplicate lines (ctrl-d)
5.) Introspection (e.g., jump between function definition and usages)
6.) Debugging (Interactively walk through your code one line at a time)
7.) Profiling your code (see which lines take the most time to compute.)
8.) Keep track of a history of copy-paste items and paste from past copies. (ctrl-shift-v)
"""


## Functions

def my_function(input_parameter_1, input_parameter_2):
    product = input_parameter_1 * input_parameter_2
    return product


function_output = my_function(4, 5)

# print('function_output', function_output)

##Looping.

# Example here will calculate the average of every third integer between 100 and 136.
# The range() function is a python built-in that lets you iterate through a range of numbers.

small_range = range(0, 10)
print('small_range:', small_range)

small_range_as_list = list(range(0, 10))
# print('small_range_as_list:', small_range_as_list)

big_range = range(0, 100000000000000000000000000000000000000000) # This would be over 1 dodecatillion integegers, larger than the number of atoms in the galaxy.
print('big_range', big_range)

medium_range_as_list = list(range(0, 1000000))
# print('medium_range_as_list:', medium_range_as_list)

# DO NOT RUN THIS. YOUR COMPUTER WILL LITERALLY EXPLODE.
# big_range_as_list = list(range(0, 100000000000000000000000000000000000000000))
# print('big_range_as_list', big_range_as_list)

sum = 0 # Set the initial variable values
num = 0

# Here is a for loop. Also note that python EXPLICITLY USES TAB-LEVEL (the inner part of the loop is tabbed 1 level up)
for i in range(100, 136, 3):
    sum = sum + i
    num = num + 1
    print('sum', sum)

mean = sum / num
print('mean', mean)

## In class activity: Combine the function definition process with a loop to calculate the Sum of Squared Numbers
# from 1 to 100
# HINT, ** is the exponent operator in python.
#
# BONUS: Make sure you're actually right by inserting a print statement in each step.




