# Other types

list_1 = [4, 5, 6]

dictionary_1 = {23: "Favorite number", 24: "Second favorite number"}

# Although it is not usually best, you can run specific lines by selecting them and using the shortcut Alt-Shift-E
# or right-click --> Run Selection.


# Here is a multi line string: IDE Fun tricks (also below illustrates improved capabilities of an IDE editor)

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


# Functions
def my_function(input_parameter_1, input_parameter_2):
    product = input_parameter_1 * input_parameter_2
    return product


output = my_function(4, 5)

print(output)


# Looping.
# Example here will calculate the average of every third integer between 100 and 1000.
# The range() funciton is a python built-in that lets you iterate through a range of numbers.

sum = 0 # Set the initial variable values
num = 0

# Here is a for loop. Also note that python EXPLICITLY USES TAB-LEVEL (the inner part of the loop is tabbed 1 level up)
for i in range(100, 1000, 3):
    sum = sum + i
    num = num + 1

mean = sum / num
print(mean)

# In class activity: Combine the function definition process with a loop to calculate the Sum of Squared Numbers
# from 1 to 100
# HINT, ** is the exponent operator in python.
#
# BONUS: Make sure you're actually right by inserting a print statement in each step.




