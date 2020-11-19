## PYTHON BASICS

# Comments: The hashtag makes the rest of the line a comment.
# The more programming you do, the more you focus on making good comments.

# Assign some text (a string) to a variable
some_text = 'This is the text.'
other_text = "Could also use double-quotes too."

concatenated_string = some_text + ' ' + other_text
print(concatenated_string)

# At this point, run the script (CTRL SHIFT F10).


# Assign some numbers to variables
a = 5  # Here, we implicitly told python that a is an integer
b = 4.6  # Here, we told python that b is a floating point number (a decimal)

# NOTE: I have included lots of commented-out print statements so that you only see what you want to.
# If you want to actually see what a and b look like, uncomment them and run.
# Keyboard shortcut: Ctrl /

print('a', a)
print('b', b)

# Python can be a calculator. Notice that it can add the integer and the float.
# Python "smartly" redefines variables so that they work together.
# This is different from other languages which require you to manually manage
# the "types" of your variables.
sum_of_two_numbers = a + b

# Printing output to the console
print('Our output was', sum_of_two_numbers)

# Now is also a good time to test out error-handling. Uncomment this if you want to see.
# print('Our output was', sum_of_two_blunders)
