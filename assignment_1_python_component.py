# Assignment 1 python component
import os.path

assignment_summary = """
For this assignment, you will be tasked with some basic python comprehension tests. You will
turn in 2 documents to complete this part of the assignment: the code you used (which can be
built into this exact python file for convenience), and a Word or PDF document that contains the
output for each exercise.

Finally, also use comments to explain steps you are doing.
no need to go overboard but it is always important in coding to be as descriptive as possible.
Most things in this can be solved using methods shown in lecture. However, some cases will
require you to search the internet for answers. This is intended because efficiently searching
documentation or Stackoverflow is a requirement of modern programing.

Subsequent assignments will be specific to big data, statistics and economics, but
for now we are just checking that you're okay with the basics of Python.
"""

## Exercise 1:
exercise_1 = """
Write a python function that calculates the sum of all squared 
numbers from 1 to x. Illustrate that it works for x = 20.

HINT, ** is the exponent operator in python.
HINT syntax for a python function is:

def function_name(variable_name):
    outcome = variable_name + 1
    return outcome

"""

# Exercise 2: Filesystems
exercise_2 = """
The python library named "os" is a built-in library for dealing with any file on your operating system.
Often, research tasks involve LOTS of files and you need to iterate over them. To show you know how
to do this, use the os.listdir function to answer the following questions. Note that you need to 
write "import os" to import the library into your code before you can use it:

part a
Print out a list of all the files in the class Code directory (which you have gotten from Google Drive)
I don't care how many are actually there (in case you've added some yourself) but show me how.

part b
Using a for loop over that list and the function len(), count how many letters there are in the filename 
(excluding the folders leading to the file and its extension.). HINT just like in real life,
you may need to google the len() function and see how it work on e.g. strings.

"""

## Exercise 3
exercise_3 = """
Write a Python program which iterates the integers from 1 to 50. 
For multiples of three print "Fizz" instead of the number and for 
the multiples of five print "Buzz". For numbers which are multiples of both three 
and five print "FizzBuzz". Hint: look up how to use the modulo operator (which is written as % in python)
which would let you test when the remainder of a devision is exactly 0.

Side-note, this is a hilariously over-used question that most software engineers get asked
on their first interview for a job.
"""

## Exercise 4
exercise_4 = """
Replicability of science means someone needs to be able to replicate your code. 
For this to work, your code should have everything written with "relative paths"
rather than "absolute paths", which means someone on a different computer
with a different file structure can still run your code so long as needed files
are located at the same place RELATIVE to where your script is being run.

This question will use some python functions to test that you know the difference 
between absolute and relative (which you can easily google if you don't already know).

part a: print out the full path to this current python script you are writing in.
you can use the built-in variable __file__ which contains the script file and 
is automatically loaded when you run the script.

part b: print out the relative path name to the same file using the function os.path.relpath()

part c: print out the absolute path name to the same file using the function os.path.abspath()

part d: If you haven't already, download the Data directory from the class's Google Drive.
Place the Data directory on your hard-drive. Good practice would be to place it in the same
parent folder as your Code directory. Thus, your structure would look like this
( - denotes its a subdirectory of the previous level):

Class Folder
-Code
--assignment_1_python_component.py
--other files...
-Data
--here would be all the data files in the Data directory.

You don't have to do it this way, but it may be easiest.

Using only a relative path, print out the names of any/all files in the Data directory
using os.listdir()

HINT: if you want to go "up" a level in the folder structure and
still be a relative path, you can use ../

Thus, os.listdir('../') would return everything in the parent directory of your working directory
and os.listdir('../../') would return everythin in the parent of the parent (the grandparent).
and os.listdir('../../Apples') would return everything in a folder called apples located 
in the grandparent directory.

Apologies if this seems like a simple task, but TONS of errors in problem sets (or real-life code) are from messed up
relative paths.
"""









