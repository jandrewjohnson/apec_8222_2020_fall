## Importing packages

### Built-in packages via the Python Standard Library
import math
import os, sys, time, random

number_rounded_down = math.floor(4.678)
# Remember, from here on, output print statements will start with a comment (to deactivate them) so that your output isn't overwhelming.
# when you get to a line that you actually want to see, uncomment it by deleting the #.

print('number_rounded_down', number_rounded_down)

### Using packages from elsewhere

# To get a new package from the internet, for example "numpy", simply go to
# the Conda Command line/terminal (Not this python editor, but the Conda command line we were using earlier)
# From the conda command line, activate your environment, then use the command:
#  conda install numpy -c conda-forge

import numpy as np # The as just defines a shorter name

# Create a 2 by 3 array of random integer
low = 3
high = 8
size = (2, 3)
small_array = np.random.randint(low, high, size)

# TIP: Hit Ctrl-B on a definition of a function to go directly to the code

print('Here\'s a small numpy array\n', small_array)

# Sidenote: from above backspace \ put in front of a character is the
# "escape character," which makes python interpret the next thing as a string or special text operator. \n makes a line break

# While we're at it, also conda install pandas, scikit-learn, matplotlib, gdal
# using command:
# conda install gdal scikit-learn pandas matplotlib numpy -c conda-forge

# To confirm everything worked, uncomment and run the next line. Notice that scikit-learn's name has been shortened
import numpy, pandas, sklearn, matplotlib
from osgeo import gdal



