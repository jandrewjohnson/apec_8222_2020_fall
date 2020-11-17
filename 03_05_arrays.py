
# Numpy is the highly-optimized, super-fast workhorse that underlies much of the scientific computing stack
# see Harris et al. 2020 in Nature for evidence.

import numpy as np

# We're also going to want to look at matrices, so let's import matplotlib too.
# this import statement is more complex, but of key importance, it creates a plt object we'll use later
import matplotlib.pyplot as plt

# Create a 1 dimensional array 15 elements long. np.arange is a FAST version of python range()
a = np.arange(15)

print(a) # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]

# Reshape our array into a 2 dimensional array of size 3 by 5.
# NOTE, numpy, and about 60% of computer programming, denotes things in terms of Row then Column (RC order)
# but some things, especially those that regard displaying pixels, denote things as x, y (which note is CR order)
a = np.arange(15).reshape(3, 5)

print(a)

# [[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]

a.shape # (3, 5)
a.ndim # 2
a.dtype.name # 'int64'
a.itemsize #8
a.size # 15
type(a) # <class 'numpy.ndarray'>

# Uncomment this to see what happens with error handling.
# a = np.array(1,2,3,4)    # WRONG: TypeError: array() takes from 1 to 2 positional arguments but 4 were given
a = np.array([1,2,3,4])  # RIGHT

# 2d version
b = np.array([(1.5,2,3), (4,5,6)])

np.zeros((3, 4))

np.ones((2,3,4), dtype=np.int16 )                # dtype can also be specified

c = np.empty((2,3))                                 # uninitialized
print(c)
# array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],  # may vary
#        [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])

# Array math
a = np.array( [20,30,40,50.] )
b = np.arange( 4 )

c = a-b

print('a', a)
print('b', b)
print('c', c)

d = b**2
print('d', d)

e = 10*np.sin(a)
print('e', e)

f = a<35
print('f', f)


## SLICING

a = np.arange(10)
b = np.arange(12).reshape(3, 4)

# Can access items directly, but need as many indices as there are dimensions
a[2]
b[2, 3]

# Can also access "slices", which are denoted Start, Stop, Stepsize
r = a[1: 9: 2]

# An empty slice argument means use the default
r = a[::2]

# Looping over arrays
for i in a:
    print(i**(1/3.))

# Loop to get the sum of the array
r = 0
for row in b:
    print('row', row)
    for value in row:
        print('value', value)
        r += value



# NOTE: Iterating over arrays here is just for illustration as it is VERY VERY SLOW
# and loses the magic of numpy speed. We'll learn how to bet around this later
# by "vectorizing" functions, which basically means batch calculating
# everything in a vector all in one call. For now, here's an example of the
# much faster version

r = b.sum()

# Plotting with matplotlib
plt.imshow(b)
# plt.show()

ax = plt.imshow(b)
plt.colorbar(ax)
# plt.show()


def mandelbrot(h, w, maxit=20 ):
    """Returns an image of the Mandelbrot fractal of size (h,w)."""
    y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2            # who is diverging
        div_now = diverge & (divtime==maxit)  # who is diverging now
        divtime[div_now] = i                  # note when
        z[diverge] = 2                        # avoid diverging too much

    return divtime


plt.imshow(mandelbrot(400,400))
# plt.show()





print('r', r)


a[:6:2] = 1000



