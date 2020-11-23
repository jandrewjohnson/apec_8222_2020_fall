## Using Numpy to make and process arrays.

# Numpy is the highly-optimized, super-fast workhorse that underlies much of the scientific computing stack
# see Harris et al. 2020 in Nature for evidence.

import numpy as np

# We're also going to want to look at matrices, so let's import matplotlib too.
# this import statement is more complex, but of key importance, it creates a plt object we'll use later
import matplotlib.pyplot as plt

# Create a 1 dimensional array 15 elements long. np.arange is a FAST version of python range()
a = np.arange(15)

print(a)

plt.plot(a)
plt.show() # As with printing, I've commented out most .show() commands to limit what gets printed. Uncomment to explore.

# Reshape our array into a 2 dimensional array of size 3 by 5.
# NOTE, numpy, and about 60% of computer programming, denotes things in terms of Row then Column (RC order)
# but some things, especially those that regard displaying pixels, denote things as x, y (which note is CR order)
a = np.arange(15).reshape(3, 5)

print(a)

plt.plot(a)
plt.show()

plt.imshow(a)
plt.show()

# The above show probably is not what you expected. It has interpreted it as five different lines, but we probably meant to
# think of it as a matrix. For this, use imshow

# a is an OBJECT, which has lots of useful attributes, such as:

a.shape # (3, 5)
a.ndim # 2
a.dtype.name # 'int64'
a.size # 15
a.itemsize #8 Pro-level question. Why is it 8? Hint 8 * 8 = 64.
type(a) # <class 'numpy.ndarray'>

## Creating an array from values

a = np.array([1,2,3,4])  # RIGHT
# a = np.array(1,2,3,4)    # WRONG: TypeError: array() takes from 1 to 2 positional arguments but 4 were given. Uncomment this to see what happens with error handling.

# 2d version
b = np.array([(1.5,2,3), (4,5,6)])

print('b\n', b)

# Creating an empty array of zeros # NOTICE the extra paranetheses.
np.zeros((3, 4))

# or ones.
np.ones((2, 3), dtype=np.int16)                # dtype can also be specified

# or ones.
r = np.random.random((3, 4))                # dtype can also be specified
# print('r', r)

# Or even faster, just "allocate the memory" with an empty matrix.
c = np.empty((2,3))

# What do you think this will produce? The empty matrix just points to memory, which means that if you look there,
# you will just see whatever was there from before, which will likely look crazy.

print('c:\n', c)
# array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],  # may vary
#        [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])

# Array math
a = np.array([20, 30, 40, 50.])
b = np.arange(4)

c = a-b

print('a', a)
print('b', b)
print('c', c)

# ** is the exponent operator in python
d = b**2
# print('d', d)

# Numpy also has handy array-enabled math operators
e = 10*np.sin(a)
# print('e', e)

# Con also create conditional arrays
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

# A single colon means use the full thing.
r = a[:]

# Using slices, you can also set individual elements in the array:
r[0] = 33
r[1:3] = 44
print('r', r)

# Setting in this way also can be done according to a condition:
r[r <= 6] = 5
print('r', r)

# Finally, an alternate and possibly more powerful way of setting conditional values is the np.where function
# This function sets anywhere greater than 10 to be 12, otherwise it keeps it at whatever value was already in r
r = np.where(r > 10, 12, r)
print('r', r)

# Looping over arrays
for i in a:
    r = i**(1/3.)
    # print('r', r)

# Loop to get the sum of the array
r = 0
for row in b:
    # print('row', row)
    for value in row:
        # print('value', value)
        r += value

print('slow sum', r)

# NOTE: Iterating over arrays here is just for illustration as it is VERY VERY SLOW
# and loses the magic of numpy speed. We'll learn how to bet around this later
# by "vectorizing" functions, which basically means batch calculating
# everything in a vector all in one call. For now, here's an example of the
# much faster version

r = b.sum()
print('fast sum', r)

# Vectorized multiplication (and broadcasting):
a = np.arange(20).reshape(5, 4)
b = np.arange(20).reshape(5, 4)

c = a * b # NOTE: this does element-wise multiplication, not the matrix multiplication you learned in 7-th? grade.
# print('c\n', c)

# Numpy will smartly "broadcast" two matrices of different sizes or dimensions so that it works:

d = np.arange(4)
e = a * d # WAIT! Aren't you multiplying two different matrices with different sizes? Yes!
# print('e\n', e)

# Also means you can use the same notation to multiply an array (2dim) against a scalar (0dim):
f = a * 6.0
# print('f\n', f)

# Plotting with matplotlib.

# NOTE: If you have done a plt.plot() above but not called plt.show(), this next line will add the graphics on top of the
# previously drawn graphics. Whey you call plt.show(), it shows it, then clears it.
ax = plt.imshow(b)
# plt.show()

# Perhaps you want a colorbar?
ax = plt.imshow(b)
# plt.colorbar(ax)
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


plt.imshow(mandelbrot(400, 400))
# plt.show()

## CLASS ACTIVITY:

# Create a 20 by 40 matrix of random values 0-1, then set the upper left quadrant to 1. Next, set the last COLUMN to 2.
# Finally, change all values less than .5 to be 3.
# You will likely want to use the matrix slicing notation discussed above to set values. i.e.,


a = np.random.random((40, 20))
print(a)



