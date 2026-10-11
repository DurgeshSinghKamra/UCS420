# UCS420: Cognitive Computing
# Assignment 4 - NumPy Introduction-I
# Write each question as a comment, followed by its solution.

import numpy as np


# Q1. Create a NumPy 1D array with 5 elements and perform:
# (a) Addition of 2 to every element
# (b) Multiply every element by 3
# (c) Divide every element by 2

arr1 = np.array([10, 20, 30, 40, 50])
print("Q1. Original array:", arr1)
print("Q1(a). Add 2:", arr1 + 2)
print("Q1(b). Multiply by 3:", arr1 * 3)
print("Q1(c). Divide by 2:", arr1 / 2)


# Q2(a). Reverse the NumPy array arr = [1, 2, 3, 6, 4, 5].

arr = np.array([1, 2, 3, 6, 4, 5])
print("\nQ2(a). Reversed array:", arr[::-1])


# Q2(b). Find the most frequent value and its index/indices in each array.

def print_modes_and_indices(values):
    unique_values, counts = np.unique(values, return_counts=True)
    max_count = counts.max()
    modes = unique_values[counts == max_count]

    print("Array:", values)
    print("Most frequent value(s):", modes)
    print("Frequency:", max_count)
    for mode in modes:
        print(f"Indices of {mode}:", np.where(values == mode)[0])


x = np.array([1, 2, 3, 4, 5, 1, 2, 1, 1, 1])
print("\nQ2(b)(i). For x:")
print_modes_and_indices(x)

y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
print("\nQ2(b)(ii). For y:")
print_modes_and_indices(y)


# Q3. For the given 2D array, access:
# (a) First row, second column
# (b) Third row, first column

arr2 = np.array([
    [10, 20, 30],
    [40, 50, 60],
    [70, 80, 90]
])
print("\nQ3(a). First row, second column:", arr2[0, 1])
print("Q3(b). Third row, first column:", arr2[2, 0])


# Q4. Create a 1D array with 25 evenly spaced values from 10 to 100 using
# linspace(). Print dimensions, shape, number of elements, dtype, and bytes.
# Find the transpose using reshape(). Check whether T also works.

name = input("\nEnter your name for the array variable label: ").strip()
safe_name = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
if not safe_name:
    safe_name = "student"

ucs420_name = np.linspace(10, 100, 25)
print("\nQ4. ucs420_" + safe_name + ":", ucs420_name)
print("Dimensions (ndim):", ucs420_name.ndim)
print("Shape:", ucs420_name.shape)
print("Total elements:", ucs420_name.size)
print("Data type:", ucs420_name.dtype)
print("Bytes consumed:", ucs420_name.nbytes)
print("Transpose using reshape(1, 25):\n", ucs420_name.reshape(1, 25))
print("Transpose using T attribute:", ucs420_name.T)
print("For a 1D array, .T does not change its shape; reshape(1, 25) makes it a row.")


# Q5. Create a 2D array with three rows and four columns using the given values.
# Calculate mean, median, max, min and unique elements. Reshape to 4x3 and
# resize to 2x3.

ucs420_name = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 15, 20, 35]
])

print("\nQ5. Original 3x4 array:\n", ucs420_name)
print("Mean:", np.mean(ucs420_name))
print("Median:", np.median(ucs420_name))
print("Maximum:", np.max(ucs420_name))
print("Minimum:", np.min(ucs420_name))
print("Unique elements:", np.unique(ucs420_name))

reshaped_ucs420_name = ucs420_name.reshape(4, 3)
print("reshaped_ucs420_name (4x3):\n", reshaped_ucs420_name)

# np.resize returns a new array with the requested shape. Since 6 values are
# requested from a 12-element array, it keeps the first 6 values in row order.
resized_ucs420_name = np.resize(ucs420_name, (2, 3))
print("resized_ucs420_name (2x3):\n", resized_ucs420_name)
