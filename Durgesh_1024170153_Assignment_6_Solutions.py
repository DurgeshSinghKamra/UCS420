import numpy as np

# Q1. Sensor readings: temperature = [25, 28, 31, 35, 38, 27, 33, 40].
# The sensor has an average measurement error of +2 C.
# (a) Add 2 C to every reading using vectorization.
# (b) Convert the corrected readings to Fahrenheit.
# (c) Identify readings greater than 32 C using Boolean indexing.
# (d) Count readings greater than 32 C.
# (e) Explain vectorization and Boolean indexing.

temperature = np.array([25, 28, 31, 35, 38, 27, 33, 40])
corrected_temperature = temperature + 2
fahrenheit = (9 / 5) * corrected_temperature + 32
above_32 = corrected_temperature[corrected_temperature > 32]

print("Q1. Original temperatures:", temperature)
print("Q1(a). Corrected temperatures (+2 C):", corrected_temperature)
print("Q1(b). Corrected temperatures in Fahrenheit:", fahrenheit)
print("Q1(c). Corrected readings greater than 32 C:", above_32)
print("Q1(d). Number of corrected readings greater than 32 C:",
      np.count_nonzero(corrected_temperature > 32))
print("Q1(e). Vectorization applies an operation to an entire array without an "
      "explicit Python loop. Boolean indexing selects values that satisfy a "
      "condition. Both make array operations concise and often faster.")


# Q2. Daily steps recorded by four users over three days.
# (a) Total steps, (b) mean, (c) max/min, (d) totals per day using axis=0,
# (e) totals per user using axis=1, and (f) position of maximum using argmax().

steps = np.array([
    [5000, 6200, 7100],
    [8000, 7500, 9000],
    [4500, 5100, 4800],
    [9000, 8500, 9500]
])

print("\nQ2(a). Total steps:", np.sum(steps))
print("Q2(b). Mean steps:", np.mean(steps))
print("Q2(c). Maximum steps:", np.max(steps))
print("Q2(c). Minimum steps:", np.min(steps))
print("Q2(d). Total steps for each day (axis=0):", np.sum(steps, axis=0))
print("Q2(e). Total steps for each user (axis=1):", np.sum(steps, axis=1))
max_flat_index = np.argmax(steps)
max_position = np.unravel_index(max_flat_index, steps.shape)
print("Q2(f). Maximum value's (user index, day index):", max_position)
print("Maximum value:", steps[max_position])


# Q3(a). Create original = [1, 2, 3, 4, 5, 6].
# Q3(b). Slice indices 1 to 4 into subset.
# Q3(c). Modify subset's first element and display original and subset.
# Q3(d). Create a copied slice, modify it, and display both arrays.

original = np.array([1, 2, 3, 4, 5, 6])
subset = original[1:5]
print("\nQ3(a). Original:", original)
print("Q3(b). Slice subset:", subset)
subset[0] = 999
print("Q3(c). After modifying subset (a view):")
print("Original:", original)
print("Subset:", subset)

copied_subset = original[1:5].copy()
copied_subset[0] = 500
print("Q3(d). After modifying copied slice:")
print("Original:", original)
print("Copied slice:", copied_subset)


# Q3(e). Create numbers 1 through 12 with arange() and reshape to 3x4.
# Q3(f). Extract first row, last row, second column, and rows 1-2/columns 2-3.
# Q3(g). Create flattened arrays using flatten() and ravel().
# Q3(h). Change ravel() output and check the original.
# Q3(i). Change flatten() output and check the original.
# Q3(j). Display shape, ndim, size, and dtype.

matrix = np.arange(1, 13).reshape(3, 4)
print("\nQ3(e). 3x4 matrix:\n", matrix)
print("Q3(f). First row:", matrix[0, :])
print("Q3(f). Last row:", matrix[-1, :])
print("Q3(f). Second column:", matrix[:, 1])
print("Q3(f). Rows 1-2 and columns 2-3 (zero-based indices):\n",
      matrix[1:3, 2:4])

ravel_array = matrix.ravel()
flatten_array = matrix.flatten()
print("Q3(g). ravel():", ravel_array)
print("Q3(g). flatten():", flatten_array)

ravel_array[0] = 1000
print("Q3(h). After changing ravel()[0], matrix is affected:")
print(matrix)

flatten_array[1] = 2000
print("Q3(i). After changing flatten()[1], matrix is not affected:")
print(matrix)
print("Flattened copy:", flatten_array)

print("Q3(j). Shape:", matrix.shape)
print("Q3(j). Dimensions:", matrix.ndim)
print("Q3(j). Number of elements:", matrix.size)
print("Q3(j). Data type:", matrix.dtype)


# Q4. Use the given features X and assistance score y.
# (a) Display shape and dimensions.
# (b) Calculate transpose and explain it.
# (c) Calculate X.T @ X.
# (d) Calculate inverse where possible.
# (e) Calculate beta = (X.T X)^-1 X.T y.
# (f) Explain the coefficients.
# (g) Predict the assistance score for [5, 40, 7].

X = np.array([
    [6, 70, 3],
    [5, 50, 6],
    [8, 80, 2],
    [4, 30, 8]
], dtype=float)

y = np.array([40, 65, 30, 85], dtype=float)

print("\nQ4(a). X shape:", X.shape)
print("Q4(a). X dimensions:", X.ndim)
print("Q4(b). X transpose (features become rows):\n", X.T)
xtx = X.T @ X
print("Q4(c). X.T @ X:\n", xtx)

try:
    xtx_inverse = np.linalg.inv(xtx)
    print("Q4(d). Inverse of X.T @ X:\n", xtx_inverse)
    beta = xtx_inverse @ X.T @ y
    print("Q4(e). OLS coefficients beta:", beta)
except np.linalg.LinAlgError:
    # lstsq provides a least-squares solution if X.T @ X is singular.
    print("X.T @ X is singular, so using np.linalg.lstsq instead.")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    print("Q4(e). Least-squares coefficients beta:", beta)

print("Q4(f). The coefficients correspond respectively to sleep hours, "
      "activity level, and stress level. Each coefficient indicates the "
      "model's predicted change in assistance score for a one-unit increase "
      "in that feature, holding the other features fixed. The model has no "
      "intercept because the assignment's equation does not include one.")

new_user = np.array([5, 40, 7], dtype=float)
predicted_score = new_user @ beta
print("Q4(g). New user's features:", new_user)
print("Predicted assistance score:", predicted_score)
