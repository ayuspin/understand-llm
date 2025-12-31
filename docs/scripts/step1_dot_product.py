import numpy as np

# Two lists of numbers
list_a = np.array([10, 20])  # Input data
list_b = np.array([1, 2])    # Weights

# Dot Product: multiply pairs, sum the results
result = np.dot(list_a, list_b)

print(f"List A: {list_a}")
print(f"List B: {list_b}")
print(f"Dot Product: (10 × 1) + (20 × 2) = {result}")
