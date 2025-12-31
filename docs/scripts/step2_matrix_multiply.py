import numpy as np

# Our input vector
input_vec = np.array([10, 20])

# Weight matrix with 3 rows (3 "detectors")
weights = np.array([
    [1, 0],   # Detector 1: focuses on first value
    [0, 1],   # Detector 2: focuses on second value
    [1, 1]    # Detector 3: looks at both
])

# Matrix multiplication = dot product for each row
result = weights @ input_vec

print(f"Input: {input_vec}")
print(f"\nWeight Matrix:")
print(weights)
print(f"\nResults:")
print(f"  Detector 1: {result[0]}")
print(f"  Detector 2: {result[1]}")
print(f"  Detector 3: {result[2]}")
