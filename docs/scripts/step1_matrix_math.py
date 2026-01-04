import numpy as np

# ---------------------------------------------------------
# STEP 1: VECTOR & MATRIX MULTIPLICATION
# ---------------------------------------------------------
# Concept: A Vector (Row) multiplied by a Matrix (Columns).
# Operation: Matrix multiplication is performed by calculating the Dot Product 
# of the Row from the first matrix against each Column of the second.

# 1. THE DATA
# Input Vector: shape (1, 3)
x = np.array([2.0, 5.0, 1.0]) 

# Transformation Matrix: shape (3, 2)
W = np.array([
    [ 1.0,  -1.0],  # Row 1
    [ 0.5,   2.0],  # Row 2
    [ 0.0,   1.0]   # Row 3
])

print(f"--- 1. THE DATA ---")
print(f"Row Vector (1x3): \n{x}")
print(f"\nMatrix (3x2): \n{W}")

# 2. THE CALCULATION
# We calculate the dot product of the Row against each Column.
print(f"\n--- 2. CALCULATION STEPS ---")

results = []
# Loop through columns 0 and 1
for col_idx in range(W.shape[1]):
    # Extract the column
    col_vector = W[:, col_idx]
    
    # Calculate Dot Product
    math_steps = []
    total = 0
    for i in range(len(x)):
        val = x[i] * col_vector[i]
        math_steps.append(f"({x[i]} * {col_vector[i]})")
        total += val
        
    results.append(total)
    print(f"Result for Column {col_idx} (Row * Col {col_idx}):")
    print(f"  {' + '.join(math_steps)}")
    print(f"  = {total:.2f}")

# 3. NUMPY ORIENTATION
# In linear algebra, this is often written as xW
print(f"\n--- 3. NUMPY (x @ W) ---")
y = x @ W
print(f"Final Output Vector: {y}")
