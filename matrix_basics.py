"""
MATRIX BASICS for AI: The "Row-by-Column" Rule
---------------------------------------------
This script explains Matrix Multiplication as a physical process of 
multiplying Horizontal Weights (Rows) by Vertical Data (Columns).
"""

# ---------------------------------------------------------
# STEP 1: THE DOT PRODUCT (Row meets Column)
# ---------------------------------------------------------
# Think of a Row vector (Weights) and a Column vector (Input).
# Mathematically, we multiply them element-by-element and sum the result.

row_w = [0.1, 0.8]  # Horizontal Weight Row [---]
col_x = [10, 20]    # Vertical Input Column  [ | ]

# Visualization
print("--- STEP 1: ORIENTATION ---")
print("  ROW (Weights)    COLUMN (Input)")
print(f"  [{row_w[0]}, {row_w[1]}]   x   [ {col_x[0]} ]")
print(f"               *       [ {col_x[1]} ]")

# The Math: (Weight1 * Input1) + (Weight2 * Input2)
dot_product = (row_w[0] * col_x[0]) + (row_w[1] * col_x[1])

print(f"\nCalculation: ({row_w[0]} * {col_x[0]}) + ({row_w[1]} * {col_x[1]}) = {dot_product}")
print("-" * 45 + "\n")


# ---------------------------------------------------------
# STEP 2 & 3: MATRIX-VECTOR MULTIPLICATION
# ---------------------------------------------------------
# A matrix is a stack of Rows.
# Multiplication is just repeating the dot-product for every row.

matrix_w = [
    [1, 0],  # Row 1 (Detects Feature A)
    [0, 1],  # Row 2 (Detects Feature B)
    [1, 1]   # Row 3 (Detects Both)
]

print("--- STEP 2 & 3: STACKING THE ROWS ---")
print("MATRIX (3 Rows)          COLUMN (1 Input)")
print(f"Row 1: {matrix_w[0]}       x     [ {col_x[0]} ]")
print(f"Row 2: {matrix_w[1]}             [ {col_x[1]} ]")
print(f"Row 3: {matrix_w[2]}")

print("\nProcessing results row-by-column:")
results = []
for i, row in enumerate(matrix_w):
    # Standard Math Order: Weight * Input
    score = (row[0] * col_x[0]) + (row[1] * col_x[1])
    results.append(score)
    print(f" Row {i+1} Output: ({row[0]} * {col_x[0]}) + ({row[1]} * {col_x[1]}) = {score}")

print(f"\nFinal Result (a new vector): {results}")
print("-" * 45 + "\n")


# ---------------------------------------------------------
# STEP 5: EXAMPLE - THE "LOGITS" (Final Word Scores)
# ---------------------------------------------------------
# In an LLM, the output is a matrix where every ROW is the
# weight vector for a specific word.

hidden_output = [0.5, 1.2, 0.0, 2.1]  # The "Column" input to this layer
output_matrix = [
    [0.1, 0.8, -0.5, 0.4],  # Row Weights for "the"
    [0.9, -0.1, 0.2, 0.7],  # Row Weights for "cat"
]

print("--- STEP 5: REALISTIC FORWARD PASS ---")
print("   OUTPUT MATRIX (Rows)          HIDDEN STATE (Column)")
print(f"Row 'the': {output_matrix[0]}     x     [ {hidden_output[0]} ]")
print(f"Row 'cat': {output_matrix[1]}           [ {hidden_output[1]} ]")
print(f"                                   [ {hidden_output[2]} ]")
print(f"                                   [ {hidden_output[3]} ]")

for i, word in enumerate(["the", "cat"]):
    row = output_matrix[i]
    # Detailed multiplication string
    steps = " + ".join([f"({w} * {x})" for w, x in zip(row, hidden_output)])
    score = sum(w * x for w, x in zip(row, hidden_output))
    print(f"\n Score for '{word}':\n {steps} = {score:.2f}")

print("-" * 45 + "\n")


# ---------------------------------------------------------
# STEP 4: NUMPY ORIENTATION
# ---------------------------------------------------------
import numpy as np

w_np = np.array(matrix_w)    # 3x2 Matrix
x_np = np.array(col_x)       # Vector (NumPy treats as column for @)

# The '@' operator is effectively: For every Row in W, Dot Product with X.
numpy_result = w_np @ x_np

print("--- STEP 4: NUMPY (The Professional Standard) ---")
print(f"Input Matrix Shape: {w_np.shape} (3 rows, 2 columns)")
print(f"Input Vector Shape: {x_np.shape} (size 2)")
print(f"Numpy result of W @ X: {numpy_result}")
print("\nNumPy handles the 'bending' of row-to-column automatically!")
