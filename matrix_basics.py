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
# STEP 5: REALISTIC FORWARD PASS (Input on the Left)
# ---------------------------------------------------------
# In standard math books, you see Matrix * Column.
# But in real AI code (like 'text_to_model.py'), we often do:
# [Row Input] * [Weight Matrix] = [Row Output]
#
# Here, each COLUMN in the matrix is the weight vector for a word!

hidden_output = [0.5, 1.2, 0.0, 2.1]  # Horizontal Row Data

# This matrix has 4 rows (to match the input) and 2 columns (for 2 words).
# Columns go DOWN:
# Col 1 (the): [0.1, 0.8, -0.5, 0.4]
# Col 2 (cat): [0.9, -0.1, 0.2, 0.7]
output_matrix = [
    [0.1, 0.9],   # Row 1 (Input 1 meets these two weights)
    [0.8, -0.1],  # Row 2 (Input 2 meets these two ...)
    [-0.5, 0.2],  # Row 3
    [0.4, 0.7]    # Row 4
]

print("--- STEP 5: MODERN AI ORIENTATION (x * W) ---")
print("   HIDDEN ROW (Data)         WEIGHT MATRIX (Columns are words)")
print(f"   {hidden_output}      x     | 'the' | 'cat' |")
print(f"                                |  {output_matrix[0][0]:.1f}  |  {output_matrix[0][1]:.1f}  | (multiplies 0.5)")
print(f"                                |  {output_matrix[1][0]:.1f}  |  {output_matrix[1][1]:.1f}  | (multiplies 1.2)")
print(f"                                | {output_matrix[2][0]:.1f}  |  {output_matrix[2][1]:.1f}  | (multiplies 0.0)")
print(f"                                |  {output_matrix[3][0]:.1f}  |  {output_matrix[3][1]:.1f}  | (multiplies 2.1)")

for i, word in enumerate(["the", "cat"]):
    # We take the i-th COLUMN of the matrix as the weight vector
    column_weights = [row[i] for row in output_matrix]
    
    # Detailed multiplication string
    steps = " + ".join([f"({x} * {w})" for x, w in zip(hidden_output, column_weights)])
    score = sum(x * w for x, w in zip(hidden_output, column_weights))
    print(f"\n Score for '{word}' (Input Row * Column {i+1}):\n {steps} = {score:.2f}")

print("-" * 45 + "\n")


# ---------------------------------------------------------
# STEP 4: NUMPY ORIENTATION
# ---------------------------------------------------------
import numpy as np

# Standard example: Matrix * Vector
w_np = np.array([[1, 0], [0, 1], [1, 1]])
x_np = np.array([10, 20])
numpy_standard = w_np @ x_np

# AI Forward Pass: Vector (Row) * Matrix
hidden_np = np.array(hidden_output) # 1x4 Row
out_w_np = np.array(output_matrix)   # 4x2 Matrix
numpy_logits = hidden_np @ out_w_np  # Result is 1x2 Row

print("--- STEP 4: NUMPY (Professional Orientation) ---")
print(f"Standard Row * Column result: {numpy_standard}")
print(f"Logits result (Row @ Matrix): {numpy_logits}")
print("\nNumPy is flexible: it can multiply [Matrix @ Column] OR [Row @ Matrix].")
print("Professional AI code almost ALWAYS uses [Row @ Matrix]!")
