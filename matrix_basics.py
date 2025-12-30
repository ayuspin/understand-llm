"""
MATRIX BASICS for AI
--------------------
This script explains Matrix Multiplication using only simple math (multiplication and addition).
No 'Matrix Math' knowledge is required to read this.
"""

# ---------------------------------------------------------
# STEP 1: THE "DOT PRODUCT" (The Core Building Block)
# ---------------------------------------------------------
# Imagine you have two lists of numbers. 
# A "Dot Product" is just a fancy name for:
# "Multiply the partners and add the results together."

list_a = [10, 20]
list_b = [1, 2]

# The Math: (10 * 1) + (20 * 2) = 10 + 40 = 50
dot_product = (list_a[0] * list_b[0]) + (list_a[1] * list_b[1])

print("--- STEP 1: DOT PRODUCT ---")
print(f"List A: {list_a}")
print(f"List B: {list_b}")
print(f"Calculation: ({list_a[0]} * {list_b[0]}) + ({list_a[1]} * {list_b[1]}) = {dot_product}")
print("-" * 30 + "\n")


# ---------------------------------------------------------
# STEP 2: WHAT IS A MATRIX?
# ---------------------------------------------------------
# A matrix is just a "List of Lists".
# It's a way to store multiple "Templates" in one variable.

# This matrix has 3 templates, each with 2 numbers.
matrix_w = [
    [1, 0],  # Template 1
    [0, 1],  # Template 2
    [1, 1]   # Template 3
]

# ---------------------------------------------------------
# STEP 3: MATRIX-VECTOR MULTIPLICATION
# ---------------------------------------------------------
# This is how the AI "Compares" an input to its templates.
# "Matrix Multiplication" is just doing a Dot Product for EVERY template in the list.

input_v = [10, 20] # Our input data

print("--- STEP 2 & 3: MATRIX MULTIPLICATION ---")
print(f"Input Vector: {input_v}")
print(f"Matrix (3 Templates):\n {matrix_w[0]}\n {matrix_w[1]}\n {matrix_w[2]}")
print("\nCalculating results for each template:")

results = []
for i in range(len(matrix_w)):
    template = matrix_w[i]
    
    # Do the Dot Product for this template
    score = (input_v[0] * template[0]) + (input_v[1] * template[1])
    results.append(score)
    
    print(f" Template {i+1}: ({input_v[0]} * {template[0]}) + ({input_v[1]} * {template[1]}) = {score}")

print(f"\nFinal Result Vector: {results}")
print("-" * 30 + "\n")


# ---------------------------------------------------------
# STEP 4: WHY DO WE USE NUMPY?
# ---------------------------------------------------------
# Doing it with loops and manual indices (like above) is slow and messy.
# Professionals use a library called 'NumPy' to do it in one line.

import numpy as np

# Convert our lists to 'NumPy Arrays'
w_np = np.array(matrix_w)
v_np = np.array(input_v)

# The '@' symbol in Python means "Matrix Multiply"
numpy_result = w_np @ v_np

print("--- STEP 4: THE PROFESSIONAL WAY (NumPy) ---")
print(f"Result using '@' operator: {numpy_result}")
print("It's identical to our manual loop, just much faster!")
