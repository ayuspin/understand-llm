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
# It's a way to store multiple "Weight Vectors" in one variable.

# This matrix has 3 weight vectors, each with 2 numbers.
matrix_w = [
    [1, 0],  # Weight Vector 1
    [0, 1],  # Weight Vector 2
    [1, 1]   # Weight Vector 3
]

# ---------------------------------------------------------
# STEP 3: MATRIX-VECTOR MULTIPLICATION
# ---------------------------------------------------------
# This is how the AI "Compares" an input to its neurons.
# "Matrix Multiplication" is just doing a Dot Product for EVERY Weight Vector in the list.

input_v = [10, 20] # Our input data

print("--- STEP 2 & 3: MATRIX MULTIPLICATION ---")
print(f"Input Vector: {input_v}")
print(f"Matrix (3 Weight Vectors):\n {matrix_w[0]}\n {matrix_w[1]}\n {matrix_w[2]}")
print("\nCalculating results for each weight vector:")

results = []
for i in range(len(matrix_w)):
    w_vector = matrix_w[i]
    
    # Do the Dot Product for this weight vector
    score = (input_v[0] * w_vector[0]) + (input_v[1] * w_vector[1])
    results.append(score)
    
    print(f" Weight Vector {i+1}: ({input_v[0]} * {w_vector[0]}) + ({input_v[1]} * {w_vector[1]}) = {score}")

print(f"\nFinal Result Vector: {results}")
print("-" * 30 + "\n")


# ---------------------------------------------------------
# STEP 5: EXAMPLE - THE "LOGITS" CALCULATION
# ---------------------------------------------------------
# In an LLM, the very last step is multiplying the "Hidden Layer" 
# output by the "Output Weight Matrix" to get final scores for words.

hidden_output = [0.5, 1.2, 0.0, 2.1] # Features found by the model
output_weights = [
    [0.1, 0.8, -0.5, 0.4],  # Weight Vector for word "the"
    [0.9, -0.1, 0.2, 0.7],  # Weight Vector for word "cat"
]

print("--- STEP 5: THE FINAL FORWARD PASS (Logits) ---")
print(f"Hidden Layer Output: {hidden_output}")
print(f"Output Matrix (2 words):\n {output_weights[0]} ('the')\n {output_weights[1]} ('cat')")

for i, word in enumerate(["the", "cat"]):
    w_vector = output_weights[i]
    # Dot product: sum of (h[i] * w[i])
    score = sum(h * w for h, w in zip(hidden_output, w_vector))
    print(f" Score for '{word}': {score:.2f}")

print("\nThe word with the highest score is our prediction (Argmax).")
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

print(f"w_np: {w_np}")
print(f"v_np: {v_np}")

# The '@' symbol in Python means "Matrix Multiply"
numpy_result = w_np @ v_np

print("--- STEP 4: THE PROFESSIONAL WAY (NumPy) ---")
print(f"Example 1 (Matrix * Vector):\n {numpy_result}")

# Let's also do Step 5 (the Logits) with NumPy
hidden_np = np.array(hidden_output)
output_w_np = np.array(output_weights)
numpy_logits = output_w_np @ hidden_np

print(f"\nExample 2 (Logits from Step 5):\n {numpy_logits}")
print("It's identical to our manual loop, just much faster!")
