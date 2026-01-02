import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# 1. SETUP: 3 words, each with 4 features
# "The", "cat", "sat"
# We'll use simple numbers so we can track the math
input_matrix = np.array([
    [1.0, 0.0, 0.0, 0.0],  # "The"
    [0.0, 1.0, 0.0, 1.0],  # "cat" (features for noun/animal)
    [0.0, 1.0, 1.0, 0.0],  # "sat" (features for noun/action)
])

# d_k stands for "dimension of the keys".
# We use this to scale the dot product results.
key_dimension = input_matrix.shape[1]  # dimension (4)

# query_weights: How we transform a word into a "Search Query"
# key_weights: How we transform a word into a "Search Key" (Label)
query_weights = np.zeros((4, 4)); query_weights[1, 1] = 1.0
key_weights = np.zeros((4, 4)); key_weights[1, 1] = 1.0

# value_weights: How we transform a word into its "Value" (Content).
# We use np.eye(4) which is an "Identity Matrix" (the matrix version of the number 1).
# This acts as a placeholder that passes the original word data through without 
# changing its meaning, so we can see the results of the attention blend clearly.
value_weights = np.eye(4) 

# 3. CALCULATE Queries, Keys, Values
# Every word gets its own Query, Key, and Value vector
queries = input_matrix @ query_weights
keys = input_matrix @ key_weights
values = input_matrix @ value_weights

# 4. SCALED DOT-PRODUCT
# How much does every word Query match every word Key?
# (3x4) x (4x3) = (3x3) matrix of scores
scores = queries @ keys.T

# Scaling: divide by the square root of the key_dimension to keep numbers stable
scaled_scores = scores / np.sqrt(key_dimension)

# 5. ATTENTION WEIGHTS (The Softmax)
# Turns scores into percentages that sum to 100% per row
weights = softmax(scaled_scores)

# 6. BLENDING
# Combine the Values according to the weights
output = weights @ values

print("--- INPUT SENTENCE: 'The cat sat' ---")
print(f"Input Shape: {input_matrix.shape} (3 words, 4 features)")

print("\n--- ATTENTION WEIGHTS (Who looks at Whom) ---")
print("Rows = Query (Looking), Cols = Key (Offering)")
print(f"       The    cat    sat")
for i, row in enumerate(weights):
    label = ["The", "cat", "sat"][i]
    print(f"{label:4} {row}")

print("\n--- ANALYSIS: 'Sat' Word ---")
print(f"Word 'sat' is looking for 'cat' (high similarity in noun features).")
print(f"Result for 'sat' now contains {weights[2][1]*100:.1f}% of 'cat' data.")

print("\n--- FINAL CONTEXTUAL VECTORS ---")
print(output)
