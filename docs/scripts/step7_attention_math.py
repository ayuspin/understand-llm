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

d_k = input_matrix.shape[1]  # dimension (4)

# 2. WEIGHTS: Manually defined for demonstration
# W_query: Looking for noun features (index 1)
# W_key: Offering noun features (index 1)
# W_value: The actual 'meaning' data
W_Q = np.zeros((4, 4)); W_Q[1, 1] = 1.0
W_K = np.zeros((4, 4)); W_K[1, 1] = 1.0
W_V = np.eye(4) # Value just passes through for now

# 3. CALCULATE Q, K, V
# Every word gets its own Q, K, and V
Q = input_matrix @ W_Q
K = input_matrix @ W_K
V = input_matrix @ W_V

# 4. SCALED DOT-PRODUCT
# How much does every word Query match every word Key?
# (3x4) x (4x3) = (3x3) matrix of scores
scores = Q @ K.T

# Scaling: divide by sqrt(d_k) to keep numbers stable
scaled_scores = scores / np.sqrt(d_k)

# 5. ATTENTION WEIGHTS (The Softmax)
# Turns scores into percentages that sum to 100% per row
weights = softmax(scaled_scores)

# 6. BLENDING
# Combine the Values according to the weights
output = weights @ V

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
