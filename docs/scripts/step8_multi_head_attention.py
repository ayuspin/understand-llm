import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

# 1. SETUP: 3 words, each with 4 features
# "The", "cat", "sat"
input_matrix = np.array([
    [1.0, 0.0, 0.0, 0.0],  # "The"
    [0.0, 1.0, 0.0, 1.0],  # "cat" (idx 1=animal, idx 3=capable of action)
    [0.0, 1.0, 1.0, 0.0],  # "sat" (idx 1=noun, idx 2=action verb)
])

# 2. DEFINING THE HEADS
# Total dimension (d) = 4. Number of heads = 2.
# Each head will have a dimension of 2.
d_model = 4
d_head = 2

def run_attention_head(inputs, W_Q, W_K, W_V):
    # Every head projects the 4D input into a 2D space
    Q = inputs @ W_Q
    K = inputs @ W_K
    V = inputs @ W_V
    
    # Standard Scaled Dot-Product
    scores = (Q @ K.T) / np.sqrt(d_head)
    weights = softmax(scores)
    return weights @ V, weights

# 3. HEAD 1: The "Noun/Animal" Specialist
# W_Q and W_K are 4x2 matrices.
# We set them to 'pick out' the noun feature (index 1) from the 4D input.
W_Q1 = np.zeros((4, 2)); W_Q1[1, 0] = 1.0 
W_K1 = np.zeros((4, 2)); W_K1[1, 0] = 1.0
W_V1 = np.zeros((4, 2)); W_V1[0, 0] = 1.0; W_V1[1, 1] = 1.0 # Pass through first 2 features

head1_output, head1_weights = run_attention_head(input_matrix, W_Q1, W_K1, W_V1)

# 4. HEAD 2: The "Action/Verb" Specialist
# We set them to 'pick out' the action features (index 2 and 3).
W_Q2 = np.zeros((4, 2)); W_Q2[3, 1] = 1.0
W_K2 = np.zeros((4, 2)); W_K2[3, 1] = 1.0
W_V2 = np.zeros((4, 2)); W_V2[2, 0] = 1.0; W_V2[3, 1] = 1.0 # Pass through last 2 features

head2_output, head2_weights = run_attention_head(input_matrix, W_Q2, W_K2, W_V2)

# 5. CONCATENATION
# Glue the two 2D outputs back into one 4D vector
final_output = np.concatenate([head1_output, head2_output], axis=1)

print("--- MULTI-HEAD ATTENTION: Standard Projection ---")
print(f"Model Dim: {d_model} | Head Dim: {d_head} | Total Heads: 2")

print("\n--- HEAD 1 WEIGHTS (Looking for Noun Patterns) ---")
print(head1_weights.round(2))
print("Result: Focuses heavily on 'cat' and 'sat' because they share noun features.")

print("\n--- HEAD 2 WEIGHTS (Looking for Action Patterns) ---")
print(head2_weights.round(2))
print("Result: Focuses on 'cat' because 'cat' is the thing that can act (index 3).")

print("\n--- FINAL CONCATENATED VECTORS (3x4) ---")
print(final_output.round(2))
print("\nCONCLUSION: Each 'half' of the final vector was calculated by a different specialist.")
