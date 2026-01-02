import numpy as np

# 1. SETUP: A single word vector (d=4)
# Let's imagine these are features that have become "unstable"
# Some are very large, some are very small.
word_vector = np.array([10.5, -2.2, 50.1, 0.1])

print("--- ORIGINAL UNSTABLE VECTOR ---")
print(word_vector)
print(f"Mean: {np.mean(word_vector):.2f} | Variance: {np.var(word_vector):.2f}")

# 2. LAYER NORMALIZATION: The Math
# Formula: 
#   Mean (μ) = sum(x) / n
#   Variance (σ²) = sum((x - μ)²) / n
def layer_norm(x, eps=1e-5):
    # Mean: The "Average" center point of the vector's values.
    # Variance: The "Spread" — how far the values stray from the average.
    
    # eps is a tiny number to prevent division by zero
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    
    # The normalization step
    # (x - mean) / sqrt(variance + eps)
    x_normalized = (x - mean) / np.sqrt(variance + eps)
    
    return x_normalized

# 3. APPLYING NORMALIZATION
normalized_vector = layer_norm(word_vector)

print("\n--- NORMALIZED VECTOR (Mean=0, Var=1) ---")
print(normalized_vector.round(2))
print(f"Mean: {np.mean(normalized_vector):.2f} | Variance: {np.var(normalized_vector):.2f}")

# 4. LEARNED PARAMETERS (Gamma and Beta)
# In a real model, there are two learned numbers per feature
gamma = np.array([1.0, 1.0, 2.0, 1.0]) # Scaling
beta = np.array([0.0, 0.5, 0.0, 0.0])  # Shifting

final_output = (gamma * normalized_vector) + beta

print("\n--- FINAL LAYER NORM OUTPUT (With learned adjustments) ---")
print(final_output.round(2))

print("\nCONCLUSION: LayerNorm prevents 'exploding' values from breaking the math.")
print("It keeps all word features in a healthy range (re-centered around zero).")
