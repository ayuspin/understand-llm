import numpy as np

# Hidden state from the model (a row of 4 features)
hidden = np.array([0.5, 1.2, 0.0, 2.1])

# Output weights: 4 rows, 2 columns (for 2 words)
# Each COLUMN is a word's weight vector
output_weights = np.array([
    [0.1, 0.9],   # Weights for feature 1
    [0.8, -0.1],  # Weights for feature 2
    [-0.5, 0.2],  # Weights for feature 3
    [0.4, 0.7]    # Weights for feature 4
])

# Forward pass: hidden @ weights
# Result: one score per word (column)
scores = hidden @ output_weights

print(f"Hidden state: {hidden}")
print(f"\nOutput weights shape: {output_weights.shape}")
print(f"(4 features → 2 words)")
print(f"\nWord scores:")
print(f"  'the': {scores[0]:.2f}")
print(f"  'cat': {scores[1]:.2f}")
print(f"\nPrediction: '{'the' if scores[0] > scores[1] else 'cat'}'")
