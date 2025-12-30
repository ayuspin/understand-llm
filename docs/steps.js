// Tutorial Steps Configuration
const STEPS = [
    {
        title: "Step 1: The Dot Product",
        explanation: `
            <p>The <strong>Dot Product</strong> is the core building block of all neural network math.</p>
            <p>It's surprisingly simple: <strong>multiply pairs of numbers together, then add up the results</strong>.</p>
            <p>In this example, we have two lists:</p>
            <p><code>list_a = [10, 20]</code> (our input data)</p>
            <p><code>list_b = [1, 2]</code> (our weights)</p>
            <p>The calculation is: <code>(10 × 1) + (20 × 2) = 10 + 40 = 50</code></p>
            <p>This single number tells us how much the input "aligns" with the weights. Higher alignment = higher score.</p>
            <p><strong>Click "Run" to see it in action! →</strong></p>
        `,
        code: `import numpy as np

# Two lists of numbers
list_a = np.array([10, 20])  # Input data
list_b = np.array([1, 2])    # Weights

# Dot Product: multiply pairs, sum the results
result = np.dot(list_a, list_b)

print(f"List A: {list_a}")
print(f"List B: {list_b}")
print(f"Dot Product: (10 × 1) + (20 × 2) = {result}")`
    },
    {
        title: "Step 2: Matrix Multiplication",
        explanation: `
            <p>A <strong>Matrix</strong> is just a stack of weight vectors.</p>
            <p>When we multiply a matrix by a vector, we're doing a <strong>dot product for every row</strong>.</p>
            <p>In AI terminology:</p>
            <ul style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li>Each <strong>row</strong> in the matrix is a "detector" for a specific feature.</li>
                <li>The result is a new vector of scores—one for each detector.</li>
            </ul>
            <p>This is how neural networks "compare" inputs against learned patterns.</p>
        `,
        code: `import numpy as np

# Our input vector
input_vec = np.array([10, 20])

# Weight matrix with 3 rows (3 "detectors")
weights = np.array([
    [1, 0],   # Detector 1: focuses on first value
    [0, 1],   # Detector 2: focuses on second value
    [1, 1]    # Detector 3: looks at both
])

# Matrix multiplication = dot product for each row
result = weights @ input_vec

print(f"Input: {input_vec}")
print(f"\\nWeight Matrix:")
print(weights)
print(f"\\nResults:")
print(f"  Detector 1: {result[0]}")
print(f"  Detector 2: {result[1]}")
print(f"  Detector 3: {result[2]}")`
    },
    {
        title: "Step 3: The AI Orientation (x × W)",
        explanation: `
            <p>In real AI code, we flip the order: <strong>Input × Weights</strong> instead of Weights × Input.</p>
            <p>This means:</p>
            <ul style="margin: 16px 0; padding-left: 20px; color: var(--text-secondary);">
                <li>Input is a <strong>horizontal row</strong></li>
                <li>Each <strong>column</strong> in the weight matrix is a word's weights</li>
            </ul>
            <p>This is exactly how language models work. They take a hidden state (row) and multiply it against an output matrix where each column represents a different word in the vocabulary.</p>
            <p>The highest score wins—that's the predicted word!</p>
        `,
        code: `import numpy as np

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
print(f"\\nOutput weights shape: {output_weights.shape}")
print(f"(4 features → 2 words)")
print(f"\\nWord scores:")
print(f"  'the': {scores[0]:.2f}")
print(f"  'cat': {scores[1]:.2f}")
print(f"\\nPrediction: '{'the' if scores[0] > scores[1] else 'cat'}'")`
    }
];
