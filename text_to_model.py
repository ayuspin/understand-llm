import numpy as np

# 1. TOKENIZER (Vocabulary Mapping)
# This stage converts strings into integer 'Tokens' and vice-versa.
text = "the cat sat. the cat ate. the dog sat."
words = text.replace(".", "").split()
vocab = sorted(list(set(words)))
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}
vocab_size = len(vocab)
vector_size = 4  # The 'Embedding Dimension' (Hyperparameter)

print(f"Vocabulary: {vocab}")

# 2. MODEL INITIALIZATION (Parameters)
# Embeddings: A look-up table (Matrix) for word representations.
embeddings = np.random.randn(vocab_size, vector_size) 
# Linear Weights: The 'Templates' used to project features back into vocabulary space.
weights = np.random.randn(vector_size, vocab_size)

# 3. FORWARD PASS (Inference)
input_word = "the"
input_id = word_to_id[input_word]

# Step A: EMBEDDING LOOKUP (Fetching the vector for our token)
vector = embeddings[input_id]  

# Step B: LINEAR TRANSFORMATION (Calculating 'Logits' via Dot Product)
# 'Logits' are the raw scores before they are turned into probabilities.
logits = np.dot(vector, weights)

# Step C: ARGMAX (Choosing the most likely token)
predicted_id = np.argmax(logits)

print(f"\nInput: '{input_word}'")
print(f"Predicted next word: '{id_to_word[predicted_id]}'")

# 4. TRAINING / BACKPROPAGATION (Gradient Descent)
# The Ground Truth (The actual next word in the sequence)
target_word = "cat"
target_id = word_to_id[target_word]

print(f"Actual next word: '{target_word}'")

# LOSS CALCULATION 
# If prediction != target, we have high 'Loss'. 
# We update the parameters (Embeddings and Weights) using an 'Optimizer' logic.
learning_rate = 0.1
if predicted_id != target_id:
    # GRADIENT UPDATE (Shifting the numbers to reduce Loss)
    # 1. Adjust the Embedding Vector (Input-side tweak)
    embeddings[input_id] += learning_rate * weights[:, target_id]
    # 2. Adjust the Weights (Output-side tweak)
    weights[:, target_id] += learning_rate * vector
    print("\n[Optimizer Update] Adjusted Embeddings and Weights to reduce Loss.")

# 5. VALIDATION (Checking improvement)
new_vector = embeddings[input_id]
new_logits = np.dot(new_vector, weights)
new_predicted_id = np.argmax(new_logits)
print(f"New prediction after update: '{id_to_word[new_predicted_id]}'")
