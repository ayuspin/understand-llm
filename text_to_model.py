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

# --- NEW: THE HIDDEN LAYER ---
# This layer allows the model to combine features in complex ways.
# Shape: (Vector Size x Hidden Units)
hidden_size = 8
hidden_weights = np.random.randn(vector_size, hidden_size)

# Output Weights: Final prediction layer.
# Shape: (Hidden Units x Vocabulary Size)
output_weights = np.random.randn(hidden_size, vocab_size)

def relu(x):
    """ACTVATION FUNCTION: The 'Non-Linearity'. 
    Without this, 100 layers would mathematically collapse into 1 layer.
    """
    return np.maximum(0, x)

# 3. FORWARD PASS (Inference)
input_word = "the"
input_id = word_to_id[input_word]

# Step A: EMBEDDING LOOKUP
vector = embeddings[input_id]  

# Step B: HIDDEN LAYER + ACTIVATION
# Here, we transform the word vector into 'Hidden Features'
hidden_output = relu(np.dot(vector, hidden_weights))

# Step C: OUTPUT LAYER (Logits)
logits = np.dot(hidden_output, output_weights)

# Step D: ARGMAX
predicted_id = np.argmax(logits)

print(f"\nInput: '{input_word}'")
print(f"Predicted next word: '{id_to_word[predicted_id]}'")

# 4. TRAINING / BACKPROPAGATION (Simplistic Gradient Descent)
target_word = "cat"
target_id = word_to_id[target_word]
print(f"Actual next word: '{target_word}'")

learning_rate = 0.1
if predicted_id != target_id:
    # UPDATING ALL LAYERS (Nudging everything)
    # Output Layer Update
    output_weights[:, target_id] += learning_rate * hidden_output
    
    # Hidden Layer Update
    # (We nudge the weights that contributed to the correct output)
    hidden_weights += learning_rate * np.outer(vector, (output_weights[:, target_id] > 0))
    
    # Embedding Update
    embeddings[input_id] += learning_rate * np.dot(hidden_weights, output_weights[:, target_id])
    
    print("\n[Optimizer Update] Adjusted Embeddings, Hidden Layer, and Output Weights.")

# 5. VALIDATION
new_vector = embeddings[input_id]
new_hidden = relu(np.dot(new_vector, hidden_weights))
new_logits = np.dot(new_hidden, output_weights)
new_predicted_id = np.argmax(new_logits)
print(f"New prediction after update: '{id_to_word[new_predicted_id]}'")
