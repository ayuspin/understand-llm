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
embeddings = np.random.randn(vocab_size, vector_size) 

# HIDDEN LAYER 1
h1_size = 8
w_h1 = np.random.randn(vector_size, h1_size)

# HIDDEN LAYER 2
h2_size = 8
w_h2 = np.random.randn(h1_size, h2_size)

# OUTPUT LAYER
w_out = np.random.randn(h2_size, vocab_size)

def relu(x):
    return np.maximum(0, x)

# 3. FORWARD PASS (Inference)
input_word = "the"
input_id = word_to_id[input_word]

# Step A: EMBEDDING LOOKUP
vector = embeddings[input_id]  

# Step B: LAYER 1 + ACTIVATION
h1_output = relu(np.dot(vector, w_h1))

# Step C: LAYER 2 + ACTIVATION
h2_output = relu(np.dot(h1_output, w_h2))

# Step D: OUTPUT LAYER (Logits)
logits = np.dot(h2_output, w_out)
print(f"Last hidden layer output: {h2_output}")
print(f"Output layer weights: {w_out}")
print(f"Logits from forward pass: {logits}")
predicted_id = np.argmax(logits)

print(f"\nInput: '{input_word}'")
print(f"Predicted next word: '{id_to_word[predicted_id]}'")

# 4. TRAINING / BACKPROPAGATION (Manual Gradient Descent)
target_word = "cat"
target_id = word_to_id[target_word]
print(f"Actual next word: '{target_word}'")

learning_rate = 0.1
if predicted_id != target_id:
    # We nudge every layer that contributed to the final result.
    # Output Layer Update
    w_out[:, target_id] += learning_rate * h2_output
    
    # Layer 2 Update (Nudge based on correctly identified 'cat' indicators)
    w_h2 += learning_rate * np.outer(h1_output, (w_out[:, target_id] > 0))
    
    # Layer 1 Update
    w_h1 += learning_rate * np.outer(vector, (np.dot(w_h2, (w_out[:, target_id] > 0)) > 0))
    
    # Embedding Update
    embeddings[input_id] += learning_rate * np.dot(w_h1, np.dot(w_h2, w_out[:, target_id]))
    
    print("\n[Optimizer Update] Adjusted Embeddings and ALL 3 Weight Matrices.")

# 5. VALIDATION
new_vector = embeddings[input_id]
new_h1 = relu(np.dot(new_vector, w_h1))
new_h2 = relu(np.dot(new_h1, w_h2))
new_logits = np.dot(new_h2, w_out)
new_predicted_id = np.argmax(new_logits)
print(f"New prediction after update: '{id_to_word[new_predicted_id]}'")
