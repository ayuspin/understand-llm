import numpy as np

# 1. UTILS: Softmax and ReLU
def softmax(x):
    # Stabilized Softmax for numeric stability
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

# 2. POSITIONAL ENCODING: RoPE (Rotary Positional Embedding)
def apply_rope(x):
    # Demonstrating the 'Clock Hand' rotation for d=16 features
    seq_len, d_model = x.shape
    x_out = np.zeros_like(x)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Calculate rotation angle based on position and feature index
            theta = 10000**(-i / d_model)
            m_theta = pos * theta
            cos, sin = np.cos(m_theta), np.sin(m_theta)
            
            # Rotate [x0, x1] feature pair
            x0, x1 = x[pos, i], x[pos, i+1]
            x_out[pos, i] = x0 * cos - x1 * sin
            x_out[pos, i+1] = x0 * sin + x1 * cos
    return x_out

# 3. CAUSAL MULTI-HEAD ATTENTION
def masked_attention_head(x, W_Q, W_K, W_V, d_head):
    # Projection: Extract specialized features for this head
    Q = x @ W_Q # (seq_len, d_head)
    K = x @ W_K
    V = x @ W_V
    
    # Scaled Dot-Product
    scores = (Q @ K.T) / np.sqrt(d_head) # (seq_len, seq_len)
    
    # CAUSAL MASKING: The "Blindfold" to prevent cheating during training
    seq_len = x.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    scores = scores - (mask * 1e9) # Set future tokens to near-negative infinity
    
    weights = softmax(scores)
    return weights @ V

# 4. DATA SETUP (Tiny Dataset)
text = "the cat sat on the mat."
words = text.replace(".", "").split()
vocab = sorted(list(set(words)))
word_to_id = {word: i for i, word in enumerate(vocab)}
id_to_word = {i: word for word, i in word_to_id.items()}
vocab_size = len(vocab)
d_model = 16 # Dimensionality of the model

# 5. INITIALIZATION (Random Parameters)
np.random.seed(42)
embeddings = np.random.randn(vocab_size, d_model) * 0.1

# Attention Weights (2 Heads, each head gets 8 features)
W_Q1, W_K1, W_V1 = [np.random.randn(d_model, 8) * 0.1 for _ in range(3)]
W_Q2, W_K2, W_V2 = [np.random.randn(d_model, 8) * 0.1 for _ in range(3)]
W_out_attn = np.random.randn(d_model, d_model) * 0.1 # Projection back to d_model

# MLP Weights (Up-projects to 64 features then down-projects)
W_up = np.random.randn(d_model, 64) * 0.1
W_down = np.random.randn(64, d_model) * 0.1

# Final Prediction Head
W_final = np.random.randn(d_model, vocab_size) * 0.1

# 6. TRANSFORMER BLOCK (The Complete Forward Pass)
def transformer_forward(input_ids):
    # Step A: Embeddings + Position
    x = embeddings[input_ids] 
    x = apply_rope(x)
    
    # Step B: Multi-Head Attention + Residual Connection
    h1 = masked_attention_head(x, W_Q1, W_K1, W_V1, 8)
    h2 = masked_attention_head(x, W_Q2, W_K2, W_V2, 8)
    # Concat heads and project back to original size
    attn_out = np.concatenate([h1, h2], axis=1) @ W_out_attn
    x = x + attn_out # RESIDUAL HIGHWAY 1
    
    # Step C: MLP (Thinking) + Residual Connection
    mlp_out = relu(x @ W_up) @ W_down
    x = x + mlp_out # RESIDUAL HIGHWAY 2
    
    # Step D: Logits (Predictions for EVERY word in the sequence)
    logits = x @ W_final
    return logits

# 7. EXECUTION (Parallel Training Mode)
input_text = "the cat sat on the"
input_tokens = [word_to_id[w] for w in input_text.split()]
print(f"Input: {input_text}")

logits = transformer_forward(input_tokens)

print("\n--- NEXT WORD PREDICTIONS (Parallel Pass) ---")
for i, token_id in enumerate(input_tokens):
    pred_id = np.argmax(logits[i])
    print(f"Word '{id_to_word[token_id]:4}': Model predicts next is '{id_to_word[pred_id]}'")

print("\n[Architecture Status] Synchronized with Roadmap Phase 5:")
print("- RoPE: Enabled")
print("- Multi-Head Attention: Enabled (2 Heads)")
print("- Causal Masking: Enabled (No cheating)")
print("- Residual Connections: Enabled")
print("- MLP: Enabled (Up/Down Projection)")
