# Project Roadmap: From Math to Transformers

This roadmap outlines the journey from understanding basic matrix math to the complex architecture of modern Transformers like GPT-4.

## Phase 1: Foundations (Completed ✅)
- [x] **Vectorization**: Understanding that words are just lists of numbers.
- [x] **Linear Layers**: Matching inputs against "Weight Vectors" using Dot Products.
- [x] **Non-Linearity (ReLU)**: Why deep networks need mathematical "filters" to learn complex logic.
- [x] **Universal Function Approximation**: The theory that any complex logic can be represented as math.

## Phase 2: Sequential & Positional Math (Completed ✅)
- [x] **Softmax**: Turning raw scores into clean probabilities.
- [x] **Positional Encoding: Basic Concept**: "Stamping" time into word vectors via addition.
- [x] **RoPE Math (Rotary Positional Embedding)**: The modern standard. Rotating vector pairs like clock hands.

## Phase 3: The Attention Mechanism (Completed ✅) 🧠
- [x] **Self-Attention**: The "Big Breakthrough." How words in a sentence "pay attention" to each other.
- [x] **Scaled Dot-Product Attention**: The fundamental $Q, K, V$ math and the $\sqrt{d}$ scaling hack.
- [x] **Multi-Head Attention**: Parallel experts capturing different types of relationships simultaneously.
- [x] **Masking**: How the model is prevented from "looking at the future" during training.

## Phase 4: Training & Masking (Completed ✅) 🎭
- [x] **Causal Masking**: The "Blindfold" mechanism that prevents the model from looking at future tokens during training.
- [x] **Residual Connections**: The "Highways" that allow information and gradients to flow through deep networks without vanishing.

## Phase 5: Transformer Architecture (In Progress 🚀) 🏗️
- [x] **Transformer Block**: Integrated Attention, MLP, and Residuals into a single runnable pipeline in [text_to_model.py](file:///Users/yuspin/code/understand-llm/text_to_model.py).
- [x] **Layer Norm**: How models keep their internal activations stable and normalized.
- [x] **KV Cache**: The optimization that makes inference fast by caching projected Key and Value vectors.
- [ ] **The Training Loop**: Implementing Cross-Entropy Loss and Gradient Descent for the full block.
- [ ] **The Decoder Loop**: Moving from training (parallel) to inference (sequential auto-regression).

## Phase 6: Scaling and Generation
- [ ] **Temperature and Top-K**: Controlling how "creative" or "random" the model is.
- [ ] **BPE Tokenization**: How real models split words into subwords for better vocabulary coverage.
- [ ] **GPU Acceleration**: Moving from NumPy to hardware-optimized math.
- [ ] **Inference**: Scaling to billions of parameters.
