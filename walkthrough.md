# Walkthrough - Understanding LLM Basics

We've covered the fundamental journey from simple digit recognition to the core mechanics of a Language Model.

## Accomplishments

### 1. Digit Recognition Basics
- Explored the intuition of "Template Matching" for hand-written digits.
- Maped the concept of "templates" to **Weight Matrices** and **Dot Products**.
- Created [digit_basics.py](file:///Users/yuspin/code/understand-llm/digit_basics.py) to demonstrate this similarity math.

### 2. Text to Model Conversion
- Demystified how text is "digitized" using a **Tokenizer**.
- Explained the role of **Embeddings** as a trainable map of word meanings.
- Provided a technical script, [text_to_model.py](file:///Users/yuspin/code/understand-llm/text_to_model.py), that implements a tiny "Forward Pass" and "Update" logic.
- Defined industry-standard terms: *Tokens, Logits, Forward Pass, Backpropagation, and Hyperparameters*.

### 3. Repository Setup
- Initialized a Git repository.
- Created a [.gitignore](file:///Users/yuspin/code/understand-llm/.gitignore) to keep the workspace clean.
- Committed the current state.

## Core Concepts Summary

| Term | Simple Meaning |
| :--- | :--- |
| **Tokenizer** | Mapping words to unique ID numbers. |
| **Embedding** | A list of numbers representing a word's "features". |
| **Dot Product** | Comparing two lists of numbers to see how well they "match". |
| **Argmax** | Picking the winning word with the highest score. |
| **Loss** | The mathematical "error" when the model guesses wrong. |
| **Optimizer** | The math that nudges the numbers to fix the error. |

---
Everything is now saved in your local git repository. What would you like to dive into next?
