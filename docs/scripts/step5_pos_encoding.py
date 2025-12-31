import numpy as np

# 1. Word Embedding: What the word IS
# Let's say word size is 4
cat_vector = np.array([0.5, -0.2, 0.1, 0.8])

# 2. Position Vectors: WHERE the word is
# These are unique patterns for each index in the sentence.
# (For now, we use simple unique values for demonstration)
pos_pattern_0 = np.array([0.1, 0.1, 0.1, 0.1])
pos_pattern_1 = np.array([0.2, 0.2, 0.2, 0.2])

# 3. Mixing (Addition)
# This 'colors' the word vector with its position.
cat_at_start = cat_vector + pos_pattern_0
cat_at_index1 = cat_vector + pos_pattern_1

print("--- WORD: 'cat' ---")
print(f"Original Vector:  {cat_vector}")

print("\n--- POSITION 0 ---")
print(f"Position Pattern: {pos_pattern_0}")
print(f"Result (Word+Pos): {cat_at_start}")

print("\n--- POSITION 1 ---")
print(f"Position Pattern: {pos_pattern_1}")
print(f"Result (Word+Pos): {cat_at_index1}")

print("\n--- CONCLUSION ---")
print("The 'meaning' of cat is still there, but the numbers are now")
print("unique to the spot it occupies in the sentence.")
