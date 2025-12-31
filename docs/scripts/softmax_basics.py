import numpy as np

def softmax(logits):
    """
    Turns raw scores (logits) into a probability distribution.
    
    Recipe:
    1. e^x (Exponentiate)
    2. sum(e^x) (Sum)
    3. e^x / sum(e^x) (Normalize)
    """
    # 1. Exponentiate: raises e (2.718...) to the power of each score
    # This makes everything positive and amplifies the differences.
    exp_scores = np.exp(logits)
    
    # 2. Sum: find the total 'score budget'
    sum_exp_scores = np.sum(exp_scores)
    
    # 3. Normalize: divide each by the total
    probabilities = exp_scores / sum_exp_scores
    
    return probabilities, exp_scores, sum_exp_scores

# Example Logits (Raw scores from a model)
logits = np.array([5.2, -1.2, 0.4])
words = ["the", "cat", "dog"]

print("--- STEP 0: RAW LOGITS ---")
for i, word in enumerate(words):
    print(f"'{word}': {logits[i]}")

# Run Softmax
probs, exps, total = softmax(logits)

print("\n--- STEP 1: EXPONENTIATION (e^x) ---")
print("Everything is now positive. Notice how small gaps become huge!")
for i, word in enumerate(words):
    print(f"'{word}': e^{logits[i]:.1f} = {exps[i]:.4f}")

print(f"\nTOTAL SUM: {total:.4f}")

print("\n--- STEP 2: NORMALIZATION (The 'Probability Budget') ---")
print("Recipe: Divide each e^x by the TOTAL SUM.")
for i, word in enumerate(words):
    print(f"'{word}': {exps[i]:.4f} / {total:.4f} = {probs[i]:.4f} ({probs[i]*100:.2f}%)")

print(f"\nCHECK SUM: {np.sum(probs):.1f}")
