import numpy as np

def create_toy_template(digit):
    """Creates a very simple 16x16 template for a digit."""
    template = np.zeros((16, 16))
    if digit == 0:
        template[4:12, 4] = 1.0  # Left
        template[4:12, 11] = 1.0 # Right
        template[4, 4:12] = 1.0  # Top
        template[11, 4:12] = 1.0 # Bottom
    elif digit == 1:
        template[4:12, 8] = 1.0  # Vertical line
    # (Simplified for demonstration)
    return template.flatten()

def recognize(image_vector, templates):
    """Compares image to templates using dot product (similarity)."""
    # The dot product is essentially: sum(image[i] * template[i])
    # It's high when the 'black' pixels align.
    scores = np.dot(templates, image_vector)
    return scores

# 1. Create templates for '0' and '1'
template_0 = create_toy_template(0)
template_1 = create_toy_template(1)
templates = np.stack([template_0, template_1]) # Shape: (2, 256)

# 2. Simulate an input 'image' (a noisy '1')
input_image = create_toy_template(1) + np.random.normal(0, 0.1, 256)

# 3. Recognize
scores = recognize(input_image, templates)

print(f"Similarity Score for '0': {scores[0]:.2f}")
print(f"Similarity Score for '1': {scores[1]:.2f}")

best_match = np.argmax(scores)
print(f"\nModel recognizes this as digit: {best_match}")
