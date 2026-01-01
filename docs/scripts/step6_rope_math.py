import numpy as np

def rotate_vector(v, angle_deg):
    """
    Standard 2D rotation math:
    x' = x*cos - y*sin
    y' = x*sin + y*cos
    """
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    
    x, y = v[0], v[1]
    return np.array([
        x * c - y * s,
        x * s + y * c
    ])

# 1. Our Word Vector (4 elements = 2 pairs)
cat_vec = np.array([1.0, 0.0, 1.0, 0.0])
pair1 = cat_vec[0:2]
pair2 = cat_vec[2:4]

# 2. Modern LLM logic: Different frequencies for different pairs
# Pair 1 rotates FAST (high freq)
# Pair 2 rotates SLOWLY (low freq)
freq1 = 45.0  # 45 degrees per step
freq2 = 10.0  # 10 degrees per step

def apply_rope(pos):
    # Calculate angles for this position
    angle1 = pos * freq1
    angle2 = pos * freq2
    
    # Rotate each pair
    new_pair1 = rotate_vector(pair1, angle1)
    new_pair2 = rotate_vector(pair2, angle2)
    
    # Put them back together
    return np.concatenate([new_pair1, new_pair2])

print(f"Original 'cat' vector: {cat_vec}")
print("-" * 30)

for pos in range(3):
    rope_vec = apply_rope(pos)
    print(f"\nPOSITION {pos}:")
    print(f"Angles: Pair1={pos*freq1}°, Pair2={pos*freq2}°")
    print(f"Result: {np.round(rope_vec, 3)}")
    
    # Show that the 'magnitude' (length) never changes
    mag = np.linalg.norm(rope_vec)
    print(f"Vector Length: {mag:.2f} (Stays constant!)")

print("\n--- CONCLUSION ---")
print("Rotation encodes the position without changing the word's 'energy'.")
print("Different speeds for different pairs create a unique signature.")
