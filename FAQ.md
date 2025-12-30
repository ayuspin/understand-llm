# LLM Fundamentals: FAQ

This document captures the technical Q&A from our exploration of how Large Language Models (LLMs) work, starting from basic digit recognition.

### **1. How does a computer "recognize" a hand-written digit?**
It uses a process similar to "Template Matching." For a 16x16 image (256 pixels), the model has a **Weight Matrix** containing "templates" for each digit (0-9). Each template is a list of 256 numbers. The computer performs a **Dot Product** between the image's pixels and these templates. The template with the highest "match" score identifies the digit.

### **2. Why use this same "pixel/matrix" approach for language?**
Because computers can only do math on numbers. To treat words like images, we turn them into **Vectors** (lists of numbers). Instead of pixels representing "darkness," these numbers represent abstract **Features** (e.g., how "food-like" or "living" a word is). This allows the computer to use the same dot-product math to find relationships between words.

### **3. How does the computer know the specific numbers for a word (e.g., Pear = [0.0, 0.1, 0.9])?**
It doesn't know initially. It starts with **random numbers**. During training, it tries to predict the "neighbor" of a word in a sentence. When it guesses wrong, it uses **Calculus (Backpropagation)** to nudge the numbers for that word closer to its neighbors. Over time, words that appear in similar contexts (like "apple" and "pear") naturally end up with similar numbers.

### **4. Where do the "Template" numbers come from?**
They are learned at the same time as the word numbers. The model's matrices are essentially "carved" by errors. If the model fails to predict "cat" after "the," it slightly adjusts both the `the` word vector and the `cat` prediction weights (templates) until they align better.

### **5. What is a Tokenizer?**
It is the "Translator" between human text and computer IDs. In our code, it involves:
- **`replace(".", "")`**: Cleaning punctuation so "sat." and "sat" are treated as the same identity.
- **`.split()`**: Breaking a string into a list of individual words (tokens).
- **`word_to_id`**: A dictionary that maps a word (e.g., "cat") to a unique index (e.g., 1).

### **6. Why is `vector_size` (Embedding Dimension) a constant like 4 or 12,288?**
It represents the "resolution" or capacity of the model's understanding. 
- A **small number (4)** can only track a few simple features.
- A **large number (12,288)** allows the model to capture thousands of tiny nuances in meaning.
It is a **Hyperparameter**—a setting chosen by the designers before training begins.

### **7. Why do we adjust the Embeddings during training? Why not keep them fixed?**
We adjust them so the model can "draw its own map." By allowing the embedding vectors to move, the model discovers which words are similar based on how they are used. This "self-organization" is what allows AI to understand that "dog" and "puppy" are related without being explicitly told.

### **8. What are "Logits" and "Argmax"?**
- **Logits**: The raw, unrefined scores that come out of the dot-product math.
- **Argmax**: A function that looks at those scores and returns the **Index** of the highest one. It turns "Probability Math" into a "Final Decision."
