# LLM Fundamentals: FAQ

This document captures the technical Q&A from our exploration of how Large Language Models (LLMs) work, starting from basic digit recognition.

### **1. How does a computer "recognize" a hand-written digit?**
It uses a process similar to "Pattern Alignment." For a 16x16 image (256 pixels), the model has a **Weight Matrix** containing **Weight Vectors** for each digit (0-9). Each weight vector is a list of 256 numbers. The computer performs a **Dot Product** between the image's pixels and these vectors. The vector with the highest "match" score identifies the digit.

### **2. Why use this same "pixel/matrix" approach for language?**
Because computers can only do math on numbers. To treat words like images, we turn them into **Vectors** (lists of numbers). Instead of pixels representing "darkness," these numbers represent abstract **Features** (e.g., how "food-like" or "living" a word is). This allows the computer to use the same dot-product math to find relationships between words.

### **3. How does the computer know the specific numbers for a word (e.g., Pear = [0.0, 0.1, 0.9])?**
It doesn't know initially. It starts with **random numbers**. During training, it tries to predict the "neighbor" of a word in a sentence. When it guesses wrong, it uses **Calculus (Backpropagation)** to nudge the numbers for that word closer to its neighbors. Over time, words that appear in similar contexts (like "apple" and "pear") naturally end up with similar numbers.

### **4. Where do the "Weight Vector" numbers come from?**
They are learned at the same time as the word numbers. The model's matrices are essentially "carved" by errors. If the model fails to predict "cat" after "the," it slightly adjusts both the `the` word vector and the `cat` prediction weights (weight vectors) until they align better.

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

### **9. What is a "Layer" in this context?**
A layer is simply a **Weight Matrix** that sits between the Input and the Output. 
- In our simple model, we added a **Hidden Layer**.
- Instead of going straight from Word $\rightarrow$ Score, the computer now goes: Word $\rightarrow$ **Abstract Concepts** $\rightarrow$ Score.
- Each layer allows the model to "think" in more complex steps (e.g., Layer 1 understands grammar, Layer 2 understands sentiment, Layer 3 understands logic).

### **10. Why do we need the `ReLU` function (Activation Function)?**
If you just multiply 10 matrices together, mathematically it's the exact same as multiplying by 1 big matrix. To actually get the benefit of multiple layers, you need a "Non-Linear" step. 
- **ReLU (Rectified Linear Unit)** is the most common. It basically says: *"If the number is negative, turn it to 0. If it's positive, keep it."*
- This "filtering" allows the model to ignore some features and focus on others, making it much smarter than a simple linear calculator.
### **11. Is the goal just about finding a value $y$ for a given $x$?**
Exactly. In mathematics, this is called **General Function Approximation**. 
- **The Input ($x$)**: In an LLM, this is the context (e.g., the word "The").
- **The Output ($y$)**: This is the prediction (e.g., the word "cat").
- **The Process**: We are looking for a massive, multi-dimensional function $f$ such that $f(x) = y$. 
Because language is incredibly complex and non-linear, we cannot write this function by hand. Instead, we use billions of tiny numbers (Weights) and non-linear steps (ReLU) so that the computer can "learn" the shape of the function through trial and error.

### **12. Why do we need hidden layers?**
Hidden layers allow the model to learn **Hierarchies of Logic**. 
- A single layer can only understand simple, direct relationships. 
- Multiple layers allow the model to recognize "patterns of patterns." For example, Layer 1 might recognize individual lines, while Layer 2 recognizes how those lines form a square. This layered approach is how models move from simple pixels to complex concepts like "sentiment" or "logic."

### **13. Is it a strict requirement to have ReLU between every layer?**
- **Theoretically, YES**: If you have two linear layers (matrices) back-to-back without a ReLU, they mathematically collapse into a single, less powerful matrix. To keep your model "Deep" and capable of learning complex language, you must put a non-linearity (like ReLU) between every pair of weight matrices.
- **Practically, NO**: Developers sometimes stack two linear layers without a ReLU for **Compression** (called Low-Rank Factorization). This doesn't make the model smarter, but it can make it smaller and faster by replacing one giant matrix with two thinner ones.

### **14. What does it mean in reality that a layer "sees" something?**
It means **Mathematical Alignment**. Every column in a layer's weight matrix is a numerical **Weight Vector**. When the input vector is multiplied by that vector (a Dot Product), the result is a high score only if the input's numbers "align" with the weight vector's numbers. "Seeing" is simply the computer using math to measure how well the input correlates with its stored patterns.

### **15. What is the mechanical reality of Matrix Multiplication?**
It is just a massive series of **Multiplications and Additions**. 
- In the modern AI orientation ($y = xW$), your input is a horizontal **Row** of numbers.
- That row gets multiplied against each **Column** of the weight matrix. Each column is one "Weight Vector" for a specific word.
- For each column: multiply the corresponding elements and add them all up (a **Dot Product**).
- The results are collected into a new row.
There is no "magic"—it is simply a structured way to compare an input row of numbers against many different weight vectors (columns) simultaneously.

### **16. Why does AI code look "backward" compared to math textbooks?**
In math textbooks, you usually see **$y = Wx$** (Weights on the left, Input as a vertical column). 
However, in modern AI code (like NumPy and PyTorch), we often use **$y = xW$** (Input on the left as a horizontal row).
- **Why?** It is more natural for computers to process a "stream" of data as a row.
- **The Result**: In this orientation, each **Column** of the weight matrix represents a single "word" or "concept."
- **The Rule**: Whether it's $Wx$ or $xW$, the core rule is the same: the dimensions must "touch" (e.g., a Row of 8 must meet a Matrix with 8 heights).
