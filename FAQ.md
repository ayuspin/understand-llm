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

### **17. What is Softmax, and why do we need it?**

Softmax is a mathematical function that converts "Logits" (raw, messy scores) into a "Probability Distribution" (clean percentages that add up to 100%).

The process has three steps:
1.  **Exponentiation ($e^x$):** Raise $e$ (2.718...) to the power of each score. This ensures every result is positive and makes large scores significantly larger than small ones.
2.  **Summing:** Add up all the exponentiated values to find the "total budget."
3.  **Normalizing:** Divide each individual value by the total budget.

This is essential because models need a standardized way to compare outputs and decide which word is most likely. It also enables "Temperature" settings and allows the model to distribute "Attention" across multiple words.


### **18. Why is Positional Encoding a vector and not just a single number?**

There are three main reasons:
1.  **Signal-to-Noise:** Adding a single number (like '1') to one feature is easy for the model to lose or confuse with other features. A vector spreads the "watermark" across the entire word, making it more robust.
2.  **Scale:** If you use huge numbers like 1, 2, ..., 500, they will overwhelm the small decimal values of the word vectors. A vector allows us to keep the values small while still being unique.
3.  **Dimensional Independence:** In high-dimensional space, the model can learn to use specific "lanes" of the vector for meaning and others for position without them interfering with each other.

### **19. What is RoPE (Rotary Positional Embedding), and why is it the modern standard?**

RoPE is the method used by models like Llama 3 and GPT-NeoX to handle position. Instead of adding a position vector to the word vector, RoPE **rotates** the vector in mathematical space.

**How it works:**
The word vector is split into pairs (like $x, y$ coordinates). For a word at position $m$, each pair is rotated by an angle $\theta$. 

**Why it's better:**
1.  **Relative Distance:** Rotation naturally preserves the distance between words. The "angle" between two words is identical whether they are at indices 1 and 2 or indices 1001 and 1002.
2.  **No Boundary:** Because it’s a mathematical formula (Sines/Cosines), the model can potentially handle sequences longer than those it was trained on (Extrapolation).
3.  **Stability:** Rotation doesn't change the "norm" (energy) of the word vector, keeping the math stable through deep layers.

**How is it implemented efficiently?**
In a real LLM with 12,000+ features, we don't calculate thousands of rotations one-by-one in a loop. Instead:
- **Pre-computed Matrices:** The model pre-calculates a massive matrix of Sine and Cosine values for every possible position.
- **Vectorized Math:** Using GPUs, the model applies all rotations across the entire vector in a single parallel operation: `New_Vector = (Old_Vector * Cos) + (Swapped_Vector * Sin)`.
- **Frequency Spectrum:** Not all pairs rotate at the same speed. We use a **decaying exponential formula** (usually $\theta_i = 10000^{-2i/d}$) so that:
    - The first pairs rotate **fast** (capturing local order).
    - The last pairs rotate **extremely slowly** (capturing long-range structure).

**What happens if a vector turns 360 degrees?**
If we only had one pair, the model would eventually see the same orientation twice (Aliasing), making it impossible to tell Word #1 from Word #7. 

RoPE solves this by having **thousands of pairs** (hands) all rotating at different speeds. Like a **combination lock** with multiple dials, the unique combination of all 6,000+ dial positions will not repeat for trillions of tokens. Even if the "seconds hand" (fast pair) wraps around, the "hour hand" (slow pair) has only moved a fraction of a degree, keeping the absolute position unique.


### **20. What is Self-Attention, and why is it a "Search Engine" for words?**

Self-Attention allows a word to "look around" the sentence and pull in information from other words that are relevant to its meaning. It uses a **Query, Key, and Value** system:
- **Query (The Search)**: What kind of information am I looking for? (e.g., "I am a verb, looking for my subject").
- **Key (The Metadata)**: What kind of information do I offer? (e.g., "I am a noun, I can be a subject").
- **Value (The Content)**: The actual data I carry (e.g., the concept of a "cat").

By matching Queries to Keys using a **Dot Product**, the model builds a custom weight matrix for every sentence, letting words "attend" to their context dynamically.

### **21. Why do we divide the Attention scores by $\sqrt{d}$ (Scaling Factor)?**

In high-dimensional space (large $d$), the Dot Product of two vectors can result in very large numbers. When these large numbers are fed into the **Softmax** function, it causes the function to "saturate." 

Saturation means that one value becomes nearly 1.0 and all others become nearly 0. When this happens, the gradient (the information the model uses to learn) becomes near-zero, and the model stops being able to train effectively. Dividing by $\sqrt{d}$ keeps the values in a range where the Softmax function is still "sensitive" and able to distribute attention across multiple words.

### **22. Why do we use multiple "Heads" in Attention instead of one large one?**

Language is multi-dimensional. In a single sentence, a word might have a **grammatical** relationship with one word, a **logical** relationship with another, and a **rhyming** relationship with a third.

If we only had one Attention mechanism, the model would have to pick the "strongest" relationship and ignore the others, or blur them together into a messy average. 

**By using Multi-Head Attention:**
- **Projection**: Instead of just slicing the data, each head uses its own learned weight matrices ($W_Q, W_K, W_V$). This acts as a "filter" that projects the word into a specialized space.
- **Specialization**: Each head can learn to look for one specific thing (e.g., "Head 1 looks for grammatical subjects," "Head 8 looks for punctuation," "Head 12 looks for emotional context").
- **Resolution**: Like having multiple cameras recording a scene from different angles, Multi-Head Attention allows the model to "see" overlapping patterns in the data simultaneously, resulting in a much richer context vector.

