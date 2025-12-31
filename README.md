# Understand LLM

A foundational exploration of Large Language Models (LLMs), breaking down the black box into pure mathematics and code without the use of confusing analogies.

**🚀 [Live Interactive Tutorial](https://ayuspin.github.io/understand-llm/)**


## Project Goal
The goal of this project is to understand the fundamental building blocks of neural networks and language models:
- How text is converted into numbers (Vectorization/Embeddings).
- How "learning" works via dot products and weight matrices.
- Why non-linearity (ReLU) and multi-layer hierarchies are necessary.
- The mechanical reality of matrix multiplication in modern AI.

## Project Structure

### 1. [digit_basics.py](digit_basics.py)
A starting point demonstrating the concept of **Template Matching**. It shows that "recognizing" a digit is essentially just calculating the similarity (dot product) between an input and a set of learned patterns.

### 2. [text_to_model.py](text_to_model.py)
A functional mini-LLM implementation containing:
- **Tokenizer**: Mapping words to IDs.
- **Embeddings**: Multi-dimensional word representations.
- **Forward Pass**: Multi-layer processing with ReLU activation.
- **Manual Backpropagation**: A step-by-step demonstration of how errors are used to update model weights.

### 3. [matrix_basics.py](matrix_basics.py)
An educational script designed to clarify the orientation and mechanics of matrix multiplication. It bridges the gap between textbook math ($Wx$) and real-world AI code standard ($xW$), providing visual ASCII diagrams of the "Row-by-Column" process.

### 4. [FAQ.md](FAQ.md)
A comprehensive technical reference capturing 16 key questions and answers from the exploration process.

### 5. [roadmap.md](roadmap.md)
The path forward. Outlines the transition from basic math to sequential context and the **Attention Mechanism**.

## Getting Started

### Prerequisites
- Python 3.8+
- NumPy

### Setup
```bash
# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy
```

### Running the Scripts
Each script is self-contained and provides printed explanations in the terminal:
```bash
python3 digit_basics.py
python3 text_to_model.py
python3 matrix_basics.py
```

## License
MIT License - see the [LICENSE](LICENSE) file for details.
