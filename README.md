# Understand LLM

A foundational exploration of Large Language Models (LLMs), breaking down the black box into pure mathematics and code without the use of confusing analogies.

**🚀 [Live Interactive Tutorial](https://ayuspin.github.io/understand-llm/)**


## Project Goal
The goal of this project is to understand the fundamental building blocks of neural networks and language models:
- How text is converted into numbers ([Embeddings](docs/scripts/step3_ai_orientation.py)).
- How neural network [Layers](text_to_model.py) work.
- Why [ReLU](text_to_model.py) is necessary.
- [Matrix Multiplication](docs/scripts/step2_matrix_multiply.py) in AI code.
- [Softmax](docs/scripts/step4_softmax.py) and Probability.

## Project Structure

### 1. [text_to_model.py](text_to_model.py)
The main codebase demonstrating a functional mini-LLM with:
- **Tokenizer**: Mapping words to IDs.
- **Embeddings**: Multi-dimensional word representations.
- **Forward Pass**: Multi-layer processing with ReLU activation.
- **Manual Backpropagation**: A step-by-step demonstration of training.

### 2. [Tutorial Scripts](docs/scripts/)
A collection of bite-sized Python scripts that power the interactive tutorial:
- `step1_dot_product.py`: Understanding similarity.
- `step2_matrix_multiply.py`: The "Row-by-Column" rule.
- `step3_ai_orientation.py`: How LLMs use $xW$ orientation.
- `step4_softmax.py`: Turning scores into probabilities.

### 3. [FAQ.md](FAQ.md)
A comprehensive technical reference capturing key concepts and terminology.

### 4. [roadmap.md](roadmap.md)
The path forward to context windows, Attention, and Transformers.

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
The main model can be run directly:
```bash
python3 text_to_model.py
```
Tutorial scripts can be found in `docs/scripts/` and are also accessible via the [Live Tutorial](https://ayuspin.github.io/understand-llm/).

## License
MIT License - see the [LICENSE](LICENSE) file for details.
