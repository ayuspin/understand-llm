# Understand LLM

A foundational exploration of Large Language Models (LLMs), breaking down the black box into pure mathematics and code without the use of confusing analogies.

**🚀 [Live Interactive Tutorial](https://ayuspin.github.io/understand-llm/)**


## Project Goal
Understand the fundamental building blocks of neural networks and language models:
- How text is converted into numbers (Embeddings).
- How neural network layers work (Matrix Multiplication, ReLU).
- How probability distributions are created (Softmax).
- How models track word order (Positional Encoding).

## Project Structure

### [text_to_model.py](text_to_model.py)
The main codebase demonstrating a functional mini-LLM with:
- **Tokenizer**: Mapping words to IDs.
- **Embeddings**: Multi-dimensional word representations.
- **Forward Pass**: Multi-layer processing with ReLU activation.
- **Manual Backpropagation**: A step-by-step demonstration of training.

### [Tutorial Scripts](docs/scripts/)
Bite-sized Python scripts powering the [Interactive Tutorial](https://ayuspin.github.io/understand-llm/).

### [FAQ.md](FAQ.md)
A comprehensive technical reference covering key concepts and terminology.

### [roadmap.md](roadmap.md)
The path forward to Attention and Transformers.

## Getting Started

### Prerequisites
- Python 3.8+
- NumPy

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy
```

### Running
```bash
python3 text_to_model.py
```

## License
MIT License - see the [LICENSE](LICENSE) file for details.
