# GPT Model Implementation

A PyTorch-based implementation of a GPT (Generative Pre-trained Transformer) language model from scratch.

## Project Overview

This project implements a 124M parameter GPT model with the following architecture:
- **Vocabulary Size**: 50,257 (GPT-2 tokenizer)
- **Context Length**: 256 tokens
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Layers**: 12
- **Dropout Rate**: 0.1

## Key Components

- **`gpt.py`** - Main training script and model orchestration
- **`GPT_model/model.py`** - GPT architecture implementation (MultiHeadAttention, Transformer layers)
- **`tokenizer/tokenizer_gpt.py`** - Custom data loading and tokenization utilities
- **`checker.py`** - Utility functions for model validation
- **`file_reader.py`** - File I/O helpers

## Dataset

Trained on **WikiText-2** dataset with split into:
- Training data: `wikitext2_train.txt`
- Validation data: `wikitext2_val.txt`
- Test data: `wikitext2_test.txt`

## Setup

### Requirements

- PyTorch >= 2.2.2
- tiktoken >= 0.5.1 (GPT-2 tokenizer)
- matplotlib >= 3.7.1 (visualization)
- tqdm >= 4.66.1 (progress bars)
- numpy >= 1.26

### Installation

1. Create a virtual environment:
```bash
python -m venv my_proj
source my_proj/Scripts/activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run training:
```bash
python gpt.py
```

## Features

- Multi-head self-attention mechanism
- Causal attention masking for autoregressive generation
- Mixed precision training (AMP)
- Text generation capabilities
- Training visualization with matplotlib

## Documentation

- `Theories/GPT1_theory.md` - Theoretical background on GPT models
- `optimizer/introduction.md` - Optimizer documentation
- `Theories/asked_questions.md` - Q&A on implementation details
