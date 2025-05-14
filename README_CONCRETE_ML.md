# Encryption-friendly LLM Architecture with Concrete-ML

This project implements an encryption-friendly language model architecture that can perform inference on encrypted data using Fully Homomorphic Encryption (FHE). The implementation uses a transformer architecture based on nanoGPT with modifications to make it compatible with Concrete-ML's FHE capabilities.

## Overview

The project demonstrates how to adapt transformer models to work with FHE using Concrete-ML. It includes:

1. **Setup Scripts**: Tools to properly set up the Concrete-ML environment
2. **FHE-Compatible Models**: Transformer models modified to work with FHE
3. **Example Scripts**: Scripts to demonstrate the FHE capabilities

## Latest Achievement: FHE-Reduced Bit-width GPT

Our newest model (`fhe_nanogpt_reduced_bitwidth.py`) achieves perfect FHE compatibility with:
- 3 transformer layers
- 64-dimensional embeddings
- 32 token context window
- 256 vocabulary size
- 0.13M parameters
- Perfect accuracy under FHE (MAE: ~1.77e-17)

## Requirements

- Python 3.10
- PyTorch
- Concrete-ML 1.9.0
- NumPy

## Directory Structure

- `setup_concrete_ml.py`: Script to set up the Concrete-ML package structure
- `fhe_nanogpt_reduced_bitwidth.py`: Latest successful FHE model with explicit bit-width control
- `fhe_nanogpt_concrete_v3.py`: Reference implementation of minimal FHE models
- `fhe_nanogpt_final_v2.py`: Previous minimal FHE transformer model
- `fhe_nanogpt_final.py`: FHE-compatible version of nanoGPT
- `fhe_nanogpt_concrete_v2.py`: Implementation of FHE-friendly attention mechanism
- `fhe_nanogpt_concrete.py`: Initial attempt at creating a hybrid FHE model
- `concrete_ml_basic.py`: Basic examples of using Concrete-ML

## Getting Started

1. **Set up the environment**:
   ```bash
   # Create and activate a Python 3.10 virtual environment
   python3.10 -m venv fhe_llm_env_py310
   source fhe_llm_env_py310/bin/activate
   
   # Install dependencies
   pip install torch numpy
   ```

2. **Run the setup script**:
   ```bash
   python setup_concrete_ml.py
   ```

3. **Run the latest FHE model**:
   ```bash
   python fhe_nanogpt_reduced_bitwidth.py
   ```

## FHE-Compatible Transformer Architecture

Our latest FHE-compatible transformer architecture introduces several key innovations:

1. **Explicit Bit-width Control**:
   - Scaling between layers (x * 0.1)
   - Modified residual connections using averaging
   - Aggressive weight initialization (std=0.005)

2. **Input Processing**:
   - Token embeddings scaled by 0.01
   - Positional embeddings scaled by 0.01
   - One-hot encoded inputs for FHE circuit

3. **Transformer Layers**:
   - FHE-friendly attention with explicit scaling
   - Polynomial activation (x² + x) with scaling
   - Bit-width reduction between blocks

4. **Output Processing**:
   - Scaled language model head (0.5)
   - Direct logits output without softmax

## Key Components

### FHEReducedBitGPT

The `FHEReducedBitGPT` class in `fhe_nanogpt_reduced_bitwidth.py` is our most successful FHE-compatible model. It achieves:
- Perfect accuracy in both simulation and encryption
- Successful compilation with 3-bit quantization
- Stable numerical behavior through multiple layers

### FHE-Friendly Attention

Our latest attention implementation:
- Uses aggressive scaling (0.01) for queries and keys
- Implements polynomial activation for attention weights
- Scales attention outputs for bit-width control

### FHE-Friendly MLP

The improved MLP implementation:
- Uses scaled polynomial activation
- Implements explicit bit-width control
- Removes biases for better FHE compatibility

## Best Practices for FHE-Compatible Models

Updated based on our latest findings:

1. **Bit-width Control**:
   - Use explicit scaling between operations (x * 0.1)
   - Implement averaging for residual connections
   - Initialize weights with very small values (std=0.005)

2. **Architecture Design**:
   - Use single-head attention instead of multi-head
   - Remove unnecessary biases from linear layers
   - Keep embedding dimensions small (≤64)
   - Limit number of layers (≤3)

3. **Quantization**:
   - Use 3-bit quantization for maximum compatibility
   - Scale inputs and embeddings aggressively
   - Monitor bit-width growth through layers

4. **FHE Circuit Design**:
   - Use one-hot encoded inputs
   - Avoid complex reshaping operations
   - Keep sequence lengths manageable (≤32)

## Limitations

1. **Model Size Constraints**:
   - Maximum tested size: 0.13M parameters
   - Maximum embedding dimension: 64
   - Maximum layers: 3
   - Maximum context window: 32 tokens

2. **Input Processing**:
   - Requires one-hot encoded inputs
   - Limited vocabulary size (256)
   - Token-by-token generation

3. **Computational Overhead**:
   - FHE compilation time
   - Encryption/decryption overhead
   - Memory requirements for FHE operations

## Future Work

1. **Model Scaling**:
   - Test larger embedding dimensions
   - Experiment with more layers
   - Increase context window size

2. **Optimization**:
   - Reduce compilation time
   - Optimize memory usage
   - Improve generation efficiency

3. **Architecture Improvements**:
   - Develop more efficient attention mechanisms
   - Test alternative activation functions
   - Implement efficient token generation

## References

- [Concrete-ML Documentation](https://docs.zama.ai/concrete-ml/)
- [FHE Constraints and Best Practices](https://docs.zama.ai/concrete-ml/advanced-topics/constraints)
- [nanoGPT](https://github.com/karpathy/nanoGPT) 