# Encryption-friendly LLM Architecture

This project implements a Fully Homomorphic Encryption (FHE) compatible approach to Large Language Models, focusing on privacy-preserving inference.

## Key Technologies
- **Fully Homomorphic Encryption**: Allows computation on encrypted data without decryption
- **Concrete-ML**: FHE library that only supports integer operations
- **PyTorch**: For model training and conversion to FHE

## Project Structure

### Main Components
- `simple_fhe_shakespeare.py`: Integer-only transformer for Shakespeare text generation
- `fhe_convert.py`: Converts the trained model to FHE format
- `fhe_convert_simple.py`: Simple demonstration of integer-only FHE model
- `fhe_nanogpt_reduced_bitwidth.py`: Original FHE-compatible nanoGPT (using floating point)

## Integer-Only FHE Implementation

Our key insight is that Concrete-ML's FHE implementation only supports integer operations. Our implementation addresses this by:

1. **Integer-only operations**: Replacing all floating point operations with integer equivalents
2. **Integer weights**: Quantizing all weights to integers by scaling, rounding, and rescaling
3. **Bit-width control**: Using bit shifts for division operations to control numerical precision
4. **FHE-friendly activation**: Using only integer-compatible activation functions (ReLU)
5. **One-hot encoding**: Using integer (0/1) one-hot encoding for token inputs

### Technical Approach

#### Integer-Only Matrix Operations
Instead of standard floating-point matrix multiplications, we:
- Scale weights by 100, round to integers, and store them as integer parameters
- Use integer matrix multiplication
- Apply integer division (via bit shifting) to rescale results

#### Avoiding Floating-Point Activations
Our activation functions are integer-only:
- ReLU: Simple max(0, x) which works with integers
- No softmax (replaced with simple integer normalization)
- No layer normalization (removed to maintain integer constraints)

#### Scaling and Bit Control
To prevent integer overflow:
- Apply integer division via bit shifting (`x >> 1` divides by 2)
- Carefully control the bit width through the model
- Quantize to integers periodically during training

## Running the Project

### Setup
1. Create a Python 3.10 environment for Concrete-ML compatibility:
```bash
python -m venv fhe_llm_env_py310
source fhe_llm_env_py310/bin/activate
```

2. Install dependencies:
```bash
pip install torch numpy
pip install concrete-ml==1.9.0
```

### Training
Train the integer-only model:
```bash
python simple_fhe_shakespeare.py --train
```

### Text Generation
Generate text with the trained model:
```bash
python simple_fhe_shakespeare.py --generate "King "
```

### FHE Conversion
Convert the trained model to FHE format:
```bash
python fhe_convert.py
```

Try the simple integer-only FHE example:
```bash
python fhe_convert_simple.py
```

## FHE Limitations and Workarounds

1. **Integer-only constraint**: Replace all floating-point ops with integer equivalents
2. **No biases**: Remove biases from all linear layers
3. **Limited activations**: Only use integer-compatible activation functions
4. **Bit width control**: Carefully manage integer width to avoid overflow
5. **Reduced dimensions**: Keep model smaller due to FHE computational overhead

## Results

Our project successfully demonstrates:
1. Training a Shakespeare language model with FHE-compatible architecture
2. Converting the model to use only integer operations
3. Compiling the model for FHE execution using Concrete-ML
4. Simulating and executing FHE inference

This work provides a foundation for privacy-preserving LLM inference, allowing computation on encrypted inputs without access to the plaintext data.

## Team
- Arpit Kumar
- Chirag Aggarwal
- Vinay Yadav