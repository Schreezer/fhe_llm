# Privacy-Preserving LLMs with Fully Homomorphic Encryption
## By Arpit Kumar, Chirag Aggarwal, Sujal

---

## Introduction & Motivation

- **Privacy concerns** are growing with the widespread adoption of Large Language Models
- Users must share sensitive queries with service providers, creating privacy risks
- **Fully Homomorphic Encryption (FHE)** enables computation on encrypted data
- Allows LLMs to process user queries without seeing the actual content
- Our motivation: Learn and implement privacy-preserving ML techniques, focusing on FHE for LLMs

---

## Problem Formulation

### Key Challenges in Applying FHE to LLMs:

1. **Computational Complexity**: FHE operations are extremely computationally intensive
2. **Bit-width Growth**: Numerical precision expands with each operation in FHE
3. **Non-polynomial Functions**: Standard activation functions (ReLU, GELU, softmax) are incompatible with FHE
4. **Model Size**: Large models are prohibitively expensive to run under FHE
5. **Resource Requirements**: State-of-the-art approaches require significant compute resources

---

## Journey & Methodology

### Initial Exploration:
- Studied transformer architectures and nanoGPT implementation
- Explored HE-LLM library (found it required 32GB RAM + 8 GPUs)
- Investigated Concrete-ML as a more accessible alternative

### Key Challenges Encountered:
- Bit-width explosion during model execution
- Activation function limitations
- Residual connection issues
- Quantization challenges

### Progressive Implementation Approach:
1. Started with simple linear networks in FHE
2. Advanced to basic transformer blocks
3. Finally built a reduced-size nanoGPT variant

---

## Proposed Methodology

### FHE-Compatible Transformer Architecture:

1. **Bit-width Control**:
   - Explicit scaling between operations (x * 0.1)
   - Modified residual connections using averaging (x * 0.5 + y * 0.5)
   - Small weight initialization (std=0.005)

2. **FHE-Friendly Operations**:
   - Polynomial activation function (x² + x) with scaling
   - Simplified attention mechanism (single-head)
   - Element-wise operations where possible
   - Aggressive quantization (3-bit)

3. **Architectural Modifications**:
   - Reduced model size (0.13M parameters)
   - Smaller embeddings (64-dimensional)
   - Fewer layers (3 transformer blocks)
   - Limited context window (32 tokens)
   - Reduced vocabulary (256 tokens)

---

## Novelty

### Key Innovations in Our Implementation:

1. **Explicit Bit-width Reduction Strategy**:
   - Strategic scaling between layers prevents bit-width explosion
   - Novel approach to residual connections (averaging vs addition)
   - Progressive quantization throughout the network

2. **FHE-Optimized Attention Mechanism**:
   - Replaced softmax with polynomial approximation
   - Aggressive scaling of attention weights
   - Single-head attention with controlled numerical precision

3. **End-to-End FHE Compatibility**:
   - Successfully adapted transformer architecture for FHE execution
   - Maintained perfect accuracy between regular and FHE inference
   - Achieved stable numerical behavior through multiple transformer layers

---

## Results & Discussion

### FHE-Compatible Model Specifications:
- 3 transformer layers
- 64-dimensional embeddings
- 32 token context window
- 256 vocabulary size
- 0.13M parameters
- Perfect accuracy under FHE (MAE: ~1.77e-17)

### Comparison with Original nanoGPT:

| Feature | Original nanoGPT | FHE-compatible GPT |
|---------|------------------|-------------------|
| Block Size | 1024 | 32 |
| Vocabulary Size | 50,304 | 256 |
| Layers | 12 | 3 |
| Embedding Dimension | 768 | 64 |
| Attention Heads | 12 | 1 |
| Parameters | 124M | 0.13M |
| Activation | GELU | Polynomial (x² + x) |
| Layer Norm | Full | Simplified scaling |
| Residual | Addition | Averaging |

### FHE Performance:
- Successfully compiled with 3-bit quantization
- Perfect accuracy in both simulation and encryption modes
- Mean Absolute Error: ~1.77e-17 (effectively zero)

---

## Limitations & Future Work

### Current Limitations:
1. **Model Size**: Maximum tested size is 0.13M parameters
2. **Input Requirements**: One-hot encoded inputs for FHE circuit
3. **Generation Method**: Token-by-token generation only
4. **Context Window**: Limited to 32 tokens
5. **Computational Overhead**: FHE operations remain slow

### Future Directions:
1. **Scaling to Larger Models**: Increase embedding dimensions and layers
2. **Efficient Token Generation**: Optimize the generation process
3. **Real-world Testing**: Apply to actual applications and datasets
4. **Circuit Optimization**: Reduce compilation and execution time
5. **Reducing Encryption Overhead**: Optimize key management and encryption

---

## Conclusion

- Successfully implemented a working FHE-compatible language model
- Developed innovative techniques for bit-width control in transformers
- Created a fully functional model with perfect accuracy under encryption
- Demonstrated feasibility of privacy-preserving inference with LLMs
- Established foundation for future research in FHE-compatible deep learning

---

## Acknowledgements

- Original nanoGPT implementation by Andrej Karpathy
- Concrete-ML library by Zama