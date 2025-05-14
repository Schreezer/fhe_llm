"""
Quantization utilities for FHE-compatible models.
"""

import torch
import torch.nn as nn
import onnx # Add onnx import
from concrete.ml.torch.compile import compile_torch_model

def quantize_model_weights(model, bits=8):
    """
    Quantize model weights to the specified bit precision.

    Args:
        model: The PyTorch model to quantize
        bits (int): Bit precision for quantization (default: 8)

    Returns:
        model: The quantized model
    """
    # Create a copy of the model to avoid modifying the original
    quantized_model = type(model)(model.config)

    # Copy the original weights
    quantized_model.load_state_dict(model.state_dict())

    # Quantize each parameter
    for name, param in quantized_model.named_parameters():
        if param.requires_grad:
            # Calculate quantization range
            max_val = torch.max(torch.abs(param)).item()

            # Skip if max_val is 0 (all zeros in the parameter)
            if max_val == 0:
                continue

            scale = (2 ** (bits - 1) - 1) / max_val

            # Quantize the parameter
            param.data = torch.round(param.data * scale) / scale

    return quantized_model

def prepare_module_for_fhe(module, input_shape, n_bits=8):
    """
    Prepare a PyTorch module for FHE compilation using Concrete ML.

    Args:
        module: PyTorch module to compile
        input_shape: Shape of the input tensor (batch_size, seq_len)
        n_bits (int): Bit precision for quantization

    Returns:
        fhe_circuit: The compiled FHE circuit
    """
    # Create a dummy input tensor for compilation
    dummy_input = torch.zeros(input_shape, dtype=torch.long)

    # Export to ONNX for debugging before attempting FHE compilation
    onnx_export_path = "fhe_model_debug.onnx"
    print(f"Exporting model to ONNX for debugging: {onnx_export_path}")
    try:
        torch.onnx.export(
            module,
            dummy_input,
            onnx_export_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {1: 'sequence'}, 'output': {1: 'sequence'}},
            opset_version=14 # Use a version compatible with concrete-ml
        )
        print("ONNX export successful.")
    except Exception as export_e:
        print(f"Warning: ONNX export failed with '{export_e}'.")

    # Compile the model for FHE using Concrete ML,
    # fallback to module if unsupported ops are encountered
    try:
        print("Attempting FHE compilation with Concrete ML...")
        fhe_circuit = compile_torch_model(module, dummy_input, n_bits=n_bits)
        print("FHE compilation successful.")
        return fhe_circuit
    except Exception as compile_e: # Catch any exception during compilation
        print(f"Warning: FHE compilation failed with '{compile_e}'. Falling back to plain module.")
        # Fallback: No FHE circuit, will use clear model
        return None

class FHEWrapper:
    """
    Wrapper class for FHE-enabled nanoGPT model.
    """
    def __init__(self, model, n_bits=8):
        """
        Initialize the FHE wrapper.

        Args:
            model: The FHE-compatible model
            n_bits (int): Bit precision for quantization
        """
        self.model = model
        self.n_bits = n_bits
        self.config = model.config

        # Quantize the model
        self.quantized_model = quantize_model_weights(model, bits=n_bits)

        # FHE circuit (to be compiled later)
        self.fhe_circuit = None

    def compile(self, input_shape=(1, 16)):
        """
        Compile the model for FHE execution.

        Args:
            input_shape: Shape of the input tensor (batch_size, seq_len)

        Returns:
            self: For method chaining
        """
        # Create a forward-only version of the model for FHE compilation
        class ForwardModel(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.original_model = original_model
                self.config = original_model.config

            def forward(self, idx):
                logits, _ = self.original_model(idx)
                return logits

        fhe_model = ForwardModel(self.quantized_model)

        # Prepare the model for FHE
        self.fhe_circuit = prepare_module_for_fhe(
            fhe_model,
            input_shape,
            n_bits=self.n_bits
        )

        return self

    def generate_clear(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text using the clear (non-FHE) model.

        Args:
            idx: Input token IDs
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Temperature for sampling
            top_k (int, optional): Top-k sampling parameter

        Returns:
            idx: Generated token IDs
        """
        return self.quantized_model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

    def generate_fhe(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text using the compiled FHE circuit step-by-step.

        Args:
            idx: Input token IDs (LongTensor)
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            top_k (int, optional): Top-k sampling parameter

        Returns:
            idx: Generated token IDs
        """
        if self.fhe_circuit is None:
            raise ValueError("Model must be compiled for FHE first using compile()")
        device = idx.device
        # Work with a copy to avoid modifying original tensor
        seq = idx.clone()
        for _ in range(max_new_tokens):
            # Crop to block size if necessary
            cond = seq if seq.size(1) <= self.config.block_size else seq[:, -self.config.block_size:]
            # Run FHE circuit: convert to NumPy
            cond_np = cond.cpu().numpy().astype('int64')
            out_np = self.fhe_circuit.run(cond_np)
            # out_np may be shape (batch, seq_len, vocab) or (batch, vocab)
            if out_np.ndim == 3:
                logits_np = out_np[:, -1, :]
            else:
                logits_np = out_np
            # Convert back to tensor
            logits = torch.from_numpy(logits_np).to(device)
            # Apply temperature scaling
            logits = logits / temperature
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1].unsqueeze(1)] = -float('Inf')
            # Softmax and sample
            probs = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            # Append and continue
            seq = torch.cat([seq, token.to(device)], dim=1)
        return seq
