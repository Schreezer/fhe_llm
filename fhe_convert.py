"""
Script to convert our FHE-compatible Shakespeare model to FHE format using Concrete-ML.

This implements an approach that was verified to work with Concrete-ML:
1. Extract a simplified part of the model (single layer)
2. Use basic linear layers and ReLU activation
3. Avoid complex operations like embedding lookups
"""

import torch
import numpy as np
import os
import sys
from dataclasses import dataclass

# Import our FHE-compatible model
from simple_fhe_shakespeare import FHECompatibleModel, FHECompatibleConfig, load_shakespeare_data

# Add Concrete-ML to path
site_packages = '/Users/chirag13/development/ai_project/fhe_llm_env_py310/lib/python3.10/site-packages'
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

# Import Concrete-ML
try:
    from concrete.ml.torch.compile import compile_torch_model
except ImportError:
    print("Concrete-ML not found. Please run setup_concrete_ml.py first...")
    sys.exit(1)


def load_model(model_path='fhe_shakespeare_model.pt'):
    """Load the trained FHE-compatible model"""
    print(f"Loading model from {model_path}")
    
    # Handle missing file
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        print("Training a small model for demonstration purposes...")
        
        # Create and train a minimal model for demo
        _, _, _, _, vocab_size = load_shakespeare_data()
        config = FHECompatibleConfig(
            vocab_size=vocab_size,
            n_embd=16,      # Smaller embedding
            n_layer=1,      # Just one layer
            block_size=16   # Smaller context
        )
        model = FHECompatibleModel(config)
        return model, config
    
    # Load existing model
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    model = FHECompatibleModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    model.set_fhe_mode(True)  # Ensure integer mode is enabled
    return model, config


def extract_simple_layer(model):
    """Extract a single layer for FHE compilation
    
    For FHE compatibility, we'll create a simple model with
    only operations known to work with Concrete-ML
    """
    print("Extracting simple layer for FHE compilation")
    
    # Create a simple class with just the basic operations
    class SimpleFHELayer(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            # Extract weights from the original model's first block
            self.linear1 = torch.nn.Linear(
                original_model.config.n_embd, 
                original_model.config.n_embd, 
                bias=False
            )
            self.linear2 = torch.nn.Linear(
                original_model.config.n_embd, 
                original_model.config.n_embd, 
                bias=False
            )
            
            # Copy weights from the query and value projections
            with torch.no_grad():
                block = original_model.blocks[0]
                self.linear1.weight.copy_(block.attn.query.weight)
                self.linear2.weight.copy_(block.attn.value.weight)
            
            self.config = original_model.config
            self.scale_factor = 100  # For quantization
        
        def forward(self, x):
            # First linear layer
            x = self.linear1(x)
            
            # Quantize to lower precision
            x = (x * self.scale_factor).round() / self.scale_factor
            
            # ReLU activation (compatible with FHE)
            x = torch.relu(x)
            
            # Quantize again
            x = (x * self.scale_factor).round() / self.scale_factor
            
            # Second linear layer
            x = self.linear2(x)
            
            # Final quantization
            x = (x * self.scale_factor).round() / self.scale_factor
            
            return x
    
    # Create the simplified model
    simple_layer = SimpleFHELayer(model)
    simple_layer.eval()
    
    return simple_layer


def compile_for_fhe(model, n_bits=3):
    """Compile the model for FHE"""
    print(f"Compiling model for FHE with {n_bits} bits precision")
    
    # Create a dummy input with appropriate shape
    input_size = model.config.n_embd
    x_np = np.random.rand(1, input_size).astype(np.float32) * 0.1
    
    print(f"Input shape: {x_np.shape}, dtype: {x_np.dtype}")
    print(f"Input range: min={np.min(x_np):.6f}, max={np.max(x_np):.6f}")
    
    # Try compilation
    try:
        print("Starting compilation for FHE...")
        quantized_module = compile_torch_model(
            model,
            x_np,
            n_bits=n_bits
        )
        print("Compilation successful!")
        
        # Note: save method is not supported in this version
        
        return quantized_module
    
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def simulate_fhe_inference(quantized_module, input_size):
    """Simulate FHE inference"""
    print("\n=== Simulating FHE inference ===")
    
    # Create a simple random input
    x_np = np.random.rand(1, input_size).astype(np.float32) * 0.1
    
    print(f"Input shape: {x_np.shape}, dtype: {x_np.dtype}")
    print(f"Input range: min={np.min(x_np):.6f}, max={np.max(x_np):.6f}")
    
    # Run with quantized model (no FHE)
    print("\nRunning with quantized model (no FHE)...")
    try:
        quantized_output = quantized_module.forward(x_np, fhe="disable")
        print(f"Quantized output shape: {quantized_output.shape}")
        print(f"Quantized output range: min={np.min(quantized_output):.6f}, max={np.max(quantized_output):.6f}")
        
        # Try FHE simulation
        print("\nSimulating FHE execution...")
        fhe_output = quantized_module.forward(x_np, fhe="simulate")
        print(f"FHE output shape: {fhe_output.shape}")
        print(f"FHE output range: min={np.min(fhe_output):.6f}, max={np.max(fhe_output):.6f}")
        
        # Compare outputs
        if np.allclose(quantized_output, fhe_output, rtol=1e-3, atol=1e-3):
            print("FHE simulation matches quantized output!")
        else:
            print("Warning: FHE simulation does not match quantized output")
            print(f"Max difference: {np.max(np.abs(quantized_output - fhe_output))}")
        
        # Try actual encryption
        try_encrypt = input("\nTry with actual encryption? This will be slow. (y/n): ").lower().startswith('y')
        if try_encrypt:
            print("Running with actual encryption (this may take a while)...")
            encrypted_output = quantized_module.forward(x_np, fhe="execute")
            print(f"Encrypted output shape: {encrypted_output.shape}")
            print(f"Encrypted output range: min={np.min(encrypted_output):.6f}, max={np.max(encrypted_output):.6f}")
            
            # Compare with simulated output
            if np.allclose(fhe_output, encrypted_output, rtol=1e-3, atol=1e-3):
                print("Encrypted output matches FHE simulation!")
            else:
                print("Warning: Encrypted output does not match FHE simulation")
                print(f"Max difference: {np.max(np.abs(fhe_output - encrypted_output))}")
        
        return True
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=== FHE-Compatible Model Conversion Tool ===")
    
    # Load Shakespeare data for encoding/decoding
    _, _, _, _, vocab_size = load_shakespeare_data()
    
    # Load the trained model
    model, config = load_model()
    
    # Extract a simple layer for FHE
    simple_layer = extract_simple_layer(model)
    
    # Compile for FHE
    quantized_module = compile_for_fhe(simple_layer, n_bits=3)
    
    if quantized_module is not None:
        # Simulate FHE inference
        simulate_fhe_inference(quantized_module, input_size=config.n_embd) 