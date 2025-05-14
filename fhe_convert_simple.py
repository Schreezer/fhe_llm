"""
Ultra-simple FHE-compatible model for Concrete-ML.

This script creates a minimal model that uses only operations
that are confirmed to be compatible with Concrete-ML's FHE compiler.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

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


class SimpleFHEModel(nn.Module):
    """
    Ultra-simple model with operations known to be compatible with Concrete-ML.
    
    Only using:
    - Linear layers (no bias)
    - ReLU activation
    - Basic quantization
    """
    def __init__(self, input_size=8, hidden_size=16, output_size=8):
        super().__init__()
        self.scale_factor = 100  # Scaling factor for quantization
        
        # Simple feedforward network with no bias
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        
        # Initialize weights with small values
        with torch.no_grad():
            self.fc1.weight.uniform_(-0.1, 0.1)
            self.fc2.weight.uniform_(-0.1, 0.1)
        
        # FHE mode flag
        self.use_integer_ops = False
        
        print(f"Simple FHE model created with {sum(p.numel() for p in self.parameters())} parameters")
    
    def set_fhe_mode(self, enable=True):
        """Set whether to use integer-friendly operations for FHE compatibility"""
        self.use_integer_ops = enable
    
    def forward(self, x):
        """Forward pass with quantization for FHE compatibility"""
        # First linear layer
        x = self.fc1(x)
        
        # For integer ops, quantize after linear
        if self.use_integer_ops:
            x = (x * self.scale_factor).round() / self.scale_factor
        
        # ReLU activation (compatible with FHE)
        x = torch.relu(x)
        
        # For integer ops, quantize after activation
        if self.use_integer_ops:
            x = (x * self.scale_factor).round() / self.scale_factor
        
        # Second linear layer
        x = self.fc2(x)
        
        # For integer ops, quantize output
        if self.use_integer_ops:
            x = (x * self.scale_factor).round() / self.scale_factor
        
        return x


def train_simple_model(model, num_samples=100, num_epochs=50):
    """Train the model using floating point operations"""
    print("Training model with floating point operations...")
    
    # Ensure model is NOT in FHE mode during training
    model.set_fhe_mode(False)
    
    # Create a simple regression dataset
    X = torch.randn(num_samples, 8) * 0.1
    Y = torch.randn(num_samples, 8) * 0.1
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, Y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
    
    # Quantize weights after training
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Quantize to lower precision by rounding
            param.mul_(model.scale_factor).round_().div_(model.scale_factor)
    
    print("Training complete!")
    return model


def convert_to_fhe(model, n_bits=3):
    """Convert the model to FHE format"""
    print(f"Converting model to FHE with {n_bits} bits precision")
    
    # Ensure model is in FHE mode
    model.set_fhe_mode(True)
    
    # Create a dummy input with appropriate shape
    x_np = np.random.rand(1, 8).astype(np.float32) * 0.1
    
    print(f"Input shape: {x_np.shape}, dtype: {x_np.dtype}")
    print(f"Input range: min={np.min(x_np):.6f}, max={np.max(x_np):.6f}")
    
    # Try compilation with the known compatible model
    try:
        print("Starting compilation for FHE...")
        quantized_module = compile_torch_model(
            model,
            x_np,
            n_bits=n_bits
        )
        print("Compilation successful!")
        
        # Model cannot be saved in this version of Concrete-ML
        # quantized_module.save("simple_fhe_model.zip")
        # print("Model saved to simple_fhe_model.zip")
        
        return quantized_module
    
    except Exception as e:
        print(f"Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def simulate_fhe_inference(quantized_module):
    """Simulate FHE inference with the quantized model"""
    # Create a simple input
    x_np = np.random.rand(1, 8).astype(np.float32) * 0.1
    
    print(f"Input shape: {x_np.shape}, dtype: {x_np.dtype}")
    print(f"Input range: min={np.min(x_np):.6f}, max={np.max(x_np):.6f}")
    
    # Run inference with the quantized model (no FHE)
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
        
        # Try with actual encryption
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


def compare_modes(model):
    """Compare floating point vs integer-friendly operations"""
    print("\n=== Comparing Floating Point vs Integer-Friendly Operations ===")
    
    # Create a simple input
    x = torch.rand(1, 8) * 0.1
    
    # Test with floating point
    model.set_fhe_mode(False)
    with torch.no_grad():
        fp_output = model(x)
        print(f"Floating point output range: min={fp_output.min().item():.6f}, max={fp_output.max().item():.6f}")
    
    # Test with integer-friendly ops
    model.set_fhe_mode(True)
    with torch.no_grad():
        int_output = model(x)
        print(f"Integer-friendly output range: min={int_output.min().item():.6f}, max={int_output.max().item():.6f}")
    
    # Compare the two
    diff = torch.abs(fp_output - int_output).mean()
    print(f"Mean absolute difference: {diff.item():.6f}")
    
    return diff.item() < 0.1  # Check if reasonably close


if __name__ == "__main__":
    print("=== Ultra-Simple FHE Model Demonstration ===")
    
    # Create model
    model = SimpleFHEModel(input_size=8, hidden_size=16, output_size=8)
    
    # Train the model with floating point
    model = train_simple_model(model)
    
    # Compare floating point vs integer-friendly modes
    compare_modes(model)
    
    # Convert to FHE
    quantized_module = convert_to_fhe(model, n_bits=3)
    
    if quantized_module is not None:
        # Simulate FHE inference
        simulate_fhe_inference(quantized_module) 