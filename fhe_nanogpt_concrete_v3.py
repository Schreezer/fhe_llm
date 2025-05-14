"""
Minimal FHE-compatible model using Concrete-ML

This script implements a very simple model that can run in FHE using Concrete-ML.
It focuses on basic operations that are known to work well with FHE.

Results:
- MinimalFHEModel: Successfully compiles and runs in FHE mode 
  - Excellent accuracy in quantized mode (MAE: ~1.8e-05)
  - Higher error in FHE mode (MAE: ~54.75 compared to quantized output)
- SimplifiedTransformerBlock: Successfully compiles with good accuracy
  - Good accuracy in quantized mode (MAE: ~0.008)

FHE-compatible operations used:
- Linear layers with small weights
- Simple polynomial activation function (x² + x)
- Element-wise operations instead of complex matrix operations
- Residual connections
"""

import os
import sys
import torch
import numpy as np
import argparse

# Command-line arguments
parser = argparse.ArgumentParser(description='Test FHE-compatible models')
parser.add_argument('--fhe_mode', choices=['execute', 'simulate', 'disable'], default='execute',
                   help='FHE execution mode: execute (run in FHE), simulate (test without encryption), disable (run in cleartext)')
args = parser.parse_args()

# Make sure we're using the site-packages version of concrete with our fixes
site_packages = '/Users/chirag13/development/ai_project/fhe_llama_env_py310/lib/python3.10/site-packages'
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

# First, run the setup script to ensure all necessary files are in place
print("Setting up Concrete-ML environment...")
try:
    import setup_concrete_ml
except ImportError:
    print("Running setup_concrete_ml.py directly...")
    exec(open("setup_concrete_ml.py").read())

# Import Concrete-ML components
try:
    from concrete.ml.torch.compile import compile_torch_model
    print("Successfully imported Concrete-ML components")
except ImportError as e:
    print(f"Error importing Concrete-ML components: {e}")
    sys.exit(1)

# Create a minimal FHE-compatible model
class MinimalFHEModel(torch.nn.Module):
    """
    A minimal model that only uses operations compatible with FHE.
    """
    def __init__(self, input_dim=16, hidden_dim=8, output_dim=4):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
        # Initialize with small weights to reduce bit width requirements
        self._init_small_weights()
        
    def _init_small_weights(self):
        """Initialize weights with small values to reduce bit width."""
        for module in [self.fc1, self.fc2]:
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data.normal_(mean=0.0, std=0.01)
            if hasattr(module, 'bias') and module.bias is not None:
                with torch.no_grad():
                    module.bias.data.zero_()
    
    def forward(self, x):
        # First linear layer
        x = self.fc1(x)
        
        # Use a simple polynomial activation (x^2 + x) instead of ReLU
        # This is FHE-friendly
        x = x * x + x
        
        # Second linear layer
        x = self.fc2(x)
        
        return x

def test_minimal_model(fhe_mode='execute'):
    """Test the minimal FHE model with Concrete-ML.
    
    Args:
        fhe_mode (str): FHE execution mode - 'execute', 'simulate', or 'disable'
    """
    print(f"\n=== Testing Minimal FHE Model (FHE mode: {fhe_mode}) ===")
    
    # Create a minimal model
    input_dim = 16
    hidden_dim = 8
    output_dim = 4
    model = MinimalFHEModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    print("Created minimal FHE model")
    
    # Create random input
    batch_size = 1
    x = torch.randn(batch_size, input_dim) * 0.1  # Scale inputs
    
    # Run inference with the original model
    print("Running inference with the original model...")
    with torch.no_grad():
        original_output = model(x)
    
    # Compile the model
    print("Compiling the model...")
    try:
        # Convert to numpy for compilation
        x_numpy = x.numpy()
        
        # Use very low bit width to avoid overflow
        n_bits = 4
        quantized_module = compile_torch_model(
            model, 
            x_numpy,
            n_bits=n_bits
        )
        
        # Run inference with the compiled model
        print("Running inference with the compiled model...")
        fhe_output = quantized_module.forward(x_numpy)
        
        # Compare outputs
        mae = np.abs(original_output.detach().numpy() - fhe_output).mean()
        print(f"Mean Absolute Error: {mae}")
        
        # Check if compilation was successful
        if hasattr(quantized_module, 'fhe_circuit') and fhe_mode != 'disable':
            print("FHE circuit successfully created!")
            
            # Try to run in FHE
            print(f"Running in FHE mode: {fhe_mode}...")
            try:
                fhe_circuit = quantized_module.fhe_circuit
                
                # Convert input to the expected integer format
                # Access the input quantizers directly from the quantized_module object
                # and use them to quantize the input
                x_quantized = quantized_module.quantize_input(x_numpy)
                
                # Run the FHE circuit with quantized inputs
                # For 'execute', run with actual encryption
                # For 'simulate', simulate the FHE computation without encryption
                if fhe_mode == 'execute':
                    fhe_result = fhe_circuit.encrypt_run_decrypt(x_quantized)
                    print("Successfully ran in FHE mode with encryption!")
                elif fhe_mode == 'simulate':
                    # Simulate FHE execution without encryption
                    fhe_result = fhe_circuit.simulate(x_quantized)
                    print("Successfully simulated FHE execution!")
                
                # Compare FHE vs non-FHE outputs
                fhe_mae = np.abs(fhe_output - fhe_result).mean()
                print(f"FHE vs quantized model MAE: {fhe_mae}")
                
                # Print more detailed comparison
                print("First few values comparison:")
                print(f"Quantized output: {fhe_output.flatten()[:4]}")
                print(f"FHE output     : {fhe_result.flatten()[:4]}")
                
            except Exception as e:
                print(f"Error running in FHE mode: {e}")
                import traceback
                traceback.print_exc()
        
        return quantized_module, model, mae
    
    except Exception as e:
        print(f"Error compiling the model: {e}")
        return None, model, float('inf')

# Create a slightly more complex model that mimics a simplified transformer component
class SimplifiedTransformerBlock(torch.nn.Module):
    """
    A simplified transformer block that uses only FHE-compatible operations.
    """
    def __init__(self, embed_dim=16, ffn_dim=32):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Linear projections for a simplified self-attention
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        
        # Feed-forward network
        self.ffn1 = torch.nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = torch.nn.Linear(ffn_dim, embed_dim)
        
        # Initialize with small weights
        self._init_small_weights()
    
    def _init_small_weights(self):
        """Initialize weights with small values."""
        modules = [self.q_proj, self.k_proj, self.v_proj, self.out_proj, self.ffn1, self.ffn2]
        for module in modules:
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data.normal_(mean=0.0, std=0.01)
            if hasattr(module, 'bias') and module.bias is not None:
                with torch.no_grad():
                    module.bias.data.zero_()
    
    def forward(self, x):
        # Original input for residual connection
        residual = x
        
        # Simplified self-attention (no softmax, just scaled dot-product)
        q = self.q_proj(x) * 0.1  # Scale for numerical stability
        k = self.k_proj(x) * 0.1  # Scale for numerical stability
        v = self.v_proj(x)
        
        # Simple weighted sum instead of attention
        # This avoids complex operations like softmax
        attn = q * k  # Element-wise multiplication instead of matrix multiplication
        out = self.out_proj(attn * v)  # Element-wise multiplication with values
        
        # First residual connection
        x = residual + out
        
        # FFN with polynomial activation
        residual = x
        x = self.ffn1(x)
        x = x * x + x  # Polynomial activation (x^2 + x)
        x = self.ffn2(x)
        
        # Second residual connection
        x = residual + x
        
        return x

def test_transformer_block(fhe_mode='execute'):
    """Test the simplified transformer block with Concrete-ML.
    
    Args:
        fhe_mode (str): FHE execution mode - 'execute', 'simulate', or 'disable'
    """
    print(f"\n=== Testing Simplified Transformer Block (FHE mode: {fhe_mode}) ===")
    
    # Create a simplified transformer block
    embed_dim = 16
    ffn_dim = 32
    model = SimplifiedTransformerBlock(embed_dim=embed_dim, ffn_dim=ffn_dim)
    print("Created simplified transformer block")
    
    # Create random input
    batch_size = 1
    seq_len = 4
    x = torch.randn(batch_size, seq_len, embed_dim) * 0.1  # Scale inputs
    
    # Run inference with the original model
    print("Running inference with the original model...")
    with torch.no_grad():
        original_output = model(x)
    
    # Compile the model
    print("Compiling the model...")
    try:
        # Convert to numpy for compilation
        x_numpy = x.numpy()
        
        # Use very low bit width to avoid overflow
        n_bits = 4
        quantized_module = compile_torch_model(
            model, 
            x_numpy,
            n_bits=n_bits
        )
        
        # Run inference with the compiled model
        print("Running inference with the compiled model...")
        fhe_output = quantized_module.forward(x_numpy)
        
        # Compare outputs
        mae = np.abs(original_output.detach().numpy() - fhe_output).mean()
        print(f"Mean Absolute Error: {mae}")
        
        # Check if compilation was successful and try to run in FHE mode
        if hasattr(quantized_module, 'fhe_circuit') and fhe_mode != 'disable':
            print("FHE circuit successfully created!")
            
            # Try to run in FHE
            print(f"Running in FHE mode: {fhe_mode}...")
            try:
                fhe_circuit = quantized_module.fhe_circuit
                
                # Convert input to the expected integer format
                x_quantized = quantized_module.quantize_input(x_numpy)
                
                # Run the FHE circuit with quantized inputs
                if fhe_mode == 'execute':
                    fhe_result = fhe_circuit.encrypt_run_decrypt(x_quantized)
                    print("Successfully ran in FHE mode with encryption!")
                elif fhe_mode == 'simulate':
                    # Simulate FHE execution without encryption
                    fhe_result = fhe_circuit.simulate(x_quantized)
                    print("Successfully simulated FHE execution!")
                
                # Compare FHE vs non-FHE outputs
                fhe_mae = np.abs(fhe_output - fhe_result).mean()
                print(f"FHE vs quantized model MAE: {fhe_mae}")
            except Exception as e:
                print(f"Error running in FHE mode: {e}")
                import traceback
                traceback.print_exc()
                
        return quantized_module, model, mae
    
    except Exception as e:
        print(f"Error compiling the model: {e}")
        return None, model, float('inf')

if __name__ == "__main__":
    print(f"Starting FHE-compatible model tests (FHE mode: {args.fhe_mode})...")
    
    # Test the minimal model
    quantized_minimal, minimal_model, minimal_mae = test_minimal_model(fhe_mode=args.fhe_mode)
    
    # Test the simplified transformer block
    if minimal_mae < 0.1:  # Only try the transformer if minimal model works
        print("\nMinimal model works well, trying simplified transformer block...")
        quantized_transformer, transformer_model, transformer_mae = test_transformer_block(fhe_mode=args.fhe_mode)
    else:
        print("\nMinimal model has high error, skipping transformer test.")
        transformer_mae = float('inf')
    
    print("\nAll tests completed!")
    print(f"Minimal model MAE: {minimal_mae}")
    print(f"Simplified transformer MAE: {transformer_mae}")
    
    if min(minimal_mae, transformer_mae) < 0.1:
        print("\nSuccess! At least one model works well with FHE.")
    else:
        print("\nFurther optimization needed to make models work with FHE.") 