"""
Implementation of a Hybrid FHE model for nanoGPT using Concrete-ML

This script demonstrates how to use Concrete-ML's HybridFHEModel to convert
parts of a nanoGPT model to run in FHE.
"""

import os
import sys
import torch
import numpy as np
from nanoGPT_model import GPT, GPTConfig, MLP, Block

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

# Now try to import the HybridFHEModel
try:
    from concrete.ml.torch.hybrid_model import HybridFHEModel
    print("Successfully imported HybridFHEModel")
except ImportError as e:
    print(f"Error importing HybridFHEModel: {e}")
    sys.exit(1)

# Create a tiny GPT model for testing with FHE
def create_tiny_gpt():
    """Create a tiny GPT model for testing with FHE."""
    config = GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=1,
        n_head=2,
        n_embd=16,  # Very small embedding size to reduce bit width
        dropout=0.0,
        bias=True
    )
    model = GPT(config)
    
    # Initialize weights with small values to avoid bit width issues
    def init_small_weights(module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            # Initialize with small values to reduce bit width after multiplication
            with torch.no_grad():
                if hasattr(module, 'weight'):
                    module.weight.data.normal_(mean=0.0, std=0.01)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.zero_()
    
    model.apply(init_small_weights)
    return model

# Create a custom MLP with small weights for FHE
class TinyMLP(torch.nn.Module):
    def __init__(self, dim=16, hidden_dim=32):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, dim)
        
        # Initialize with small weights
        with torch.no_grad():
            self.fc1.weight.data.normal_(mean=0.0, std=0.01)
            self.fc1.bias.data.zero_()
            self.fc2.weight.data.normal_(mean=0.0, std=0.01)
            self.fc2.bias.data.zero_()
    
    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x

# Create a hybrid model that runs part of the network in FHE
def create_hybrid_fhe_gpt(model, n_bits=6):
    """
    Create a hybrid FHE model from a GPT model.
    
    Args:
        model: The GPT model to convert
        n_bits: Number of bits for quantization (reduced to avoid overflow)
        
    Returns:
        A HybridFHEModel that can run parts of the model in FHE
    """
    # Define which modules to run in FHE
    # We'll start with just the MLP in the first block
    module_names = ["transformer.h.0.mlp"]
    
    # Create a random input for compilation
    batch_size = 1
    seq_length = 4  # Very small sequence length for testing
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
    
    print(f"Creating hybrid model with modules: {module_names}")
    hybrid_model = HybridFHEModel(model, module_names=module_names)
    
    print("Compiling the hybrid model (this may take a few minutes)...")
    hybrid_model.compile_model(x, n_bits=n_bits)
    
    return hybrid_model

# Test the hybrid model
def test_hybrid_model(hybrid_model, model):
    """Test the hybrid model against the original model."""
    batch_size = 1
    seq_length = 4
    x = torch.randint(0, model.config.vocab_size, (batch_size, seq_length))
    
    print("Running inference with the original model...")
    with torch.no_grad():
        original_output = model(x)
    
    print("Running inference with the hybrid model...")
    with torch.no_grad():
        hybrid_output = hybrid_model(x)
    
    # Compare outputs
    if isinstance(original_output, tuple):
        original_logits = original_output[0]
    else:
        original_logits = original_output
        
    if isinstance(hybrid_output, tuple):
        hybrid_logits = hybrid_output[0]
    else:
        hybrid_logits = hybrid_output
    
    # Calculate the mean absolute error
    mae = torch.abs(original_logits - hybrid_logits).mean().item()
    print(f"Mean Absolute Error between original and hybrid model: {mae}")
    
    return mae

# Test the standalone TinyMLP with FHE
def test_tiny_mlp_fhe():
    """Test running a very small MLP in FHE."""
    # Create a tiny MLP
    mlp_model = TinyMLP(dim=16, hidden_dim=32)
    
    # Create random input for the MLP
    batch_size = 1
    seq_length = 4
    dim = 16
    x = torch.randn(batch_size, seq_length, dim) * 0.1  # Scale down inputs
    
    # Create hybrid model for the MLP
    print("Creating hybrid model for TinyMLP...")
    hybrid_mlp = HybridFHEModel(mlp_model, module_names=["fc1", "fc2"])
    
    print("Compiling the TinyMLP hybrid model...")
    hybrid_mlp.compile_model(x, n_bits=6)  # Use fewer bits
    
    # Test the models
    print("Running inference with the original TinyMLP...")
    with torch.no_grad():
        original_output = mlp_model(x)
    
    print("Running inference with the FHE TinyMLP...")
    with torch.no_grad():
        hybrid_output = hybrid_mlp(x)
    
    # Compare outputs
    mae = torch.abs(original_output - hybrid_output).mean().item()
    print(f"Mean Absolute Error between original and hybrid TinyMLP: {mae}")
    
    return hybrid_mlp, mae

# Run a complete test of the hybrid FHE model
def run_complete_test():
    """Run a complete test of the hybrid FHE model."""
    print("\n=== Testing TinyMLP with FHE ===")
    try:
        hybrid_mlp, mlp_mae = test_tiny_mlp_fhe()
        print("TinyMLP FHE test completed successfully!")
        
        print("\n=== Testing tiny GPT model with FHE ===")
        try:
            print("Creating tiny GPT model...")
            model = create_tiny_gpt()
            print(f"Model created with {model.get_num_params()} parameters")
            
            hybrid_model = create_hybrid_fhe_gpt(model)
            mae = test_hybrid_model(hybrid_model, model)
            print("\nFHE Hybrid GPT test completed successfully!")
            return hybrid_model, model, mae
        except Exception as e:
            print(f"\nError in hybrid GPT test: {e}")
            print("Falling back to TinyMLP model")
            return hybrid_mlp, None, mlp_mae
    except Exception as e:
        print(f"\nError in TinyMLP test: {e}")
        print("All FHE tests failed")
        return None, None, float('inf')

# Save the hybrid model for later use
def save_hybrid_model(hybrid_model, filename="hybrid_fhe_model.pt"):
    """Save the hybrid model to a file."""
    if hybrid_model is None:
        print("No hybrid model to save")
        return
        
    torch.save(hybrid_model.state_dict(), filename)
    print(f"Hybrid model saved to {filename}")

# Entry point
if __name__ == "__main__":
    print("Starting Hybrid FHE GPT test...")
    hybrid_model, original_model, mae = run_complete_test()
    
    # Save the models
    save_hybrid_model(hybrid_model)
    
    print("\nTest complete!")
    print(f"Final Mean Absolute Error: {mae}") 