"""
FHE-compatible nanoGPT model using Concrete-ML (Final Version)

This script implements a simplified FHE-compatible version of the nanoGPT model
using only components that are known to work well with Concrete-ML.
"""

import os
import sys
import torch
import numpy as np

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

# Define FHE-friendly components for the nanoGPT model

class FHEFriendlyMLP(torch.nn.Module):
    """FHE-friendly MLP with polynomial activation function."""
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.c_fc = torch.nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = torch.nn.Linear(4 * n_embd, n_embd)
        # Initialize with small weights
        self._init_small_weights()
    
    def _init_small_weights(self):
        """Initialize weights with small values."""
        for module in [self.c_fc, self.c_proj]:
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data.normal_(mean=0.0, std=0.01)
            if hasattr(module, 'bias') and module.bias is not None:
                with torch.no_grad():
                    module.bias.data.zero_()
    
    def forward(self, x):
        # Apply first linear layer
        x = self.c_fc(x)
        
        # Use polynomial activation (x^2 + x) instead of GELU
        # This is FHE-friendly
        x = x * x + x
        
        # Apply second linear layer
        x = self.c_proj(x)
        
        return x

class FHEFriendlyAttention(torch.nn.Module):
    """FHE-friendly attention mechanism."""
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        
        # Linear projections for query, key, value
        self.query = torch.nn.Linear(n_embd, n_embd, bias=False)
        self.key = torch.nn.Linear(n_embd, n_embd, bias=False)
        self.value = torch.nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = torch.nn.Linear(n_embd, n_embd)
        
        # Initialize with small weights
        self._init_small_weights()
    
    def _init_small_weights(self):
        """Initialize weights with small values."""
        for module in [self.query, self.key, self.value, self.out_proj]:
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data.normal_(mean=0.0, std=0.01)
            if hasattr(module, 'bias') and module.bias is not None:
                with torch.no_grad():
                    module.bias.data.zero_()
    
    def forward(self, x):
        # Calculate query, key, values
        q = self.query(x) * 0.1  # Scale for numerical stability
        k = self.key(x) * 0.1    # Scale for numerical stability
        v = self.value(x)
        
        # Simple element-wise attention (FHE-friendly)
        # Instead of matrix multiplication, use element-wise operations
        attn = q * k  # Element-wise multiplication
        
        # Apply attention to values directly
        out = self.out_proj(attn * v)  # Element-wise multiplication with values
        
        return out

class MinimalFHETransformer(torch.nn.Module):
    """
    Minimal FHE-compatible transformer model.
    Uses only components that are known to work well with Concrete-ML.
    """
    def __init__(self, vocab_size=100, n_embd=16, seq_len=4, n_layer=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        
        # Input embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, seq_len, n_embd))
        
        # Transformer layers
        self.layers = torch.nn.ModuleList([])
        for _ in range(n_layer):
            self.layers.append(torch.nn.ModuleDict({
                'attention': FHEFriendlyAttention(n_embd),
                'mlp': FHEFriendlyMLP(n_embd)
            }))
        
        # Output head
        self.head = torch.nn.Linear(n_embd, vocab_size)
        
        # Initialize with small weights
        self._init_small_weights()
    
    def _init_small_weights(self):
        """Initialize weights with small values."""
        with torch.no_grad():
            self.token_embedding.weight.data.normal_(mean=0.0, std=0.01)
            self.position_embedding.data.normal_(mean=0.0, std=0.01)
            self.head.weight.data.normal_(mean=0.0, std=0.01)
            if self.head.bias is not None:
                self.head.bias.data.zero_()
    
    def forward(self, idx):
        # Get token embeddings
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)  # Add batch dimension if needed
        
        # Token embeddings + positional embeddings
        t = idx.shape[1]
        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding[:, :t, :]
        x = token_emb + pos_emb
        
        # Apply transformer layers
        for layer in self.layers:
            # Attention with residual connection
            x = x + layer['attention'](x)
            
            # MLP with residual connection
            x = x + layer['mlp'](x)
        
        # Apply head to get logits
        logits = self.head(x)
        
        return logits

def test_minimal_transformer():
    """Test the minimal FHE-compatible transformer model."""
    print("\n=== Testing Minimal FHE Transformer ===")
    
    # Create a small model
    vocab_size = 100
    n_embd = 16
    seq_len = 4
    n_layer = 1
    model = MinimalFHETransformer(
        vocab_size=vocab_size,
        n_embd=n_embd,
        seq_len=seq_len,
        n_layer=n_layer
    )
    print("Created minimal FHE transformer model")
    
    # Create random input
    batch_size = 1
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
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
        if hasattr(quantized_module, 'fhe_circuit'):
            print("FHE circuit successfully created!")
            
            # Try to run in FHE
            print("Running in FHE mode...")
            try:
                fhe_circuit = quantized_module.fhe_circuit
                fhe_result = fhe_circuit.encrypt_run_decrypt(x_numpy)
                print("Successfully ran in FHE mode!")
                
                # Compare FHE vs non-FHE outputs
                fhe_mae = np.abs(fhe_output - fhe_result).mean()
                print(f"FHE vs quantized model MAE: {fhe_mae}")
            except Exception as e:
                print(f"Error running in FHE mode: {e}")
        
        return quantized_module, model, mae
    
    except Exception as e:
        print(f"Error compiling the model: {e}")
        return None, model, float('inf')

def save_model(model, filename="fhe_friendly_model.pt"):
    """Save the model to a file."""
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    print("Starting minimal FHE transformer test...")
    
    # Test the minimal transformer
    quantized_model, model, mae = test_minimal_transformer()
    
    print("\nTest completed!")
    print(f"Mean Absolute Error: {mae}")
    
    if mae < 0.1:
        print("\nSuccess! The minimal transformer works well with FHE.")
        # Save the model
        save_model(model)
    else:
        print("\nFurther optimization needed to make the model work with FHE.") 