"""
Improved Hybrid FHE model for nanoGPT using Concrete-ML

This script focuses on implementing the attention mechanism of nanoGPT
in FHE using Concrete-ML's capabilities.
"""

import os
import sys
import torch
import numpy as np
from nanoGPT_model import GPT, GPTConfig, MLP, Block, CausalSelfAttention

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
    from concrete.ml.torch.hybrid_model import HybridFHEModel
    print("Successfully imported Concrete-ML components")
except ImportError as e:
    print(f"Error importing Concrete-ML components: {e}")
    sys.exit(1)

# Create an extremely simplified attention mechanism for FHE
class SuperSimpleAttention(torch.nn.Module):
    """
    A super simplified attention mechanism that is FHE-friendly.
    Avoids complex reshaping and multi-head operations.
    """
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        
        # Single projection for query, key, value
        self.query = torch.nn.Linear(n_embd, n_embd, bias=False)
        self.key = torch.nn.Linear(n_embd, n_embd, bias=False)
        self.value = torch.nn.Linear(n_embd, n_embd, bias=False)
        
        # Output projection
        self.out_proj = torch.nn.Linear(n_embd, n_embd)
        
        # Create a fixed causal mask using basic operations
        self.register_buffer("mask", self._create_causal_mask(block_size))
        
        # Initialize with small weights to reduce bit width requirements
        self._init_small_weights()
        
    def _create_causal_mask(self, size):
        """Create a causal mask without using torch.triu"""
        # Create indices for rows and columns
        row_indices = torch.arange(size).unsqueeze(1).expand(size, size)
        col_indices = torch.arange(size).unsqueeze(0).expand(size, size)
        
        # Create mask: 1 where row_idx >= col_idx (lower triangular including diagonal)
        # and 0 elsewhere (upper triangular excluding diagonal)
        mask = (row_indices >= col_indices).float()
        
        return mask
        
    def _init_small_weights(self):
        """Initialize weights with small values to reduce bit width."""
        for module in [self.query, self.key, self.value, self.out_proj]:
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data.normal_(mean=0.0, std=0.01)
            if hasattr(module, 'bias') and module.bias is not None:
                with torch.no_grad():
                    module.bias.data.zero_()
    
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values (no multi-head splitting)
        q = self.query(x) * 0.1  # Scale for numerical stability
        k = self.key(x) * 0.1    # Scale for numerical stability
        v = self.value(x)
        
        # Compute attention scores - simplified for FHE
        # Simple dot product attention
        att = torch.bmm(q, k.transpose(1, 2))  # (B, T, T)
        
        # Apply causal mask - FHE-friendly version
        if T <= self.block_size:
            mask = self.mask[:T, :T]
            # Apply mask: multiply by mask (keep lower triangular) and set upper triangular to small negative value
            att = att * mask + (-1.0) * (1 - mask)
        else:
            # Handle case where T > block_size (shouldn't happen in normal usage)
            print(f"Warning: Sequence length {T} exceeds block size {self.block_size}")
            mask = self._create_causal_mask(T)
            att = att * mask + (-1.0) * (1 - mask)
        
        # Simple normalization instead of softmax
        # This is more FHE-friendly than exponential operations
        att = att / (torch.sum(att, dim=-1, keepdim=True) + 1e-5)
        
        # Apply attention to values
        y = torch.bmm(att, v)  # (B, T, C)
        
        # Output projection
        y = self.out_proj(y)
        
        return y

# Create a simple model with attention for FHE testing
class SimpleAttentionModel(torch.nn.Module):
    """A simple model with attention for FHE testing."""
    def __init__(self, n_embd=16, block_size=4):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(100, n_embd)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.attention = SuperSimpleAttention(n_embd, block_size)
        self.ln = torch.nn.LayerNorm(n_embd)
        self.head = torch.nn.Linear(n_embd, n_embd)
        
        # Initialize with small weights
        self._init_small_weights()
    
    def _init_small_weights(self):
        """Initialize weights with small values."""
        with torch.no_grad():
            self.token_embedding.weight.data.normal_(mean=0.0, std=0.01)
            self.pos_embedding.data.normal_(mean=0.0, std=0.01)
            self.head.weight.data.normal_(mean=0.0, std=0.01)
            if self.head.bias is not None:
                self.head.bias.data.zero_()
    
    def forward(self, idx):
        """Forward pass."""
        # Get token embeddings
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)  # Add batch dimension if needed
            
        # Token embeddings + positional embeddings
        t = idx.shape[1]
        token_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding[:, :t, :]
        x = token_emb + pos_emb
        
        # Apply attention
        attn_out = self.attention(x)
        
        # Layer norm and residual connection
        x = self.ln(x + attn_out)
        
        # Final projection
        x = self.head(x)
        
        return x

def test_attention_component():
    """Test just the attention component with Concrete-ML."""
    print("\n=== Testing Standalone Attention Component ===")
    
    # Create just the attention component
    block_size = 4
    n_embd = 16
    attention = SuperSimpleAttention(n_embd=n_embd, block_size=block_size)
    print("Created standalone attention component")
    
    # Create random input
    batch_size = 1
    x = torch.randn(batch_size, block_size, n_embd) * 0.1  # Scale inputs
    
    # Run inference with the original attention
    print("Running inference with the original attention...")
    with torch.no_grad():
        original_output = attention(x)
    
    # Compile the attention component
    print("Compiling the attention component...")
    try:
        # Convert to numpy for compilation
        x_numpy = x.numpy()
        
        # Use very low bit width to avoid overflow
        n_bits = 4
        quantized_module = compile_torch_model(
            attention, 
            x_numpy,
            n_bits=n_bits
        )
        
        # Run inference with the compiled attention
        print("Running inference with the compiled attention...")
        fhe_output = quantized_module.forward(x_numpy)
        
        # Compare outputs
        mae = np.abs(original_output.detach().numpy() - fhe_output).mean()
        print(f"Mean Absolute Error: {mae}")
        
        return quantized_module, attention, mae
    
    except Exception as e:
        print(f"Error compiling the attention component: {e}")
        return None, attention, float('inf')

def test_attention_model():
    """Test the attention model with Concrete-ML."""
    print("\n=== Testing Attention Model with Concrete-ML ===")
    
    # Create a simple model
    block_size = 4
    n_embd = 16
    model = SimpleAttentionModel(n_embd=n_embd, block_size=block_size)
    print("Created simple attention model")
    
    # Create random input - use token indices
    batch_size = 1
    x = torch.randint(0, 100, (batch_size, block_size))
    
    # Run inference with the original model
    print("Running inference with the original model...")
    with torch.no_grad():
        original_output = model(x)
    
    # Try to compile the entire model
    print("Attempting to compile the entire model...")
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
        
        print("Successfully compiled the entire model!")
        
        # Run inference with the compiled model
        print("Running inference with the compiled model...")
        fhe_output = quantized_module.forward(x_numpy)
        
        # Compare outputs
        mae = np.abs(original_output.detach().numpy() - fhe_output).mean()
        print(f"Mean Absolute Error: {mae}")
        
        return quantized_module, model, mae
    
    except Exception as e:
        print(f"Error compiling the entire model: {e}")
        print("Falling back to hybrid model approach...")
        
        # Try using HybridFHEModel to compile just the attention module
        try:
            print("Creating hybrid model with attention in FHE...")
            hybrid_model = HybridFHEModel(model, module_names=["attention"])
            
            print("Compiling the hybrid model...")
            hybrid_model.compile_model(x, n_bits=n_bits)
            
            # Run inference with the hybrid model
            print("Running inference with the hybrid model...")
            with torch.no_grad():
                hybrid_output = hybrid_model(x)
            
            # Compare outputs
            mae = torch.abs(original_output - hybrid_output).mean().item()
            print(f"Mean Absolute Error: {mae}")
            
            return hybrid_model, model, mae
            
        except Exception as e:
            print(f"Error creating/compiling hybrid model: {e}")
            print("Unable to compile the model for FHE.")
            return None, model, float('inf')

if __name__ == "__main__":
    print("Starting FHE-friendly attention tests...")
    
    # Test the attention component
    quantized_attention, attention_model, attention_mae = test_attention_component()
    
    # Test the full model with attention
    if attention_mae < 0.1:  # Only try the full model if attention works well
        print("\nAttention component works well, trying full model...")
        quantized_model, full_model, model_mae = test_attention_model()
    else:
        print("\nAttention component has high error, skipping full model test.")
        model_mae = float('inf')
    
    print("\nAll tests completed!")
    print(f"Attention component MAE: {attention_mae}")
    print(f"Full model MAE: {model_mae}")
    
    if min(attention_mae, model_mae) < 0.1:
        print("\nSuccess! At least one component works well with FHE.")
    else:
        print("\nFurther optimization needed to make the model work with FHE.") 