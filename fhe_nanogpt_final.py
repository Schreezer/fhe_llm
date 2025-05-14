"""
FHE-compatible nanoGPT model using Concrete-ML

This script implements an FHE-compatible version of the nanoGPT model
using the techniques we've developed with Concrete-ML.
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
        x = x * x + x
        
        # Apply second linear layer
        x = self.c_proj(x)
        
        return x

class FHEFriendlyAttention(torch.nn.Module):
    """FHE-friendly attention mechanism."""
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.block_size = block_size
        
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
        B, T, C = x.size()
        
        # Calculate query, key, values
        q = self.query(x) * 0.1  # Scale for numerical stability
        k = self.key(x) * 0.1    # Scale for numerical stability
        v = self.value(x)
        
        # Simple element-wise attention (FHE-friendly)
        # Instead of matrix multiplication, use element-wise operations
        attn = q * k  # Element-wise multiplication
        
        # Apply attention to values directly (no causal mask)
        # This is a simplification to make it FHE-friendly
        out = self.out_proj(attn * v)  # Element-wise multiplication with values
        
        return out

class FHEFriendlyLayerNorm(torch.nn.Module):
    """FHE-friendly layer normalization."""
    def __init__(self, n_embd, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(n_embd))
        self.bias = torch.nn.Parameter(torch.zeros(n_embd))
    
    def forward(self, x):
        # Simple centering without standard deviation normalization
        # This is more FHE-friendly than full LayerNorm
        mean = torch.mean(x, dim=-1, keepdim=True)
        x = x - mean  # Center the data
        
        # Scale and shift
        x = x * self.weight + self.bias
        
        return x

class FHEFriendlyBlock(torch.nn.Module):
    """FHE-friendly transformer block."""
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.ln_1 = FHEFriendlyLayerNorm(n_embd)
        self.attn = FHEFriendlyAttention(n_embd, block_size)
        self.ln_2 = FHEFriendlyLayerNorm(n_embd)
        self.mlp = FHEFriendlyMLP(n_embd)
    
    def forward(self, x):
        # First attention block with residual connection
        x = x + self.attn(self.ln_1(x))
        
        # MLP block with residual connection
        x = x + self.mlp(self.ln_2(x))
        
        return x

class FHEFriendlyGPT(torch.nn.Module):
    """FHE-friendly GPT model."""
    def __init__(self, vocab_size=100, n_embd=16, block_size=8, n_layer=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        
        # Input embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, n_embd)
        self.position_embedding = torch.nn.Parameter(torch.zeros(1, block_size, n_embd))
        
        # Transformer blocks
        self.blocks = torch.nn.ModuleList([
            FHEFriendlyBlock(n_embd, block_size) for _ in range(n_layer)
        ])
        
        # Final layer norm
        self.ln_f = FHEFriendlyLayerNorm(n_embd)
        
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
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final layer norm
        x = self.ln_f(x)
        
        # Apply head to get logits
        logits = self.head(x)
        
        return logits

def test_fhe_friendly_gpt():
    """Test the FHE-friendly GPT model with Concrete-ML."""
    print("\n=== Testing FHE-friendly GPT Model ===")
    
    # Create a small model
    vocab_size = 100
    n_embd = 16
    block_size = 4
    n_layer = 1
    model = FHEFriendlyGPT(
        vocab_size=vocab_size,
        n_embd=n_embd,
        block_size=block_size,
        n_layer=n_layer
    )
    print("Created FHE-friendly GPT model")
    
    # Create random input
    batch_size = 1
    seq_len = block_size
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
        
        return quantized_module, model, mae
    
    except Exception as e:
        print(f"Error compiling the model: {e}")
        return None, model, float('inf')

def test_fhe_friendly_components():
    """Test individual FHE-friendly components."""
    print("\n=== Testing FHE-friendly Components ===")
    
    # Test parameters
    n_embd = 16
    block_size = 4
    
    # Test MLP
    print("\nTesting FHE-friendly MLP...")
    mlp = FHEFriendlyMLP(n_embd)
    x_mlp = torch.randn(1, 4, n_embd) * 0.1
    
    with torch.no_grad():
        mlp_out = mlp(x_mlp)
    
    try:
        quantized_mlp = compile_torch_model(mlp, x_mlp.numpy(), n_bits=4)
        mlp_fhe_out = quantized_mlp.forward(x_mlp.numpy())
        mlp_mae = np.abs(mlp_out.numpy() - mlp_fhe_out).mean()
        print(f"MLP MAE: {mlp_mae}")
    except Exception as e:
        print(f"Error compiling MLP: {e}")
        mlp_mae = float('inf')
    
    # Test Attention
    print("\nTesting FHE-friendly Attention...")
    attn = FHEFriendlyAttention(n_embd, block_size)
    x_attn = torch.randn(1, 4, n_embd) * 0.1
    
    with torch.no_grad():
        attn_out = attn(x_attn)
    
    try:
        quantized_attn = compile_torch_model(attn, x_attn.numpy(), n_bits=4)
        attn_fhe_out = quantized_attn.forward(x_attn.numpy())
        attn_mae = np.abs(attn_out.numpy() - attn_fhe_out).mean()
        print(f"Attention MAE: {attn_mae}")
    except Exception as e:
        print(f"Error compiling Attention: {e}")
        attn_mae = float('inf')
    
    # Test LayerNorm
    print("\nTesting FHE-friendly LayerNorm...")
    ln = FHEFriendlyLayerNorm(n_embd)
    x_ln = torch.randn(1, 4, n_embd) * 0.1
    
    with torch.no_grad():
        ln_out = ln(x_ln)
    
    try:
        quantized_ln = compile_torch_model(ln, x_ln.numpy(), n_bits=4)
        ln_fhe_out = quantized_ln.forward(x_ln.numpy())
        ln_mae = np.abs(ln_out.numpy() - ln_fhe_out).mean()
        print(f"LayerNorm MAE: {ln_mae}")
    except Exception as e:
        print(f"Error compiling LayerNorm: {e}")
        ln_mae = float('inf')
    
    # Test Block
    print("\nTesting FHE-friendly Block...")
    block = FHEFriendlyBlock(n_embd, block_size)
    x_block = torch.randn(1, 4, n_embd) * 0.1
    
    with torch.no_grad():
        block_out = block(x_block)
    
    try:
        quantized_block = compile_torch_model(block, x_block.numpy(), n_bits=4)
        block_fhe_out = quantized_block.forward(x_block.numpy())
        block_mae = np.abs(block_out.numpy() - block_fhe_out).mean()
        print(f"Block MAE: {block_mae}")
    except Exception as e:
        print(f"Error compiling Block: {e}")
        block_mae = float('inf')
    
    return {
        'mlp': mlp_mae,
        'attention': attn_mae,
        'layernorm': ln_mae,
        'block': block_mae
    }

if __name__ == "__main__":
    print("Starting FHE-friendly GPT tests...")
    
    # Test individual components first
    component_results = test_fhe_friendly_components()
    
    # Test the full model if components work well
    if min(component_results.values()) < 0.1:
        print("\nSome components work well, trying the full model...")
        quantized_model, model, model_mae = test_fhe_friendly_gpt()
    else:
        print("\nAll components have high error, skipping full model test.")
        model_mae = float('inf')
    
    print("\nAll tests completed!")
    print("Component MAEs:")
    for component, mae in component_results.items():
        print(f"  {component}: {mae}")
    print(f"Full model MAE: {model_mae}")
    
    if model_mae < 0.1 or min(component_results.values()) < 0.1:
        print("\nSuccess! At least one component works well with FHE.")
    else:
        print("\nFurther optimization needed to make the model work with FHE.") 