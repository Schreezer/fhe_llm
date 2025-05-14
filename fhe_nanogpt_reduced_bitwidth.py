"""
FHE-compatible nanoGPT with explicit bit-width control for FHE

This version implements enhanced bit-width controls:
1. Explicit scaling between blocks to prevent bit-width growth
2. More aggressive quantization (low std dev for weights)
3. Modified residual connections to avoid bit-width accumulation
4. Smaller dimensions for better FHE compatibility
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from dataclasses import dataclass

# Import Concrete-ML
site_packages = '/Users/chirag13/development/ai_project/fhe_llama_env_py310/lib/python3.10/site-packages'
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

try:
    from concrete.ml.torch.compile import compile_torch_model
except ImportError:
    print("Running setup_concrete_ml.py first...")
    exec(open("setup_concrete_ml.py").read())
    from concrete.ml.torch.compile import compile_torch_model

# FHE-friendly activation function
class FHEActivation(nn.Module):
    """FHE-friendly polynomial activation (x² + x) with scaling"""
    def forward(self, x):
        # Scale down activation to keep bit width small
        return x * x * 0.1 + x * 0.1

# FHE-friendly layer normalization (simplified)
class FHELayerNorm(nn.Module):
    """FHE-friendly layer normalization (simple scaling)"""
    def __init__(self, n_embd):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        
    def forward(self, x):
        # Simple scaling without normalization
        # Scale by small factor to reduce bit width
        return x * self.weight * 0.1

# FHE-friendly attention with bit-width control
class FHEAttention(nn.Module):
    """FHE-friendly attention with explicit bit-width reduction"""
    def __init__(self, config):
        super().__init__()
        # Single head attention (simpler)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        
        # Very small weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with very small weights for better FHE compatibility"""
        for module in [self.query, self.key, self.value, self.out_proj]:
            nn.init.normal_(module.weight, mean=0.0, std=0.005)  # Even smaller std
    
    def forward(self, x):
        # Project queries, keys, values with aggressive scaling
        q = self.query(x) * 0.01  # Extra aggressive scaling
        k = self.key(x) * 0.01
        v = self.value(x) * 0.1
        
        # Compute attention scores with bit-width control
        attn_weights = torch.bmm(q, k.transpose(1, 2)) * 0.01

        # Apply polynomial activation instead of softmax
        attn_weights = attn_weights * attn_weights * 0.1 + attn_weights * 0.1
        
        # Apply attention
        attn_output = torch.bmm(attn_weights, v) * 0.01
        
        # Output projection with scaling
        output = self.out_proj(attn_output) * 0.1
        
        return output

# FHE-friendly MLP
class FHEMLP(nn.Module):
    """FHE-friendly MLP with bit-width control"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 2, bias=False)
        self.activation = FHEActivation()
        self.fc2 = nn.Linear(config.n_embd * 2, config.n_embd, bias=False)
        
        # Very small weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with very small weights for better FHE compatibility"""
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.005)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.005)
    
    def forward(self, x):
        # Apply fc1 with scaling
        x = self.fc1(x) * 0.05
        # Apply activation
        x = self.activation(x)
        # Apply fc2 with scaling
        x = self.fc2(x) * 0.05
        return x

# FHE-friendly transformer block with bit-width control
class FHEBlock(nn.Module):
    """FHE-friendly transformer block with explicit bit-width reduction"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = FHELayerNorm(config.n_embd)
        self.attn = FHEAttention(config)
        self.ln_2 = FHELayerNorm(config.n_embd)
        self.mlp = FHEMLP(config)
    
    def forward(self, x):
        # First sub-block: attention with controlled residual
        attn_output = self.attn(self.ln_1(x))
        # Modified residual: average instead of add
        # This prevents bit width growth while preserving information flow
        x = x * 0.5 + attn_output * 0.5
        
        # Explicit bit-width reduction between operations
        x = x * 0.1
        
        # Second sub-block: MLP with controlled residual
        mlp_output = self.mlp(self.ln_2(x))
        # Modified residual: average instead of add
        x = x * 0.5 + mlp_output * 0.5
        
        # Final bit-width reduction
        x = x * 0.1
        
        return x

# Configuration class
@dataclass
class FHEGPTConfig:
    block_size: int = 32      # Small context window
    vocab_size: int = 256     # Small vocabulary
    n_layer: int = 1          # Single layer for FHE compatibility
    n_embd: int = 16          # Very small embedding size for FHE

# Main FHE-GPT model with bit-width control
class FHEReducedBitGPT(nn.Module):
    """
    FHE-compatible GPT with explicit bit-width reduction between components
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks (only 1-2 for FHE)
        self.blocks = nn.ModuleList([FHEBlock(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = FHELayerNorm(config.n_embd)
        
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"FHE-GPT parameters: {self.get_num_params() / 1e6:.2f}M")
    
    def _init_weights(self, module):
        """Initialize weights with very small values for FHE compatibility"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.005)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.005)
    
    def get_num_params(self):
        """Return number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, idx):
        """Forward pass with token indices as input"""
        device = idx.device
        b, t = idx.size()
        
        # Check sequence length
        if t > self.config.block_size:
            raise ValueError(f"Cannot forward sequence length {t}, max is {self.config.block_size}")
        
        # Get position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Token and position embeddings with scaling
        token_emb = self.wte(idx) * 0.01  # Scale down embeddings
        pos_emb = self.wpe(pos) * 0.01    # Scale down embeddings
        
        # Add embeddings with scaling
        x = token_emb + pos_emb
        
        # Apply transformer blocks with bit-width control between each
        for i, block in enumerate(self.blocks):
            x = block(x)
            # Add explicit bit-width reduction between blocks
            if i < len(self.blocks) - 1:
                x = x * 0.1  # Additional scaling between blocks
        
        # Apply final layer normalization
        x = self.ln_f(x)
        
        # Apply language model head with scaling
        logits = self.lm_head(x) * 0.5
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Generate tokens using greedy decoding"""
        for _ in range(max_new_tokens):
            # Crop if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits = self(idx_cond)
            
            # Get predictions for the last token
            logits = logits[:, -1, :]
            
            # Greedy selection (argmax)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# Test the model with FHE
def test_fhe_model():
    """Test the FHE-compatible model with reduced bit-width"""
    print("\n=== Testing FHE-compatible reduced bit-width GPT ===")
    
    # Create a larger configuration
    config = FHEGPTConfig(
        block_size=32,    # Double context window
        vocab_size=256,   # Keep vocabulary same
        n_layer=3,        # Three layers
        n_embd=64         # Double embedding size again
    )
    
    # Create the model
    model = FHEReducedBitGPT(config)
    
    # Create a small input (but slightly longer sequence)
    batch_size = 1
    seq_len = 16  # Longer sequence
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Run inference with the original model
    print("Running inference with original model...")
    with torch.no_grad():
        original_output = model(x)
    
    # Prepare for FHE compilation
    x_np = x.numpy()
    
    # Compile the model with low bit width
    print("Compiling model for FHE...")
    try:
        # Keep very small bit width (3) for better FHE compatibility
        quantized_module = compile_torch_model(
            model,
            x_np,
            n_bits=3
        )
        
        # Run with quantized model (no FHE)
        print("Running with quantized model (no FHE)...")
        quantized_output = quantized_module.forward(x_np, fhe="disable")
        
        # Compare outputs
        original_np = original_output.detach().numpy()
        mae = np.abs(original_np - quantized_output).mean()
        print(f"Mean Absolute Error (original vs quantized): {mae}")
        
        # Try FHE simulation
        print("\nTrying FHE simulation...")
        try:
            # Since we need one-hot encoding for the FHE circuit input,
            # we'll prepare that
            one_hot = np.zeros((1, seq_len, config.vocab_size), dtype=np.int64)
            for i in range(seq_len):
                one_hot[0, i, x_np[0, i]] = 1
                
            print(f"Created one-hot encoded input with shape {one_hot.shape}")
            
            # Run simulation
            fhe_output = quantized_module.fhe_circuit.simulate(one_hot)
            print("Successfully simulated FHE execution!")
            
            # Compare FHE vs quantized outputs
            fhe_mae = np.abs(quantized_output - fhe_output).mean()
            print(f"FHE vs quantized MAE: {fhe_mae}")
            
            # Try with actual encryption
            print("\nTrying FHE with encryption...")
            encrypted_output = quantized_module.fhe_circuit.encrypt_run_decrypt(one_hot)
            print("Successfully ran with encryption!")
            
            # Compare encrypted vs simulated
            enc_mae = np.abs(fhe_output - encrypted_output).mean()
            print(f"Encrypted vs simulated MAE: {enc_mae}")
            
            return True
        
        except Exception as e:
            print(f"Error running in FHE mode: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"Error compiling the model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the model
    success = test_fhe_model()
    
    if success:
        print("\nSuccess! The reduced bit-width model works with FHE.")
    else:
        print("\nThe model still needs adjustments to work with FHE.") 