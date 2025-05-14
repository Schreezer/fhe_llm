"""
FHE-compatible model for Shakespeare text generation

This implements a minimal transformer that is compatible with FHE constraints:
- No biases
- Integer-compatible operations
- Fixed bit-width operations
- Simple activation functions
- Training with floating point, inference with integers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from dataclasses import dataclass

# Config class for model parameters
@dataclass
class FHECompatibleConfig:
    block_size: int = 32      # Context window size
    vocab_size: int = 65      # Shakespeare has ~65 unique characters
    n_embd: int = 32          # Embedding dimension (reduced from 64)
    n_head: int = 2           # Number of attention heads (reduced from 4)
    n_layer: int = 2          # Number of transformer layers (reduced from 3)
    dropout: float = 0.0      # No dropout in FHE
    scale_factor: int = 100   # Scale factor for integer quantization
    int_scale_attn: int = 8   # Scale factor for attention (div by 2^8 = 256)

class SimpleActivation(nn.Module):
    """Simple activation function for FHE compatibility"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # Simple ReLU for compatibility
        return torch.clamp(x, min=0)

class FHECompatibleModel(nn.Module):
    """
    FHE-compatible transformer for Shakespeare generation
    Key differences from standard model:
    1. No biases in linear layers (FHE constraint)
    2. Simple activation functions 
    3. Integer-compatible operations during inference
    4. Floating point during training for gradient flow
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        
        # Final linear layer - no bias for FHE
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embedding and head weights
        self.tok_emb.weight = self.lm_head.weight
        
        # FHE mode flags
        self.use_integer_ops = False  # Default to floating point for training
        
        # Stats
        print(f"FHE-compatible model parameters: {sum(p.numel() for p in self.parameters())/1e6:.2f}M")
    
    def _init_weights(self, module):
        """Initialize with small weights for better quantization"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            # Small jitter to break symmetry
            with torch.no_grad():
                noise = torch.randn_like(module.weight) * 0.01
                module.weight.add_(noise)
        
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def set_fhe_mode(self, enable=True):
        """Set the model to use integer operations for FHE compatibility"""
        self.use_integer_ops = enable
        # Set all blocks to use integer ops
        for block in self.blocks:
            if hasattr(block, 'set_fhe_mode'):
                block.set_fhe_mode(enable)
    
    def forward(self, idx):
        """Forward pass with diagnostic info"""
        b, t = idx.size()
        
        # Get token and position embeddings
        tok_emb = self.tok_emb(idx)  # (b, t, n_embd)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, t)
        pos_emb = self.pos_emb(pos)  # (1, t, n_embd)
        
        # Combine embeddings
        x = tok_emb + pos_emb  # (b, t, n_embd)
        
        # Debug stats
        if not self.training and torch.rand(1)[0] < 0.01:
            print(f"Input embedding stats: min={x.min().item():.4f}, max={x.max().item():.4f}")
        
        # If using integer ops, quantize embeddings
        if self.use_integer_ops:
            x = (x * self.config.scale_factor).round() / self.config.scale_factor
        
        # Pass through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Debug stats
            if not self.training and torch.rand(1)[0] < 0.01:
                print(f"Block {i} output stats: min={x.min().item():.4f}, max={x.max().item():.4f}")
        
        # Final projection to vocabulary
        logits = self.lm_head(x)  # (b, t, vocab_size)
        
        # Debug stats
        if not self.training and torch.rand(1)[0] < 0.01:
            print(f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
        
        return logits
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate text with temperature sampling"""
        # Set to evaluation mode
        self.eval()
        
        # Set to integer mode for inference
        was_integer = self.use_integer_ops
        self.set_fhe_mode(True)
        
        # Truncate idx if needed
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        
        # Generate tokens sequentially
        for _ in range(max_new_tokens):
            # Get predictions
            with torch.no_grad():
                logits = self(idx_cond)  # (b, t, vocab_size)
                
                # Focus on the final token
                logits = logits[:, -1, :]  # (b, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Log top tokens
                if torch.rand(1)[0] < 0.1:  # Only log occasionally
                    top_probs, top_idx = torch.topk(probs, k=5)
                    print("Top 5 predicted tokens:")
                    for i, (p, ix) in enumerate(zip(top_probs[0], top_idx[0])):
                        print(f"  {i+1}. Token {ix.item()} with prob {p.item():.4f}")
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)  # (b, 1)
            
            # Append to the sequence
            idx_cond = torch.cat((idx_cond, idx_next), dim=1)  # (b, t+1)
            
            # Truncate if needed for next iteration
            if idx_cond.size(1) > self.config.block_size:
                idx_cond = idx_cond[:, -self.config.block_size:]
        
        # Restore integer mode setting
        self.set_fhe_mode(was_integer)
        
        return idx_cond

class SimpleAttention(nn.Module):
    """Simple attention mechanism for FHE compatibility"""
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.config = config
        assert self.n_embd % self.n_head == 0
        
        self.head_size = self.n_embd // self.n_head
        
        # Key, Query, Value projections (no bias)
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # Output projection
        self.proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
        # Attention mask for causal attention
        mask = torch.triu(torch.ones(config.block_size, config.block_size), diagonal=1)
        self.register_buffer('mask', mask.bool())
        
        # Set to use floating point by default
        self.use_integer_ops = False
    
    def set_fhe_mode(self, enable=True):
        """Set to use integer operations for FHE compatibility"""
        self.use_integer_ops = enable
    
    def forward(self, x):
        b, t, c = x.size()
        
        # Project to key, query, value
        k = self.key(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)  # (b, nh, t, hs)
        q = self.query(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)  # (b, nh, t, hs)
        v = self.value(x).view(b, t, self.n_head, self.head_size).transpose(1, 2)  # (b, nh, t, hs)
        
        # Compute attention scores
        # (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        att = (q @ k.transpose(-2, -1))
        
        # Scale attention scores
        if self.use_integer_ops:
            # Integer-friendly scaling
            scale_factor = 2 ** self.config.int_scale_attn
            att = att / scale_factor
            # Quantize to lower precision
            att = (att * self.config.scale_factor).round() / self.config.scale_factor
        else:
            # Standard scaling during training
            att = att / (self.head_size ** 0.5)
        
        # Apply causal mask - zero out future positions
        mask = self.mask[:t, :t]
        att = att.masked_fill(mask, 0)
        
        # For FHE compatibility, use a simpler normalization
        if self.use_integer_ops:
            # Simple normalization for integer ops
            att = torch.clamp(att, max=1.0)
        else:
            # Use softmax during training
            att = F.softmax(att, dim=-1)
        
        # Apply attention to values
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        
        # Project to output
        y = self.proj(y)
        
        # For integer ops, quantize output
        if self.use_integer_ops:
            y = (y * self.config.scale_factor).round() / self.config.scale_factor
        
        return y

class SimpleMLP(nn.Module):
    """Simple MLP for FHE compatibility"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.n_embd, config.n_embd * 2, bias=False)
        self.activation = SimpleActivation()
        self.fc2 = nn.Linear(config.n_embd * 2, config.n_embd, bias=False)
        self.use_integer_ops = False
    
    def set_fhe_mode(self, enable=True):
        """Set to use integer operations for FHE compatibility"""
        self.use_integer_ops = enable
    
    def forward(self, x):
        # First linear layer
        x = self.fc1(x)
        
        # Activation
        x = self.activation(x)
        
        # For integer ops, quantize
        if self.use_integer_ops:
            x = (x * self.config.scale_factor).round() / self.config.scale_factor
        
        # Second linear layer
        x = self.fc2(x)
        
        # For integer ops, scale down
        if self.use_integer_ops:
            # Quantize to lower precision
            x = x * 0.25  # Scale down to prevent overflow
            x = (x * self.config.scale_factor).round() / self.config.scale_factor
        
        return x

class TransformerBlock(nn.Module):
    """Transformer block for FHE compatibility"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn = SimpleAttention(config)
        self.mlp = SimpleMLP(config)
        self.use_integer_ops = False
    
    def set_fhe_mode(self, enable=True):
        """Set to use integer operations for FHE compatibility"""
        self.use_integer_ops = enable
        self.attn.set_fhe_mode(enable)
        self.mlp.set_fhe_mode(enable)
    
    def forward(self, x):
        # First sub-block: attention with residual
        attn_output = self.attn(x)
        x = x + attn_output
        
        # For integer ops, scale down after residual
        if self.use_integer_ops:
            x = x * 0.5  # Scale down to prevent overflow
            x = (x * self.config.scale_factor).round() / self.config.scale_factor
        
        # Second sub-block: MLP with residual
        mlp_output = self.mlp(x)
        x = x + mlp_output
        
        # For integer ops, scale down again
        if self.use_integer_ops:
            x = x * 0.5  # Scale down again
            x = (x * self.config.scale_factor).round() / self.config.scale_factor
        
        return x

def load_shakespeare_data():
    """Load Shakespeare dataset for training/generation"""
    # Check if the data exists
    path = 'nanoGPT/data/shakespeare/input.txt'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shakespeare data not found at {path}")
    
    # Read the data
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Shakespeare data loaded: {len(text)} characters")
    
    # Get unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Create encode/decode functions
    def encode(s):
        return [stoi[c] for c in s]
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    # Prepare dataset
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # Split into train/val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, encode, decode, vocab_size

def get_batch(data, block_size, batch_size):
    """Create a random batch of data"""
    # Random starting indices
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Extract sequences
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x, y

def train_model(model, train_data, val_data, block_size=32, batch_size=16, 
                learning_rate=3e-3, max_iters=2000, eval_interval=200, device='cpu'):
    """Train the FHE-compatible model"""
    # Set device
    model = model.to(device)
    
    # Ensure model is in training mode with floating point
    model.train()
    model.set_fhe_mode(False)  # Use floating point during training
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for iter in range(max_iters + 1):
        # Evaluate on validation set
        if iter % eval_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(10):  # Average over 10 batches
                    xb, yb = get_batch(val_data, block_size, batch_size)
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
                    val_loss += loss.item()
                val_loss /= 10
            print(f"Iteration {iter}: Validation loss: {val_loss:.4f}")
            model.train()
        
        # Get batch
        xb, yb = get_batch(train_data, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward, backward, optimize
        optimizer.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        loss.backward()
        optimizer.step()
        
        # Report training loss
        if iter % 100 == 0:
            print(f"Iteration {iter}: Training loss: {loss.item():.4f}")
        
        # Periodically quantize weights for better FHE compatibility
        if iter % 500 == 0:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        # Quantize to lower precision by rounding
                        scale = 100
                        param.mul_(scale).round_().div_(scale)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, 'fhe_shakespeare_model.pt')
    
    print("Training complete. Model saved to fhe_shakespeare_model.pt")
    
    return model

def load_model(model_path, device='cpu'):
    """Load a trained model"""
    data = torch.load(model_path, map_location=device)
    config = data['config']
    model = FHECompatibleModel(config)
    model.load_state_dict(data['model_state_dict'])
    model = model.to(device)
    return model, config

def generate_text(model, encode, decode, prompt="King ", max_tokens=100, device='cpu'):
    """Generate text from the model"""
    model = model.to(device)
    model.eval()
    
    # Encode the prompt
    input_ids = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate text
    model.set_fhe_mode(True)  # Use integer ops for inference
    output = model.generate(input_ids, max_tokens, temperature=0.8)
    
    # Decode
    generated_text = decode(output[0].tolist())
    
    return generated_text

def test_int_conversion(model):
    """Test integer conversion capabilities"""
    print("\n=== Testing Integer Conversion ===")
    
    # Create a simple input
    x = torch.tensor([[0, 1, 2]]).to(next(model.parameters()).device)
    
    # Test with floating point
    model.eval()
    model.set_fhe_mode(False)
    with torch.no_grad():
        fp_output = model(x)
        print(f"Floating point output stats: min={fp_output.min().item():.6f}, max={fp_output.max().item():.6f}")
    
    # Test with integer ops
    model.set_fhe_mode(True)
    with torch.no_grad():
        int_output = model(x)
        print(f"Integer ops output stats: min={int_output.min().item():.6f}, max={int_output.max().item():.6f}")
    
    # Compare the two
    diff = torch.abs(fp_output - int_output).mean()
    print(f"Mean absolute difference: {diff.item():.6f}")
    
    return diff.item() < 0.1  # Check if reasonably close

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='FHE-compatible model for Shakespeare')
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--generate', type=str, default="", help='Generate text from a prompt')
    parser.add_argument('--test-int', action='store_true', help='Test integer conversion')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU usage (avoid MPS/CUDA)')
    args = parser.parse_args()
    
    # For stability, always use CPU
    device = torch.device("cpu")
    print(f"Using CPU device for stability")
    
    # Load Shakespeare data
    train_data, val_data, encode, decode, vocab_size = load_shakespeare_data()
    
    if args.train:
        # Create a new model
        config = FHECompatibleConfig(vocab_size=vocab_size)
        model = FHECompatibleModel(config)
        
        # Train the model
        print("Starting training...")
        start_time = time.time()
        model = train_model(model, train_data, val_data, device=device)
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        
        # Test integer conversion
        test_int_conversion(model)
        
        # Generate some samples
        print("\nGenerating some samples:")
        prompts = ["King ", "Queen ", "Hamlet:", "Juliet:"]
        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            print(generate_text(model, encode, decode, prompt, max_tokens=50, device=device))
    
    elif args.generate:
        # Load existing model
        try:
            model, config = load_model('fhe_shakespeare_model.pt', device=device)
            print(f"Model loaded successfully: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
            
            # Generate text
            prompt = args.generate
            print(f"\nPrompt: '{prompt}'")
            print(generate_text(model, encode, decode, prompt, max_tokens=100, device=device))
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train a model first with --train")
    
    elif args.test_int:
        # Test integer conversion
        try:
            model, config = load_model('fhe_shakespeare_model.pt', device=device)
            test_int_conversion(model)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please train a model first with --train")
    
    else:
        # No arguments provided, print usage
        print("Usage: python simple_fhe_shakespeare.py [--train] [--generate 'prompt'] [--test-int]")
        print("  --train: Train a new model")
        print("  --generate 'prompt': Generate text from a prompt using the trained model")
        print("  --test-int: Test integer conversion capabilities") 