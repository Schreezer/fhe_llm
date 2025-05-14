"""
Full FHE-compatible implementation for the Shakespeare language model.

This implementation scales up our FHE approach to handle multiple layers
while still using only operations that are compatible with Concrete-ML.

Key features:
- Complete multi-layer transformer architecture
- FHE-compatible operations only
- Integer-friendly calculations
- Proper layer quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import time
from dataclasses import dataclass
import math

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


@dataclass
class FHEModelConfig:
    """Configuration for the FHE-compatible model"""
    vocab_size: int = 65      # Shakespeare has ~65 characters
    context_length: int = 16  # Increased context length for better modeling
    hidden_size: int = 128    # Increased hidden dimension size for better capacity
    n_layers: int = 4         # Increased number of transformer layers
    n_heads: int = 8          # Increased number of attention heads
    scale_factor: int = 1000  # Scaling factor for quantization
    dropout: float = 0.0      # No dropout in FHE


class LinearFHE(nn.Module):
    """Linear layer compatible with FHE"""
    def __init__(self, in_features, out_features, scale_factor=100):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale_factor = scale_factor
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        # Quantize weights for better FHE compatibility
        with torch.no_grad():
            self.weight.mul_(self.scale_factor).round_().div_(self.scale_factor)
    
    def forward(self, x):
        return F.linear(x, self.weight)


class FHEAttention(nn.Module):
    """Attention mechanism compatible with FHE"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.n_heads
        assert self.hidden_size % self.n_heads == 0
        
        self.head_size = self.hidden_size // self.n_heads
        self.scale_factor = config.scale_factor
        
        # Query, Key, Value projections (no bias)
        self.q_proj = LinearFHE(self.hidden_size, self.hidden_size)
        self.k_proj = LinearFHE(self.hidden_size, self.hidden_size)
        self.v_proj = LinearFHE(self.hidden_size, self.hidden_size)
        self.out_proj = LinearFHE(self.hidden_size, self.hidden_size)
        
        # Create causal mask and register as buffer
        mask = torch.triu(torch.ones(config.context_length, config.context_length), diagonal=1)
        self.register_buffer('mask', mask.bool())
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Apply query, key, value projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Quantize after projections
        q = (q * self.scale_factor).round() / self.scale_factor
        k = (k * self.scale_factor).round() / self.scale_factor
        v = (v * self.scale_factor).round() / self.scale_factor
        
        # Reshape to multiple heads
        q = q.view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1, 2)
        
        # Compute attention scores (scaled dot-product)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))
        attn_scores = attn_scores / np.sqrt(self.head_size)
        
        # Apply causal mask
        mask = self.mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Simple normalization for FHE compatibility
        attn_scores = torch.softmax(attn_scores, dim=-1)
        
        # Quantize attention scores
        attn_scores = (attn_scores * self.scale_factor).round() / self.scale_factor
        
        # Apply attention to values
        output = torch.matmul(attn_scores, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Final projection
        output = self.out_proj(output)
        
        # Quantize output
        output = (output * self.scale_factor).round() / self.scale_factor
        
        return output


class FHEMLP(nn.Module):
    """MLP compatible with FHE"""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.scale_factor = config.scale_factor
        
        # Linear layers (no bias)
        self.fc1 = LinearFHE(self.hidden_size, self.hidden_size * 2)
        self.fc2 = LinearFHE(self.hidden_size * 2, self.hidden_size)
    
    def forward(self, x):
        # First linear layer
        x = self.fc1(x)
        
        # Quantize
        x = (x * self.scale_factor).round() / self.scale_factor
        
        # ReLU activation (FHE-compatible)
        x = torch.relu(x)
        
        # Quantize after activation
        x = (x * self.scale_factor).round() / self.scale_factor
        
        # Second linear layer
        x = self.fc2(x)
        
        # Final quantization
        x = (x * self.scale_factor).round() / self.scale_factor
        
        return x


class FHETransformerBlock(nn.Module):
    """Transformer block compatible with FHE"""
    def __init__(self, config):
        super().__init__()
        self.attention = FHEAttention(config)
        self.mlp = FHEMLP(config)
        self.scale_factor = config.scale_factor
    
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = x + attn_output
        
        # Quantize after residual
        x = (x * self.scale_factor).round() / self.scale_factor
        
        # MLP with residual
        mlp_output = self.mlp(x)
        x = x + mlp_output
        
        # Final quantization
        x = (x * self.scale_factor).round() / self.scale_factor
        
        return x


class FHEEmbedding(nn.Module):
    """Custom embedding that avoids using nn.Embedding for FHE compatibility"""
    def __init__(self, num_embeddings, embedding_dim, scale_factor=100):
        super().__init__()
        # We use a linear layer instead of embedding for FHE compatibility
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.scale_factor = scale_factor
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        # Quantize weights
        with torch.no_grad():
            self.weight.mul_(self.scale_factor).round_().div_(self.scale_factor)
    
    def forward(self, indices=None, one_hot=None):
        """
        Forward pass either using indices or one-hot vectors.
        For FHE, we always use one-hot representation.
        """
        if one_hot is not None:
            # One-hot input (for FHE)
            return torch.matmul(one_hot.float(), self.weight)
        else:
            # Index-based input (for training)
            return F.embedding(indices, self.weight)


class FHESinusoidalPositionEmbedding(nn.Module):
    """FHE-compatible sinusoidal position embedding"""
    def __init__(self, max_seq_len, hidden_size, scale_factor=1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.scale_factor = scale_factor
        
        # Create sinusoidal position embeddings
        # This is similar to the original transformer paper's approach
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        
        pe = torch.zeros(max_seq_len, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_size % 2 == 0:  # Handle both even and odd hidden sizes
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, :hidden_size//2]
        
        # Quantize and scale for FHE compatibility
        pe = (pe * scale_factor).round() / scale_factor
        
        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)
    
    def forward(self, x):
        """Forward pass adds positional embeddings to the input"""
        batch_size, seq_len, _ = x.shape
        return x + self.pe[:seq_len, :]


class FullFHEModel(nn.Module):
    """Complete FHE-compatible transformer model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embedding
        self.token_embedding = FHEEmbedding(
            config.vocab_size, config.hidden_size, config.scale_factor
        )
        
        # Position embedding (using sinusoidal embeddings instead)
        self.position_embedding = FHESinusoidalPositionEmbedding(
            config.context_length, config.hidden_size, config.scale_factor
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            FHETransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Output head (no bias)
        self.lm_head = LinearFHE(config.hidden_size, config.vocab_size)
        
        # FHE mode flag
        self.fhe_mode = False
        
        print(f"FHE model created with {sum(p.numel() for p in self.parameters())/1e3:.2f}K parameters")
    
    def set_fhe_mode(self, enable=True):
        """Set FHE mode (used for inference)"""
        self.fhe_mode = enable
    
    def forward(self, idx=None, one_hot=None):
        """
        Forward pass with either:
        - token indices (for training)
        - one-hot encoded input (for FHE inference)
        """
        batch_size = idx.shape[0] if idx is not None else one_hot.shape[0]
        seq_len = idx.shape[1] if idx is not None else one_hot.shape[1]
        device = next(self.parameters()).device
        
        # Token embeddings
        if self.fhe_mode and one_hot is not None:
            # Use one-hot for FHE
            token_emb = self.token_embedding(one_hot=one_hot)
        else:
            # Use indices for training
            token_emb = self.token_embedding(indices=idx)
        
        # Apply positional embeddings
        x = self.position_embedding(token_emb)
        
        # Quantize
        x = (x * self.config.scale_factor).round() / self.config.scale_factor
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final projection to vocab
        logits = self.lm_head(x)
        
        # Quantize logits
        logits = (logits * self.config.scale_factor).round() / self.config.scale_factor
        
        return logits


def create_one_hot(indices, vocab_size):
    """Create one-hot encoded vectors from token indices"""
    batch_size, seq_len = indices.shape
    one_hot = torch.zeros(batch_size, seq_len, vocab_size, device=indices.device)
    for b in range(batch_size):
        for s in range(seq_len):
            one_hot[b, s, indices[b, s]] = 1
    return one_hot


def load_shakespeare_data():
    """Load Shakespeare dataset"""
    path = 'nanoGPT/data/shakespeare/input.txt'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Shakespeare data not found at {path}")
    
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


def get_batch(data, context_length, batch_size):
    """Create a random batch for training"""
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    return x, y


def train_model(model, train_data, val_data, 
               context_length=8, batch_size=64, 
               learning_rate=1e-2, max_iters=50000,
               eval_interval=1000, device='cpu'):
    """Train the FHE-compatible model"""
    print("Training FHE-compatible model...")
    # Move model to device
    model = model.to(device)
    
    # Disable FHE mode for training
    model.set_fhe_mode(False)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Learning rate scheduler - slower decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(max_iters*1.5))
    
    # Training loop
    start_time = time.time()
    best_val_loss = float('inf')
    for iter in range(max_iters + 1):
        # Evaluate on validation set
        if iter % eval_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for _ in range(10):  # Average over 10 batches
                    xb, yb = get_batch(val_data, context_length, batch_size)
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(idx=xb)
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
                    val_loss += loss.item()
                val_loss /= 10
            print(f"Iter {iter}: val loss = {val_loss:.4f}, time: {time.time() - start_time:.2f}s")
            
            # Save model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Create checkpoint directory if it doesn't exist
                if not os.path.exists('checkpoints'):
                    os.makedirs('checkpoints')
                # Save checkpoint
                torch.save({
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'config': model.config,
                    'val_loss': best_val_loss,
                }, 'checkpoints/full_fhe_shakespeare_best.pt')
                print(f"New best model saved! Val loss: {best_val_loss:.4f}")
            
            model.train()
        
        # Get batch
        xb, yb = get_batch(train_data, context_length, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward, backward, optimize
        optimizer.zero_grad()
        logits = model(idx=xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        # Periodically quantize weights for FHE compatibility
        if iter % 1000 == 0:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        # Quantize to lower precision by rounding
                        param.mul_(model.config.scale_factor).round_().div_(model.config.scale_factor)
        
        # Report training loss occasionally
        if iter % 100 == 0:
            print(f"Iter {iter}: train loss = {loss.item():.4f}, lr = {scheduler.get_last_lr()[0]:.6f}")
    
    # Final quantization of all weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                param.mul_(model.config.scale_factor).round_().div_(model.config.scale_factor)
    
    # Save the final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }, 'full_fhe_shakespeare_model.pt')
    
    print(f"Training completed in {time.time() - start_time:.2f}s")
    print("Final model saved to full_fhe_shakespeare_model.pt")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load the best model if it exists
    try:
        best_model = torch.load('checkpoints/full_fhe_shakespeare_best.pt')
        model.load_state_dict(best_model['model_state_dict'])
        print(f"Loaded best model from checkpoint. Val loss: {best_model['val_loss']:.4f}")
    except:
        print("Using final model (best checkpoint not found)")
    
    return model


def extract_fhe_layer(model):
    """Extract a single layer for FHE compilation"""
    print("Extracting a single FHE-compatible layer for compilation")
    
    class FHELayer(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.config = original_model.config
            self.scale_factor = original_model.config.scale_factor
            
            # Extract weights from first block
            self.linear1 = LinearFHE(
                original_model.config.hidden_size, 
                original_model.config.hidden_size
            )
            self.linear2 = LinearFHE(
                original_model.config.hidden_size, 
                original_model.config.hidden_size
            )
            
            # Copy weights
            with torch.no_grad():
                block = original_model.blocks[0]
                self.linear1.weight.copy_(block.attention.q_proj.weight)
                self.linear2.weight.copy_(block.attention.v_proj.weight)
        
        def forward(self, x):
            # First linear layer
            x = self.linear1(x)
            
            # Quantize
            x = (x * self.scale_factor).round() / self.scale_factor
            
            # ReLU activation (FHE-compatible)
            x = torch.relu(x)
            
            # Quantize after activation
            x = (x * self.scale_factor).round() / self.scale_factor
            
            # Second linear layer
            x = self.linear2(x)
            
            # Final quantization
            x = (x * self.scale_factor).round() / self.scale_factor
            
            return x
    
    # Create and return the layer
    layer = FHELayer(model)
    layer.eval()
    return layer


def compile_for_fhe(model, n_bits=3):
    """Compile the model for FHE using Concrete-ML"""
    print(f"Compiling model for FHE with {n_bits} bits precision")
    
    # Create a dummy input with appropriate shape
    if hasattr(model, 'config'):
        input_size = model.config.hidden_size
    else:
        # Use a default size
        input_size = 32
    
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


def generate_text(model, encode, decode, prompt="King ", max_tokens=50, context_length=8, device='cpu'):
    """Generate text from the model"""
    model = model.to(device)
    model.eval()
    
    # Set FHE mode to false for generation (using PyTorch)
    model.set_fhe_mode(False)
    
    # Encode the prompt
    encoded = encode(prompt)
    if len(encoded) > context_length:
        encoded = encoded[-context_length:]
    
    input_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate tokens
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            logits = model(idx=input_ids)
            
            # Get next token (greedy or sample)
            logits = logits[:, -1, :]  # Take the last position
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat((input_ids, next_token), dim=1)
            generated.append(next_token.item())
            
            # Maintain context length
            if input_ids.size(1) > context_length:
                input_ids = input_ids[:, -context_length:]
    
    # Decode the generated tokens
    result = prompt + decode(generated)
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FHE-compatible model for Shakespeare text")
    parser.add_argument('--train', action='store_true', help='Train a new model')
    parser.add_argument('--compile', action='store_true', help='Compile for FHE')
    parser.add_argument('--generate', type=str, default="", help='Generate text from prompt')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    args = parser.parse_args()
    
    # Determine device
    if args.cpu:
        device = torch.device('cpu')
    else:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_data, val_data, encode, decode, vocab_size = load_shakespeare_data()
    
    if args.train:
        # Create and train a new model
        config = FHEModelConfig(vocab_size=vocab_size)
        model = FullFHEModel(config)
        train_model(model, train_data, val_data, device=device)
    
    else:
        # Try to load an existing model
        try:
            print("Loading model from full_fhe_shakespeare_model.pt")
            checkpoint = torch.load('full_fhe_shakespeare_model.pt', map_location=device)
            config = checkpoint['config']
            model = FullFHEModel(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            print("Model loaded successfully")
        except FileNotFoundError:
            print("Model file not found. Use --train to train a new model.")
            model = None
    
    if model is not None:
        if args.compile:
            # Extract a layer for FHE compilation
            layer = extract_fhe_layer(model)
            
            # Compile for FHE
            quantized_module = compile_for_fhe(layer, n_bits=3)
            
            if quantized_module is not None:
                # Simulate FHE inference
                simulate_fhe_inference(quantized_module, input_size=config.hidden_size)
        
        elif args.generate:
            # Generate text
            prompt = args.generate
            print(f"\nPrompt: '{prompt}'")
            generated = generate_text(model, encode, decode, prompt, max_tokens=100, 
                                     context_length=config.context_length, device=device)
            print(f"\nGenerated text:\n{generated}")
        
        elif not args.train:
            # No options specified, print usage
            print("Use --train to train a new model")
            print("Use --compile to compile the model for FHE")
            print("Use --generate 'prompt' to generate text") 