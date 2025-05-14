"""
Training script for a Shakespeare model that can be converted to FHE-compatible format.
This uses more standard training techniques, then converts to FHE format for inference.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
import time
from dataclasses import dataclass

# Configuration class
@dataclass
class TrainableGPTConfig:
    block_size: int = 32      # Context window
    vocab_size: int = 65      # Shakespeare vocabulary size
    n_layer: int = 3          # Number of transformer layers
    n_head: int = 4           # Number of attention heads
    n_embd: int = 64          # Embedding dimension
    dropout: float = 0.0      # Dropout rate
    bias: bool = False        # No bias for FHE compatibility

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention - trainable version"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.dropout = nn.Dropout(config.dropout)
        # parameters
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Calculate query, key, values
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Compute attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        
        # Re-assemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection
        y = self.proj(y)
        y = self.dropout(y)
        
        return y

class MLP(nn.Module):
    """Multi-layer perceptron - trainable version"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """Transformer block - trainable version"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TrainableGPT(nn.Module):
    """Trainable GPT model that can be converted to FHE format"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=True)
        
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.tok_emb.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"Trainable GPT parameters: {self.get_num_params() / 1e6:.2f}M")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # Get embeddings
        tok_emb = self.tok_emb(idx)
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        pos_emb = self.pos_emb(pos)
        
        # Forward
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Generate text using sampling"""
        for _ in range(max_new_tokens):
            # Crop if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Get predictions for the last token
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax
            probs = F.softmax(logits, dim=-1)
            
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


# Function to load and prepare Shakespeare data
def load_shakespeare_data():
    """Load and prepare Shakespeare data for training"""
    print("Loading Shakespeare dataset...")
    
    # Check if the input.txt file exists
    if not os.path.exists('nanoGPT/data/shakespeare/input.txt'):
        raise FileNotFoundError(
            "Shakespeare dataset not found. Please ensure 'nanoGPT/data/shakespeare/input.txt' exists."
        )
    
    # Load the Shakespeare text
    with open('nanoGPT/data/shakespeare/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Dataset length: {len(text)}")
    
    # Get unique characters (vocabulary)
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create mapping from characters to integers and back
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Function to encode and decode text
    def encode(s):
        return [char_to_idx[c] for c in s]
    
    def decode(l):
        return ''.join([idx_to_char[i] for i in l])
    
    # Encode the entire text
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # Split data into train and validation sets (90-10 split)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, encode, decode, vocab_size, char_to_idx, idx_to_char


# Function to get a batch of data
def get_batch(data, block_size, batch_size):
    """Generate a batch of data for training or validation"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


# Function to train the model
def train_model():
    """Train a model on Shakespeare dataset"""
    print("\n=== Training Shakespeare GPT model ===")
    
    # Set device - use MPS on Mac if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
    
    # Load data
    train_data, val_data, encode, decode, vocab_size, _, _ = load_shakespeare_data()
    
    # Set training hyperparameters
    batch_size = 64
    block_size = 32
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    
    # Create model configuration
    config = TrainableGPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=3,
        n_head=4,
        n_embd=64,
        dropout=0.1
    )
    
    # Create model
    model = TrainableGPT(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # For timing
    start_time = time.time()
    
    # Training loop
    print("Starting training...")
    for iter in range(max_iters):
        # Sample a batch of data
        xb, yb = get_batch(train_data, block_size, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        
        # Forward pass
        logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print training progress
        if iter % eval_interval == 0 or iter == max_iters - 1:
            elapsed = time.time() - start_time
            print(f"Iteration {iter}: Training loss {loss.item():.4f} | Time elapsed: {elapsed:.2f}s")
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data, block_size, batch_size)
                xv, yv = xv.to(device), yv.to(device)
                _, val_loss = model(xv, yv)
                print(f"Validation loss: {val_loss.item():.4f}")
            model.train()
    
    # Save the trained model
    torch.save(model.state_dict(), 'trained_shakespeare_gpt.pt')
    print("Model saved to trained_shakespeare_gpt.pt")
    
    return model, encode, decode


# Function to generate text
def generate_text(model, encode, decode, prompt="King ", max_tokens=100):
    """Generate text using the trained model"""
    print(f"\n=== Generating text with prompt: '{prompt}' ===")
    
    # Set device - use MPS on Mac if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS device for generation")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device for generation")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device for generation")
    
    model = model.to(device)
    model.eval()
    
    # Encode the prompt
    encoded_prompt = torch.tensor(encode(prompt)).unsqueeze(0).to(device)
    
    # Generate text with sampling
    with torch.no_grad():
        output_ids = model.generate(encoded_prompt, max_tokens, temperature=0.8)
    
    # Decode the output
    generated_text = decode(output_ids[0].tolist())
    
    return generated_text


# Function to convert trained model to FHE-compatible format (modify fhe_nanogpt_reduced_bitwidth.py to use this)
def save_fhe_compatible_weights(model_path='trained_shakespeare_gpt.pt'):
    """Save weights in a format that can be loaded by the FHE model"""
    print("Converting model to FHE-compatible format...")
    
    # Load the standard model
    config = TrainableGPTConfig()
    train_data, val_data, encode, decode, vocab_size, _, _ = load_shakespeare_data()
    config.vocab_size = vocab_size
    
    model = TrainableGPT(config)
    model.load_state_dict(torch.load(model_path))
    
    # Save in a format that can be loaded by the FHE model
    torch.save({
        'tok_emb.weight': model.tok_emb.weight.data,
        'pos_emb.weight': model.pos_emb.weight.data,
        'blocks.0.attn.key.weight': model.blocks[0].attn.key.weight.data,
        'blocks.0.attn.query.weight': model.blocks[0].attn.query.weight.data,
        'blocks.0.attn.value.weight': model.blocks[0].attn.value.weight.data,
        'blocks.0.attn.proj.weight': model.blocks[0].attn.proj.weight.data,
        'blocks.0.mlp.fc1.weight': model.blocks[0].mlp.fc1.weight.data,
        'blocks.0.mlp.fc2.weight': model.blocks[0].mlp.fc2.weight.data,
        'blocks.1.attn.key.weight': model.blocks[1].attn.key.weight.data,
        'blocks.1.attn.query.weight': model.blocks[1].attn.query.weight.data,
        'blocks.1.attn.value.weight': model.blocks[1].attn.value.weight.data,
        'blocks.1.attn.proj.weight': model.blocks[1].attn.proj.weight.data,
        'blocks.1.mlp.fc1.weight': model.blocks[1].mlp.fc1.weight.data,
        'blocks.1.mlp.fc2.weight': model.blocks[1].mlp.fc2.weight.data,
        'blocks.2.attn.key.weight': model.blocks[2].attn.key.weight.data,
        'blocks.2.attn.query.weight': model.blocks[2].attn.query.weight.data,
        'blocks.2.attn.value.weight': model.blocks[2].attn.value.weight.data,
        'blocks.2.attn.proj.weight': model.blocks[2].attn.proj.weight.data,
        'blocks.2.mlp.fc1.weight': model.blocks[2].mlp.fc1.weight.data,
        'blocks.2.mlp.fc2.weight': model.blocks[2].mlp.fc2.weight.data,
        'lm_head.weight': model.lm_head.weight.data,
        'vocab_size': vocab_size,
        'char_to_idx': train_data[0].item(),  # We'll use this to verify encoding consistency
    }, 'fhe_compatible_weights.pt')
    
    print("Saved FHE-compatible weights to fhe_compatible_weights.pt")
    return encode, decode


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'convert':
        # Just convert an existing model to FHE-compatible format
        encode, decode = save_fhe_compatible_weights()
    else:
        # Train and generate
        model, encode, decode = train_model()
        prompt = "King "
        if len(sys.argv) > 1:
            prompt = sys.argv[1]
        
        generated_text = generate_text(model, encode, decode, prompt)
        print(f"Generated text:\n{generated_text}")
        
        # Save in FHE-compatible format
        save_fhe_compatible_weights('trained_shakespeare_gpt.pt') 