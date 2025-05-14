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
import torch.nn.functional as F

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
    """FHE-compatible activation function - simplified LeCun tanh for FHE"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # In FHE, we need to use very simple activation functions
        # ReLU or approximation of tanh
        
        # For debugging, print some stats
        if torch.is_grad_enabled() == False and torch.rand(1)[0] < 0.01:  # Only occasionally during inference
            print(f"FHEActivation input range: {x.min().item():.4f} to {x.max().item():.4f}")
        
        # Apply a simplified activation: clipped ReLU approximation
        # x = torch.clamp(x, min=0)  # ReLU
        
        # Try a different approach - scale by small constant to ensure values don't vanish
        x = x * 0.5  # Just scale to prevent explosion/vanishing
        
        return x

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
    """FHE-compatible MLP with fixed-bit activations"""
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, config.mlp_dim, bias=False)
        self.activation = FHEActivation()
        self.fc2 = nn.Linear(config.mlp_dim, config.n_embd, bias=False)
    
    def forward(self, x):
        # MLP with quantized activations
        x = self.fc1(x)  # [batch, seq, n_embd] -> [batch, seq, 4*n_embd]
        x = self.activation(x)  # Apply FHE-friendly activation
        x = self.fc2(x)  # [batch, seq, 4*n_embd] -> [batch, seq, n_embd]
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
    """GPT-2 configuration for FHE"""
    block_size: int = 16  # Maximum context window
    vocab_size: int = 256  # In FHE models, we can use a byte-level encoding
    n_layer: int = 2  # Number of transformer layers
    n_head: int = 4  # Number of attention heads
    n_embd: int = 32  # Embedding dimension
    mlp_dim: int = 128  # Dimension of the MLP intermediate layer
    dropout: float = 0.0  # Not used in FHE model

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
    
    def generate(self, idx, max_new_tokens):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        # Print model stats during generation
        print(f"Generating with FHE-compatible model, context size: {idx.shape}")
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # If the sequence context is growing too long, truncate it at the block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # Forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            
            # Debug: check if logits are all zero
            if torch.all(logits == 0).item():
                print("WARNING: All logits are zero! Debugging...")
                # Try to diagnose where the zeroing happens
                for name, param in self.named_parameters():
                    if torch.all(param == 0).item():
                        print(f"Parameter {name} is all zeros!")
                    elif _ == 0:  # Only on first token
                        print(f"Parameter {name} stats: min={param.min().item():.4f}, max={param.max().item():.4f}")
            
            # Print logits stats for debugging
            print(f"Generation step {_+1}/{max_new_tokens}, logits: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
            
            # Focus on the last time step
            logits = logits[:, -1, :]  # (B, C)
            
            # If logits are too small, add small noise to prevent all-zero distributions
            if logits.max().item() < 0.01:
                print("Logits too small, adding small noise...")
                logits = logits + torch.randn_like(logits) * 0.01
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
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

# Function to load and prepare Shakespeare data
def load_shakespeare_data():
    """Load and prepare Shakespeare data for training"""
    print("Loading Shakespeare dataset...")
    
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
    
    return train_data, val_data, encode, decode, vocab_size

# Function to get a batch of data
def get_batch(data, block_size, batch_size):
    """Generate a batch of data for training or validation"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Function to train the FHE-compatible model
def train_fhe_model():
    """Train the FHE-compatible model on Shakespeare dataset"""
    print("\n=== Training FHE-compatible model on Shakespeare dataset ===")
    
    # Load Shakespeare data
    train_data, val_data, encode, decode, vocab_size = load_shakespeare_data()
    
    # Set training hyperparameters
    batch_size = 64
    block_size = 32  # Context window size
    learning_rate = 1e-3
    max_iters = 5000
    eval_interval = 500
    
    # Create the model configuration
    config = FHEGPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=3,
        n_embd=64
    )
    
    # Create the model
    model = FHEReducedBitGPT(config)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("Starting training...")
    for iter in range(max_iters):
        # Sample a batch of data
        xb, yb = get_batch(train_data, block_size, batch_size)
        
        # Forward pass
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print training progress
        if iter % eval_interval == 0 or iter == max_iters - 1:
            print(f"Iteration {iter}: Training loss {loss.item():.4f}")
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data, block_size, batch_size)
                logits = model(xv)
                val_loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), yv.view(-1))
                print(f"Validation loss: {val_loss.item():.4f}")
            model.train()
    
    # Save the trained model
    torch.save(model.state_dict(), 'fhe_nanogpt_shakespeare.pt')
    print("Model saved to fhe_nanogpt_shakespeare.pt")
    
    return model, encode, decode

# Function to generate text using the trained model
def generate_text(model, encode, decode, prompt, max_new_tokens=100):
    """Generate text using the trained model"""
    print(f"\n=== Generating text with prompt: '{prompt}' ===")
    
    # Encode the prompt
    encoded_prompt = torch.tensor(encode(prompt)).unsqueeze(0)  # Add batch dimension
    
    # Generate text
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(encoded_prompt, max_new_tokens)
    
    # Decode the output
    generated_text = decode(output_ids[0].tolist())
    
    return generated_text

if __name__ == "__main__":
    # Choose what to do
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        # Train the model and generate text with a prompt
        model, encode, decode = train_fhe_model()
        generated_text = generate_text(model, encode, decode, "King ")
        print(f"Generated text:\n{generated_text}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'generate':
        # Load the trained model and generate text
        config = FHEGPTConfig()
        train_data, val_data, encode, decode, vocab_size = load_shakespeare_data()
        
        # Update config with correct vocab size
        config.vocab_size = vocab_size
        config.n_layer = 3
        config.n_embd = 64
        config.block_size = 32
        
        model = FHEReducedBitGPT(config)
        model.load_state_dict(torch.load('fhe_nanogpt_shakespeare.pt'))
        
        prompt = "King "
        if len(sys.argv) > 2:
            prompt = sys.argv[2]
            
        generated_text = generate_text(model, encode, decode, prompt)
        print(f"Generated text:\n{generated_text}")
    
    else:
        # Run the FHE test
        success = test_fhe_model()
        
        if success:
            print("\nSuccess! The reduced bit-width model works with FHE.")
        else:
            print("\nThe model still needs adjustments to work with FHE.") 