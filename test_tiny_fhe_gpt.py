"""
Test a minimal FHE-compatible GPT variant that's compatible with Concrete-ML FHE.

This implements an extremely simplified version of the GPT architecture with:
1. A single transformer block instead of multiple
2. Aggressive scaling to keep bit widths small
3. Very small embedding dimension
4. No residual connections

It's designed to demonstrate that transformer-style language models can work under FHE.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Test a minimal FHE-compatible GPT')
parser.add_argument('--fhe_mode', choices=['execute', 'simulate', 'disable'], default='simulate',
                   help='FHE execution mode (execute/simulate/disable)')
args = parser.parse_args()

# Import Concrete ML
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
    """FHE-friendly polynomial activation (x² + x)"""
    def forward(self, x):
        return x * x * 0.1 + x * 0.1  # Scale down to keep bit width small

# Tiny FHE-compatible GPT model
class FHETinyGPT(nn.Module):
    """Extremely minimal GPT-style model compatible with FHE"""
    def __init__(self, vocab_size=256, d_model=16, context_length=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_length = context_length
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Single attention mechanism (no multi-head)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output head
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Activation
        self.act = FHEActivation()
        
        # Initialize with very small weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize all weights with small values for FHE compatibility"""
        for module in [self.query, self.key, self.value, self.out_proj, self.output]:
            if hasattr(module, 'weight'):
                with torch.no_grad():
                    module.weight.data.normal_(mean=0.0, std=0.01)
        
        # Embedding needs different initialization
        with torch.no_grad():
            self.embedding.weight.data.normal_(mean=0.0, std=0.01)
    
    def forward(self, x):
        """Forward pass with token IDs as input"""
        # Get batch size and sequence length
        b, t = x.size()
        
        # Embedding
        x = self.embedding(x) * 0.01  # Scale down embeddings
        
        # Single self-attention block
        q = self.query(x) * 0.01  # Scale for smaller values
        k = self.key(x) * 0.01    # Scale for smaller values
        v = self.value(x) * 0.1   # Less scaling for values
        
        # Attention scores with simple scaling (no softmax)
        # This uses direct element-wise operations to keep bit widths small
        attn = torch.bmm(q, k.transpose(1, 2)) * 0.01
        
        # Apply activation instead of softmax
        attn = self.act(attn)
        
        # Apply attention
        out = torch.bmm(attn, v) * 0.01
        out = self.out_proj(out) * 0.1
        
        # Apply output projection
        logits = self.output(out)
        
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Generate tokens using greedy decoding"""
        for _ in range(max_new_tokens):
            # Ensure we only use the last context_length tokens
            idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
            
            # Forward pass
            logits = self(idx_cond)
            
            # Take last token's prediction
            next_logits = logits[:, -1, :]
            
            # Greedy selection
            _, next_idx = torch.max(next_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            idx = torch.cat((idx, next_idx), dim=1)
        
        return idx

def main():
    """Main function to test the model with FHE"""
    print(f"Testing FHETinyGPT with FHE mode: {args.fhe_mode}")
    
    # Create a small model
    vocab_size = 256
    d_model = 16
    context_length = 16
    model = FHETinyGPT(vocab_size=vocab_size, d_model=d_model, context_length=context_length)
    
    # Create a dummy input
    batch_size = 1
    seq_len = 8  # Start with short sequence
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Run the original model
    print("Running original model...")
    with torch.no_grad():
        original_output = model(x)
    print(f"Original output shape: {original_output.shape}")
    
    # Convert input to NumPy for Concrete ML
    x_np = x.numpy()
    
    # Compile the model
    print("Compiling model for FHE...")
    n_bits = 3  # Very small bit width for maximum FHE compatibility
    try:
        quantized_model = compile_torch_model(
            model,
            x_np,
            n_bits=n_bits
        )
        
        # Run with the quantized model (no FHE)
        print("Running quantized model (no FHE)...")
        quantized_output = quantized_model.forward(x_np, fhe="disable")
        
        # Compare outputs
        original_np = original_output.detach().numpy()
        mae = np.abs(original_np - quantized_output).mean()
        print(f"Mean Absolute Error (original vs quantized): {mae}")
        
        # Try to run in FHE mode
        print(f"Running in FHE mode: {args.fhe_mode}...")
        if args.fhe_mode != "disable":
            try:
                fhe_circuit = quantized_model.fhe_circuit
                
                # Convert input to the expected integer format
                x_quantized = quantized_model.quantize_input(x_np)
                
                # Check the expected shape from the error
                print(f"Input shape: {x_np.shape}, Quantized input shape: {x_quantized.shape}")
                
                # The error indicates we need a (1, 8, 256) shape, which suggests one-hot encoding
                # Create a one-hot encoded version of the input
                one_hot = np.zeros((1, seq_len, vocab_size), dtype=np.int64)
                for i in range(seq_len):
                    one_hot[0, i, x_np[0, i]] = 1
                
                print(f"Created one-hot encoded input with shape {one_hot.shape}")
                
                # Run in the appropriate FHE mode
                if args.fhe_mode == "execute":
                    fhe_result = fhe_circuit.encrypt_run_decrypt(one_hot)
                    print("Successfully ran in FHE mode with encryption!")
                else:  # simulate
                    fhe_result = fhe_circuit.simulate(one_hot)
                    print("Successfully simulated FHE execution!")
                
                # Compare FHE vs non-FHE outputs
                fhe_mae = np.abs(quantized_output - fhe_result).mean()
                print(f"FHE vs quantized model MAE: {fhe_mae}")
                
                # Print some sample values
                print("\nSample output comparison:")
                print(f"Quantized: {quantized_output[0, 0, :5]}")
                print(f"FHE:       {fhe_result[0, 0, :5]}")
                
                # Try generating tokens
                print("\nGenerating tokens...")
                start_idx = torch.randint(0, vocab_size, (1, 1))
                original_tokens = model.generate(start_idx, max_new_tokens=5)
                print(f"Original model generated: {original_tokens[0].tolist()}")
                
                # We can't easily do generation with the FHE model due to recurrence,
                # but we can at least verify that the first token's prediction looks reasonable
                start_np = start_idx.numpy()
                first_logits = quantized_model.forward(start_np, fhe=args.fhe_mode)
                next_token = first_logits[0, 0].argmax()
                print(f"Next token from {args.fhe_mode} mode: {next_token}")
                
            except Exception as e:
                print(f"Error running in FHE mode: {e}")
                import traceback
                traceback.print_exc()
        
        return True
    
    except Exception as e:
        print(f"Error compiling/running the model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 