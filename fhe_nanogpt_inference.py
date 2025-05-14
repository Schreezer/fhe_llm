"""
FHE-compatible nanoGPT Inference Script

This script loads weights from a standard trained model and runs them through
the FHE-compatible architecture for Shakespeare text generation.
"""

import torch
import numpy as np
import sys
import os
from dataclasses import dataclass
from torch.nn import functional as F

# Import our FHE model architecture
from fhe_nanogpt_reduced_bitwidth import FHEGPTConfig, FHEReducedBitGPT, FHEActivation, FHELayerNorm

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


# Function to load Shakespeare data and mapping
def load_shakespeare_mappings():
    """Load the Shakespeare character mappings for encoding/decoding"""
    print("Loading Shakespeare character mappings...")
    
    # Make sure the file exists
    if not os.path.exists('nanoGPT/data/shakespeare/input.txt'):
        raise FileNotFoundError("Shakespeare dataset not found. Please run preparation script first.")
    
    # Load the text
    with open('nanoGPT/data/shakespeare/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create mappings
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # Create encode/decode functions
    def encode(s):
        return [char_to_idx[c] for c in s]
    
    def decode(l):
        return ''.join([idx_to_char[i] for i in l])
    
    return encode, decode, vocab_size


# Function to load trained weights into our FHE-compatible model
def load_fhe_model(weights_path='trained_shakespeare_gpt.pt'):
    """Load trained weights into FHE-compatible model"""
    print(f"Loading trained weights from {weights_path}...")
    
    # Get vocabulary size from Shakespeare data
    encode, decode, vocab_size = load_shakespeare_mappings()
    
    # Load the original weights to check their architecture
    trained_weights = torch.load(weights_path)
    
    # Print some weight stats for debugging
    print("\nInspecting trained weights:")
    zero_weights = 0
    total_weights = 0
    for name, tensor in trained_weights.items():
        total_weights += tensor.numel()
        zero_weights += (tensor == 0).sum().item()
        print(f"{name}: shape={tensor.shape}, min={tensor.min().item():.6f}, max={tensor.max().item():.6f}, mean={tensor.mean().item():.6f}")
    
    print(f"Zero weights: {zero_weights}/{total_weights} ({zero_weights/total_weights*100:.2f}%)")
    
    # Check if dimensions in the trained weights
    mlp_dim = None
    for key, value in trained_weights.items():
        if 'mlp.fc1.weight' in key:
            mlp_dim = value.shape[0]  # This should be 256 based on the error
            print(f"Detected MLP dimension: {mlp_dim}")
            break
    
    # Set default if not found
    if mlp_dim is None:
        mlp_dim = 256
        print(f"Using default MLP dimension: {mlp_dim}")
    
    # Create FHE model config with matching dimensions
    config = FHEGPTConfig(
        block_size=32,
        vocab_size=vocab_size,
        n_layer=3,
        n_embd=64,
        mlp_dim=mlp_dim  # Pass the detected dimension
    )
    
    # Create model
    model = FHEReducedBitGPT(config)
    
    # Add hooks to capture intermediate activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for key layers
    model.wte.register_forward_hook(get_activation('wte'))
    model.blocks[0].attn.register_forward_hook(get_activation('block0.attn'))
    model.blocks[0].mlp.register_forward_hook(get_activation('block0.mlp'))
    model.lm_head.register_forward_hook(get_activation('lm_head'))
    
    # Save activations in the model for later inspection
    model.activations = activations
    
    # Load the trained weights
    try:
        # If it's a state_dict, load directly
        if isinstance(trained_weights, dict) and 'state_dict' in trained_weights:
            model.load_state_dict(trained_weights['state_dict'])
        # If it's our custom format, map the weights
        else:
            # Map keys from trained model to FHE model
            mapping = {
                'tok_emb.weight': 'wte.weight',
                'pos_emb.weight': 'wpe.weight',
                'blocks.0.attn.key.weight': 'blocks.0.attn.key.weight',
                'blocks.0.attn.query.weight': 'blocks.0.attn.query.weight',
                'blocks.0.attn.value.weight': 'blocks.0.attn.value.weight',
                'blocks.0.attn.proj.weight': 'blocks.0.attn.out_proj.weight',
                'blocks.0.mlp.fc1.weight': 'blocks.0.mlp.fc1.weight',
                'blocks.0.mlp.fc2.weight': 'blocks.0.mlp.fc2.weight',
                'blocks.1.attn.key.weight': 'blocks.1.attn.key.weight',
                'blocks.1.attn.query.weight': 'blocks.1.attn.query.weight',
                'blocks.1.attn.value.weight': 'blocks.1.attn.value.weight',
                'blocks.1.attn.proj.weight': 'blocks.1.attn.out_proj.weight',
                'blocks.1.mlp.fc1.weight': 'blocks.1.mlp.fc1.weight',
                'blocks.1.mlp.fc2.weight': 'blocks.1.mlp.fc2.weight',
                'blocks.2.attn.key.weight': 'blocks.2.attn.key.weight',
                'blocks.2.attn.query.weight': 'blocks.2.attn.query.weight',
                'blocks.2.attn.value.weight': 'blocks.2.attn.value.weight',
                'blocks.2.attn.proj.weight': 'blocks.2.attn.out_proj.weight',
                'blocks.2.mlp.fc1.weight': 'blocks.2.mlp.fc1.weight',
                'blocks.2.mlp.fc2.weight': 'blocks.2.mlp.fc2.weight',
                'lm_head.weight': 'lm_head.weight'
            }
            
            # Load mapped weights
            fhe_state_dict = {}
            for train_key, fhe_key in mapping.items():
                if train_key in trained_weights:
                    fhe_state_dict[fhe_key] = trained_weights[train_key]
            
            # Load compatible weights
            model.load_state_dict(fhe_state_dict, strict=False)
        
        # Verify the weights were loaded
        print("\nVerifying loaded weights:")
        zero_weights = 0
        total_weights = 0
        for name, param in model.named_parameters():
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()
            print(f"{name}: shape={param.shape}, min={param.min().item():.6f}, max={param.max().item():.6f}, mean={param.mean().item():.6f}")
            
            # If weights are exactly zero, warn
            if torch.all(param == 0).item():
                print(f"WARNING: Parameter {name} is all zeros!")
        
        print(f"FHE model zero weights: {zero_weights}/{total_weights} ({zero_weights/total_weights*100:.2f}%)")
        
        # Test the model with a simple forward pass
        print("\nTesting model with a simple forward pass...")
        test_input = torch.tensor([[0, 1, 2]]).to(next(model.parameters()).device)
        with torch.no_grad():
            test_output = model(test_input)
            
            # Print activation stats
            for name, activation in model.activations.items():
                if torch.all(activation == 0).item():
                    print(f"WARNING: {name} activation is all zeros!")
                else:
                    print(f"{name} activation: shape={activation.shape}, min={activation.min().item():.6f}, max={activation.max().item():.6f}")
                
        print("Weights loaded successfully")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise
    
    return model, encode, decode


# Function to generate text with the FHE-compatible model
def generate_text_fhe(model, encode, decode, prompt="King ", max_tokens=100):
    """Generate text using the FHE-compatible model"""
    print(f"\n=== Generating text with FHE model for prompt: '{prompt}' ===")
    
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
    print(f"Encoding prompt...")
    encoded_prompt = torch.tensor(encode(prompt)).unsqueeze(0).to(device)  # Add batch dimension
    print(f"Encoded prompt shape: {encoded_prompt.shape}")
    print(f"Encoded prompt tokens: {encoded_prompt.tolist()}")
    
    # Debug forward pass of model
    print("Testing forward pass...")
    with torch.no_grad():
        logits = model(encoded_prompt)
        print(f"Logits shape: {logits.shape}")
        print(f"Logits min/max: {logits.min().item():.4f}/{logits.max().item():.4f}")
    
    # Define our own custom generation function
    def custom_generate(idx, max_new_tokens):
        """Generate text using the model with verbose debugging"""
        print("Using custom generation function for debugging...")
        
        # idx is (B, T) array of indices in the current context
        for i in range(max_new_tokens):
            print(f"\nGenerating token {i+1}/{max_new_tokens}...")
            
            # Crop to the last block_size tokens if needed
            if idx.size(1) > model.config.block_size:
                print(f"Cropping context from {idx.size(1)} to {model.config.block_size} tokens")
                input_idx = idx[:, -model.config.block_size:]
            else:
                input_idx = idx
            
            # Get the predictions
            with torch.no_grad():
                print(f"Running model on input shape: {input_idx.shape}")
                logits = model(input_idx)  # (B, T, vocab_size)
                print(f"Got logits with shape: {logits.shape}")
                
                # Focus only on the last time step
                last_logits = logits[:, -1, :]  # (B, vocab_size)
                print(f"Last token logits shape: {last_logits.shape}")
                print(f"Logits min/max: {last_logits.min().item():.4f}/{last_logits.max().item():.4f}")
                
                # Apply softmax to convert to probabilities
                probs = F.softmax(last_logits, dim=-1)
                print(f"Top 5 most likely tokens:")
                top_probs, top_idx = torch.topk(probs, k=5)
                for j, (p, ix) in enumerate(zip(top_probs[0], top_idx[0])):
                    print(f"  {j+1}. Token {ix.item()} ('{decode([ix.item()])}') with prob {p.item():.4f}")
                
                # Sample from the distribution
                try:
                    idx_next = torch.multinomial(probs, num_samples=1)
                    print(f"Sampled token: {idx_next.item()} -> '{decode([idx_next.item()])}'")
                except Exception as e:
                    print(f"Error sampling: {e}")
                    # Just take the most likely token if sampling fails
                    idx_next = top_idx[:, 0:1]
                    print(f"Falling back to most likely token: {idx_next.item()} -> '{decode([idx_next.item()])}'")
            
            # Append to the sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            print(f"Current generated text: '{decode(idx[0].tolist())}'")
        
        return idx
    
    # Generate text
    print("Starting text generation...")
    with torch.no_grad():
        try:
            print("Attempting to use built-in generate...")
            # First try using the model's built-in generate method
            output_ids = model.generate(encoded_prompt, max_tokens)
            print("Used built-in generate successfully")
        except Exception as e:
            print(f"Error using built-in generate: {e}")
            # Fall back to our custom implementation
            print("Falling back to custom generate function")
            output_ids = custom_generate(encoded_prompt, max_tokens)
    
    # Decode the output
    print("Decoding output...")
    generated_text = decode(output_ids[0].tolist())
    
    return generated_text


# Compile model for FHE simulation
def compile_and_simulate_fhe(model, prompt="K", n_bits=3):
    """Compile the model for FHE and run a simulation"""
    print("\n=== Compiling model for FHE and running simulation ===")
    
    # Get encode/decode functions
    encode, decode, _ = load_shakespeare_mappings()
    
    # Encode prompt
    encoded_prompt = torch.tensor(encode(prompt)).unsqueeze(0)  # Add batch dimension
    
    # Run standard inference first
    print("Running regular inference...")
    model.eval()
    with torch.no_grad():
        original_output = model(encoded_prompt)
    
    # Prepare for FHE compilation
    x_np = encoded_prompt.numpy()
    
    # Compile the model
    print(f"Compiling model for FHE with {n_bits}-bit quantization...")
    try:
        quantized_module = compile_torch_model(
            model,
            x_np,
            n_bits=n_bits
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
            # prepare that for the first token
            vocab_size = model.config.vocab_size
            seq_len = encoded_prompt.size(1)
            one_hot = np.zeros((1, seq_len, vocab_size), dtype=np.int64)
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
    import sys
    
    if len(sys.argv) > 1:
        # Check if we have a trained model
        if os.path.exists('trained_shakespeare_gpt.pt'):
            # Load trained weights
            model, encode, decode = load_fhe_model('trained_shakespeare_gpt.pt')
            
            # Get prompt
            prompt = sys.argv[1] if len(sys.argv) > 1 else "King "
            
            # Generate text
            generated_text = generate_text_fhe(model, encode, decode, prompt)
            print(f"Generated text:\n{generated_text}")
            
            # Also try FHE simulation if requested
            if "--simulate" in sys.argv:
                compile_and_simulate_fhe(model, prompt=prompt[:1])  # Use just first letter for simulation
        
        elif os.path.exists('fhe_compatible_weights.pt'):
            # Load FHE-compatible weights
            model, encode, decode = load_fhe_model('fhe_compatible_weights.pt')
            
            # Get prompt
            prompt = sys.argv[1] if len(sys.argv) > 1 else "King "
            
            # Generate text
            generated_text = generate_text_fhe(model, encode, decode, prompt)
            print(f"Generated text:\n{generated_text}")
            
            # Also try FHE simulation if requested
            if "--simulate" in sys.argv:
                compile_and_simulate_fhe(model, prompt=prompt[:1])  # Use just first letter for simulation
        
        else:
            print("No trained model found. Please run train_fhe_nanogpt_shakespeare.py first.")
    
    else:
        # If no arguments given, show usage
        print("Usage: python fhe_nanogpt_inference.py <prompt> [--simulate]")
        print("Example: python fhe_nanogpt_inference.py \"King \"")
        print("Add --simulate to also run FHE simulation on the first letter of the prompt.") 