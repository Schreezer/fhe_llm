"""
Train, quantize, and test an FHE-compatible nanoGPT model on the Shakespeare character dataset

This script:
1. Trains an FHE-compatible GPT model on the Shakespeare character dataset
2. Quantizes the trained model for FHE
3. Tests the model in both regular and FHE execution modes
4. Generates sample text to verify the model works correctly
"""

import os
import time
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Import our FHE-compatible model
from fhe_nanogpt import FHEGPT, FHEGPTConfig, compile_fhe_model, compile_torch_model

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train and test FHE-compatible nanoGPT')
parser.add_argument('--skip_fhe', action='store_true', help='Skip FHE compilation (for debugging)')
parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
args = parser.parse_args()

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Dataset preparation
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.block_size = block_size
        self.data = data
        
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get chunk of data starting at idx
        chunk = self.data[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_shakespeare_data(block_size):
    # Load the preprocessed Shakespeare dataset
    data_dir = os.path.join('nanoGPT', 'data', 'shakespeare_char')
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Load metadata
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    
    # Create datasets
    train_dataset = CharDataset(train_data, block_size)
    val_dataset = CharDataset(val_data, block_size)
    
    return train_dataset, val_dataset, meta

# Training loop
def train(model, train_loader, val_loader, device, lr=3e-4, epochs=1):
    """Train the model on the Shakespeare dataset"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    best_val_loss = float('inf')
    start_time = time.time()
    
    # Training statistics
    train_losses = []
    val_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss
            train_loss += loss.item()
            
            # Print progress
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'fhe_nanogpt_shakespeare.pt')
            print(f"Saved model with val loss: {val_loss:.4f}")
    
    # Print training summary
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return train_losses, val_losses

# Text generation function
def generate_text(model, meta, device, start_text="", max_new_tokens=100, temperature=1.0):
    """Generate text from the trained model"""
    # Prepare input
    if not start_text:
        # Start with a random character
        idx = torch.randint(len(meta['itos']), (1, 1), device=device)
    else:
        # Encode the start text
        encoded = [meta['stoi'][c] for c in start_text]
        idx = torch.tensor([encoded], dtype=torch.long, device=device)
    
    # Generate
    model.eval()
    with torch.no_grad():
        generated = model.generate(idx, max_new_tokens=max_new_tokens)
    
    # Decode the generated tokens
    generated_text = ''.join([meta['itos'][int(i)] for i in generated[0].tolist()])
    return generated_text

# Compare generation between original, quantized, and FHE models
def compare_generation(original_model, quantized_model, meta, device, start_chars="The"):
    """Compare text generation between original and quantized models"""
    print("\n=== Comparing Text Generation ===")
    
    # Encode the starting text
    start_indices = [meta['stoi'][c] for c in start_chars]
    start_tensor = torch.tensor([start_indices], dtype=torch.long, device=device)
    
    # Generate from original model
    print("\nGenerating from original model...")
    original_model.eval()
    with torch.no_grad():
        original_gen = original_model.generate(start_tensor, max_new_tokens=50)
    original_text = ''.join([meta['itos'][int(i)] for i in original_gen[0].tolist()])
    print(f"Original model: {original_text}")
    
    # Generate from quantized model (without FHE)
    print("\nGenerating from quantized model (without FHE)...")
    start_numpy = start_tensor.cpu().numpy()
    
    # Run the first token through quantized model (without FHE)
    quantized_output = quantized_model.forward(start_numpy, fhe="disable")
    
    # Manually implement greedy generation for the quantized model
    gen_indices = list(start_indices)
    next_idx = np.argmax(quantized_output[0, -1, :])
    gen_indices.append(next_idx)
    
    # Continue generating
    for _ in range(49):  # generate 49 more tokens
        # Prepare the current sequence
        current_seq = np.array([gen_indices[-32:]], dtype=np.int64)  # keep last 32 tokens
        
        # Forward pass
        quantized_output = quantized_model.forward(current_seq, fhe="disable")
        
        # Get the next token
        next_idx = np.argmax(quantized_output[0, -1, :])
        gen_indices.append(next_idx)
    
    quantized_text = ''.join([meta['itos'][int(i)] for i in gen_indices])
    print(f"Quantized model: {quantized_text}")
    
    # Try generating with FHE simulation if possible
    try:
        print("\nGenerating from quantized model (with FHE simulation)...")
        # We can only do one token at a time in FHE mode
        first_token = np.array([start_indices], dtype=np.int64)
        fhe_output = quantized_model.forward(first_token, fhe="simulate")
        next_token = np.argmax(fhe_output[0, -1, :])
        print(f"FHE simulation first token: '{meta['itos'][int(next_token)]}'")
    except Exception as e:
        print(f"Error generating with FHE: {e}")

def test_single_block(block_size=32, device="cpu"):
    """Test a single transformer block with FHE
    
    This function extracts just one block from the trained model and tests it with FHE,
    which has a better chance of working than the full model.
    """
    print("\n=== Testing Single Transformer Block with FHE ===")
    
    # Load the trained model
    config = FHEGPTConfig(
        block_size=block_size,
        vocab_size=65,  # From Shakespeare dataset
        n_layer=4,      # Small number of layers
        n_embd=64       # Small embedding size
    )
    
    full_model = FHEGPT(config)
    
    # Load trained weights if available
    if os.path.exists('fhe_nanogpt_shakespeare.pt'):
        print("Loading pre-trained weights...")
        full_model.load_state_dict(torch.load('fhe_nanogpt_shakespeare.pt', map_location="cpu"))
    
    # Extract a single transformer block
    print("Extracting a single transformer block...")
    single_block = full_model.blocks[0]  # Just use the first block
    
    # Create a simple input for testing
    x = torch.ones((1, block_size, config.n_embd)) * 0.01  # Small values for numerical stability
    
    # Test the block with the input
    print("Testing block with dummy input...")
    with torch.no_grad():
        output = single_block(x)
        print(f"Output shape: {output.shape}")
    
    # Compile for FHE
    print("Compiling block for FHE...")
    try:
        # Convert input to numpy
        x_np = x.numpy()
        
        # Compile with a small bit width
        quantized_block = compile_torch_model(
            single_block,
            x_np,
            n_bits=3
        )
        
        # Test with quantized model
        print("Testing block with quantized model...")
        q_output = quantized_block.forward(x_np)
        
        # Compare outputs
        mae = np.abs(output.detach().numpy() - q_output).mean()
        print(f"Mean Absolute Error: {mae}")
        
        # Try FHE simulation
        print("Testing FHE simulation...")
        fhe_output = quantized_block.forward(x_np, fhe="simulate")
        fhe_mae = np.abs(q_output - fhe_output).mean()
        print(f"FHE vs quantized MAE: {fhe_mae}")
        
        return True
    except Exception as e:
        print(f"Error compiling/running block in FHE: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Parameters
    block_size = 32
    batch_size = 64
    epochs = args.epochs
    learning_rate = 3e-4
    
    # Get device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_dataset, val_dataset, meta = get_shakespeare_data(block_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    vocab_size = meta['vocab_size']
    print(f"Vocabulary size: {vocab_size}")
    
    # Create FHE-compatible model with the correct vocabulary size
    config = FHEGPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=4,          # Small number of layers
        n_embd=64           # Small embedding size
    )
    
    model = FHEGPT(config)
    model.to(device)
    
    # Check if pre-trained model exists
    if os.path.exists('fhe_nanogpt_shakespeare.pt'):
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load('fhe_nanogpt_shakespeare.pt', map_location=device))
    else:
        print("Training model from scratch...")
        train_losses, val_losses = train(model, train_loader, val_loader, device, lr=learning_rate, epochs=epochs)
    
    # Generate some text with the trained model
    print("\n=== Generating text with trained model ===")
    generated_text = generate_text(model, meta, device, start_text="O ", max_new_tokens=100)
    print(f"Generated text:\n{generated_text}")
    
    # Save original device before quantization
    original_device = next(model.parameters()).device
    
    # Try to compile the full model for FHE
    print("\n=== Testing FHE compatibility ===")
    
    # First try a single block, which is more likely to work
    single_block_ok = test_single_block(block_size=block_size)
    
    if not single_block_ok:
        print("\nSingle block failed with FHE, full model is unlikely to work.")
        print("\nDone!")
        return
    
    # Now try the full model if the single block worked
    print("\n=== Quantizing full model for FHE ===")
    model.cpu()  # Move to CPU for quantization
    
    # Create proper dummy input for compilation
    # Use actual data formatting to ensure correct input shapes
    dummy_input = torch.ones((1, block_size), dtype=torch.long) % vocab_size  # Ensure valid token IDs
    print(f"Dummy input shape: {dummy_input.shape}, dtype: {dummy_input.dtype}")
    
    # Convert to int64 (long) not uint16 because embedding requires int/long
    dummy_input_np = dummy_input.numpy().astype(np.int64)  
    print(f"Dummy numpy input shape: {dummy_input_np.shape}, dtype: {dummy_input_np.dtype}")
    
    # Test model with dummy input to verify it works before compilation
    print("Testing model with dummy input before compilation...")
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Test output shape: {test_output.shape}")
    
    # Compile the model with a small bit width
    print("Compiling model for FHE...")
    try:
        # Use smaller bit width for faster compilation
        quantized_model = compile_fhe_model(
            model, 
            input_shape=(1, block_size), 
            n_bits=3,  # Reduced from 4 for faster compilation
            dummy_input_data=dummy_input_np  # Provide explicit dummy data
        )
        
        # Move the model back to original device for generation
        model.to(original_device)
        
        # Compare generation between original and quantized models
        compare_generation(model, quantized_model, meta, device=original_device, start_chars="KING: ")
    except Exception as e:
        print(f"Error during FHE compilation: {e}")
        import traceback
        traceback.print_exc()
        print("\nSkipping FHE compilation due to error.")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 