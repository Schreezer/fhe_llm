"""
Script to check the shape of the weights in the nanoGPT checkpoint
"""

import torch
import sys
import os

def main():
    """Check the shape of the weights in the nanoGPT checkpoint"""
    print("Checking nanoGPT weights...")
    
    # Path to the pre-trained weights
    weight_path = "nanoGPT/out-shakespeare-char/ckpt.pt"
    
    # Check if the file exists
    if not os.path.exists(weight_path):
        print(f"Error: Checkpoint file not found at {weight_path}")
        print(f"Current working directory: {os.getcwd()}")
        return
    
    # Load the checkpoint
    print(f"Loading weights from {weight_path}")
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    if 'model' in checkpoint:
        # Extract model weights from checkpoint
        weights = checkpoint['model']
    else:
        weights = checkpoint
    
    # Print the keys and shapes of the weights
    print("\nWeight keys and shapes:")
    for key, value in weights.items():
        print(f"{key}: {value.shape}")
    
    # Check the QKV weights specifically
    for i in range(2):  # Check first two transformer blocks
        prefix = f'transformer.h.{i}.'
        qkv_key = f'{prefix}attn.c_attn.weight'
        if qkv_key in weights:
            qkv_weight = weights[qkv_key]
            print(f"\nQKV weight {qkv_key}: {qkv_weight.shape}")
            
            # Try to split into Q, K, V parts
            w_q, w_k, w_v = torch.chunk(qkv_weight, 3, dim=0)
            print(f"Q part: {w_q.shape}")
            print(f"K part: {w_k.shape}")
            print(f"V part: {w_v.shape}")

if __name__ == "__main__":
    main() 