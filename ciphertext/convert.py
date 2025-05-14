import torch
import os
from safetensors.torch import load_file

# Added import for loading PyTorch checkpoints
import sys
sys.path.append('../')  # Add parent directory to path

########################################################################
### Input data are converted in the below code                       ###
### To make an appropriate input data, we should generate the input  ###
### with the following code, where the output is a container.        ###
########################################################################

class Container(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def add(self, key: str, value: torch.Tensor):
        setattr(self, key, value)

# Update dimensions to match the nanoGPT model
DIM: int = 384  # Changed from 768 to match nanoGPT
HIDDEN_DIM: int = 1536  # Changed from 3072 to match nanoGPT
HIDDEN_HALF_DIM: int = 768  # Changed from 1536 to match nanoGPT
MAX_SEQ_LEN: int = 128

NUM_HEADS: int = 6  # Changed from 12 to match nanoGPT
BATCH_SIZE: int = 16

# Caution: SUB_DIM = 128 (not 64)
SUBMATRIX_DIM: int = 128



def convert(weight: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Convert a weight tensor to the format expected by the FHE implementation
    Handles different dimensions by padding if necessary
    """
    print(f"Converting weight with shape {weight.shape} to {height}x{width}")
    
    # Ensure weight has the right shape
    if weight.shape != (height, width) and weight.shape != (width, height):
        # Try to reshape or pad the weight tensor
        if weight.shape[0] * weight.shape[1] == height * width:
            weight = weight.reshape(height, width)
        else:
            # Create a new tensor with the right shape
            new_weight = torch.zeros(height, width)
            # Copy as much of the original weight as possible
            h = min(weight.shape[0], height)
            w = min(weight.shape[1], width)
            if weight.shape[0] == width and weight.shape[1] == height:
                # Transpose if dimensions are swapped
                new_weight[:h, :w] = weight.t()[:h, :w]
            else:
                new_weight[:h, :w] = weight[:h, :w]
            weight = new_weight
    
    # Ensure dimensions are multiples of SUBMATRIX_DIM
    if height % SUBMATRIX_DIM != 0 or width % SUBMATRIX_DIM != 0:
        padded_height = ((height + SUBMATRIX_DIM - 1) // SUBMATRIX_DIM) * SUBMATRIX_DIM
        padded_width = ((width + SUBMATRIX_DIM - 1) // SUBMATRIX_DIM) * SUBMATRIX_DIM
        padded_weight = torch.zeros(padded_height, padded_width)
        padded_weight[:height, :width] = weight
        weight = padded_weight
        height, width = padded_height, padded_width
    
    # Now reshape according to the original algorithm
    return (
        weight.transpose(0, 1)
        .view(
            height // SUBMATRIX_DIM,
            SUBMATRIX_DIM,
            width // SUBMATRIX_DIM,
            SUBMATRIX_DIM,
        )
        .transpose(1, 2)
    )


def convert_norm(weight: torch.Tensor) -> torch.Tensor:
    """Convert layer norm weights to the format expected by the FHE implementation"""
    # Ensure weight has the right shape
    if len(weight.shape) == 1 and weight.shape[0] <= DIM:
        # Pad if necessary
        if weight.shape[0] < DIM:
            padded = torch.zeros(DIM)
            padded[:weight.shape[0]] = weight
            weight = padded
        
        # Ensure dimensions are multiples of SUBMATRIX_DIM
        target_dim = ((DIM + SUBMATRIX_DIM - 1) // SUBMATRIX_DIM) * SUBMATRIX_DIM
        if target_dim != DIM:
            extended = torch.zeros(target_dim)
            extended[:DIM] = weight
            weight = extended
        
        # Debug print
        print(f"Norm weight shape: {weight.shape}, target_dim: {target_dim}")
        
        try:
            return (
                weight.repeat(MAX_SEQ_LEN, 1)
                .view(MAX_SEQ_LEN, target_dim // SUBMATRIX_DIM, SUBMATRIX_DIM)
                .transpose(0, 1)
            )
        except RuntimeError as e:
            print(f"Error in convert_norm: {e}")
            
            # Fallback to a simpler approach
            result = torch.zeros(target_dim // SUBMATRIX_DIM, MAX_SEQ_LEN, SUBMATRIX_DIM)
            for i in range(target_dim // SUBMATRIX_DIM):
                result[i, :, :] = weight[i * SUBMATRIX_DIM:(i+1) * SUBMATRIX_DIM].repeat(MAX_SEQ_LEN, 1)
            return result
    else:
        print(f"Warning: unexpected shape for norm weight: {weight.shape}")
        # Create a default weight
        default_weight = torch.ones(DIM)
        return convert_norm(default_weight)

def convert_norm_final(weight: torch.Tensor) -> torch.Tensor:
    """Convert final layer norm weights to the format expected by the FHE implementation"""
    # Ensure weight has the right shape
    if len(weight.shape) == 1:
        # Pad if necessary
        padded = torch.zeros(DIM)  # Use DIM instead of hardcoded 1024
        padded[:min(weight.shape[0], DIM)] = weight[:min(weight.shape[0], DIM)]
        weight = padded
        
        # Ensure dimensions are multiples of SUBMATRIX_DIM
        target_dim = ((DIM + SUBMATRIX_DIM - 1) // SUBMATRIX_DIM) * SUBMATRIX_DIM
        if target_dim != DIM:
            extended = torch.zeros(target_dim)
            extended[:DIM] = weight
            weight = extended
        
        # Debug print
        print(f"Final norm weight shape: {weight.shape}, target_dim: {target_dim}")
        
        try:
            return (
                weight.repeat(MAX_SEQ_LEN, 1)
                .view(MAX_SEQ_LEN, target_dim // SUBMATRIX_DIM, SUBMATRIX_DIM)
                .transpose(0, 1)
            )
        except RuntimeError as e:
            print(f"Error in convert_norm_final: {e}")
            
            # Fallback to a simpler approach
            result = torch.zeros(target_dim // SUBMATRIX_DIM, MAX_SEQ_LEN, SUBMATRIX_DIM)
            for i in range(target_dim // SUBMATRIX_DIM):
                result[i, :, :] = weight[i * SUBMATRIX_DIM:(i+1) * SUBMATRIX_DIM].repeat(MAX_SEQ_LEN, 1)
            return result
    else:
        print(f"Warning: unexpected shape for final norm weight: {weight.shape}")
        # Create a default weight
        default_weight = torch.ones(DIM)
        return convert_norm_final(default_weight)

# [2, 1024] -> 2 x [8 ,128, 128]
def convert_pooling_final(weight: torch.Tensor, height:int, width:int) ->torch.Tensor:
    tmp = torch.zeros(height,width)
    tmp[:,:2] = weight.transpose(0,1)
    return (
        tmp.view(8, 128, 128)
    )

def convert_pooling_bias(weight: torch.Tensor) -> torch.Tensor:
    tmp = torch.zeros(1024,128)
    tmp[:,:1] = weight.view(1024,1)
    return (
        tmp.view(8, 128, 128)
    )

def convert_pooling_weight(weight: torch.Tensor, height:int, width:int) -> torch.Tensor:
    return (
        weight.view(
            height // 128,
            128,
            width // 128,
            128,
        )
        .transpose(1, 2)
    )

def convert_embed_mask(weight: torch.Tensor) -> torch.Tensor:
    return (
        weight[0][0].reshape(1,128).repeat(128,1)
    )

def convert_pooling_weight_a(weight: torch.Tensor) -> torch.Tensor:
    weight = torch.split(weight, 128, dim = 1)
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    for i in range(3):
        data1[i*32:(i+1)*32, :] = weight[i*2]
        data2[i*32:(i+1)*32, :] = weight[i*2+1]
    return (torch.stack((data1, data2), dim = 0))

def convert_pooling_bias_a(weight: torch.Tensor) -> torch.Tensor:
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    data1[:32,0] = weight
    return (torch.stack((data1, data2), dim = 0))

def convert_pooling_weight_b(weight: torch.Tensor) -> torch.Tensor:
    weight = weight.transpose(0,1)
    weight = torch.split(weight, 128, dim = 1)
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    for i in range(4):
        data1[i*32:(i+1)*32, :] = weight[i*2]
        data2[i*32:(i+1)*32, :] = weight[i*2+1]
    return (torch.stack((data1, data2), dim = 0))

def convert_pooling_bias_b(weight: torch.Tensor) -> torch.Tensor:
    weight = weight.view(1,1024)
    weight = torch.split(weight, 128, dim = 1)
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    for i in range(4):
        data1[i*32:i*32+1, :] = weight[i*2]
        data2[i*32:i*32+1, :] = weight[i*2+1]
    return (torch.stack((data1, data2), dim = 0))

def convert_head_weight(weight: torch.Tensor) -> torch.Tensor:
    weight = torch.split(weight, 128, dim = 1)
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    for i in range(4):
        data1[i*32:i*32+2, :] = weight[i*2]
        data2[i*32:i*32+2, :] = weight[i*2+1]
    return (torch.stack((data1, data2), dim = 0))

def convert_head_bias(weight: torch.Tensor) -> torch.Tensor:
    data1 = torch.zeros(128,128)
    data2 = torch.zeros(128,128)
    data1[:2,0] = weight
    return (torch.stack((data1, data2), dim = 0))






def main():
    """Convert nanoGPT weights for FHE inference"""
    print("Converting nanoGPT weights for FHE inference...")
    
    # Path to the pre-trained weights
    weight_path = "nanoGPT/out-shakespeare-char/ckpt.pt"
    
    # Output path for the converted weights
    save_path = "ciphertext/data/converted_weights_nanogpt.pth"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create container for the converted weights
    container = Container()
    
    # Load the checkpoint
    print(f"Loading weights from {weight_path}")
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    if 'model' in checkpoint:
        # Extract model weights from checkpoint
        weights = checkpoint['model']
    else:
        weights = checkpoint
    
    print("Converting weights for FHE format...")
    
    # Process transformer blocks - use only the first 2 blocks
    for i in range(2):
        prefix = f'transformer.h.{i}.'
        
        if f'{prefix}attn.c_attn.weight' in weights:
            # QKV weights - need to split them
            qkv_weight = weights[f'{prefix}attn.c_attn.weight']
            # Split into Q, K, V parts (assuming they're concatenated along output dimension)
            w_q, w_k, w_v = torch.chunk(qkv_weight, 3, dim=0)
            
            print(f"Processing transformer block {i} with QKV shape {qkv_weight.shape}")
            
            # Add converted weights to container
            container.add(
                f'{i}_wq', convert(w_q, DIM, DIM)
            )
            container.add(
                f'{i}_wk', convert(w_k, DIM, DIM)
            )
            container.add(
                f'{i}_wv', convert(w_v, DIM, DIM)
            )
            
            # Output projection
            container.add(
                f'{i}_wd', convert(weights[f'{prefix}attn.c_proj.weight'], DIM, DIM)
            )
            
            # MLP weights
            container.add(
                f'{i}_wdin', convert(weights[f'{prefix}mlp.c_fc.weight'], DIM, HIDDEN_DIM)
            )
            container.add(
                f'{i}_wdout', convert(weights[f'{prefix}mlp.c_proj.weight'], HIDDEN_DIM, DIM)
            )
            
            # Layer norms
            container.add(
                f'{i}_norm1_w', convert_norm(weights[f'{prefix}ln_1.weight'])
            )
            if f'{prefix}ln_1.bias' in weights:
                container.add(
                    f'{i}_norm1_b', convert_norm(weights[f'{prefix}ln_1.bias'])
                )
            else:
                # Some models don't have bias
                container.add(
                    f'{i}_norm1_b', convert_norm(torch.zeros_like(weights[f'{prefix}ln_1.weight']))
                )
                
            container.add(
                f'{i}_norm2_w', convert_norm(weights[f'{prefix}ln_2.weight'])
            )
            if f'{prefix}ln_2.bias' in weights:
                container.add(
                    f'{i}_norm2_b', convert_norm(weights[f'{prefix}ln_2.bias'])
                )
            else:
                # Some models don't have bias
                container.add(
                    f'{i}_norm2_b', convert_norm(torch.zeros_like(weights[f'{prefix}ln_2.weight']))
                )
            
            # Add LoRA weights (initialized to zeros as they'll be fine-tuned)
            container.add(
                f'{i}_lora_attn_a', torch.zeros(DIM, 2)  # rank=2
            )
            container.add(
                f'{i}_lora_attn_b', torch.zeros(2, DIM)  # rank=2
            )
    
    # Add final layer norm
    container.add(
        'final_norm_w', convert_norm_final(weights['transformer.ln_f.weight'])
    )
    if 'transformer.ln_f.bias' in weights:
        container.add(
            'final_norm_b', convert_norm_final(weights['transformer.ln_f.bias'])
        )
    else:
        container.add(
            'final_norm_b', convert_norm_final(torch.zeros_like(weights['transformer.ln_f.weight']))
        )
    
    # Add embedding weights
    container.add(
        'wte', weights['transformer.wte.weight']  # Token embeddings
    )
    container.add(
        'wpe', weights['transformer.wpe.weight']  # Position embeddings
    )
    
    # Save the converted weights
    print(f"Saving converted weights to {save_path}")
    torch.save(container, save_path)
    print("Conversion complete!")

if __name__ == "__main__":
    main()
