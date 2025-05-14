"""
Test the FHE model from fhe_nanogpt
"""
import torch
import argparse
from fhe_model import FHEGPT, FHEGPTConfig

def main():
    parser = argparse.ArgumentParser(description='Test FHE model')
    parser.add_argument('--model_path', type=str, default='../fhe_nanogpt/models/fhe_model_final.pt',
                        help='Path to the model checkpoint')
    parser.add_argument('--prompt', type=str, default='Hello world',
                        help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=20,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Set up device
    device = args.device
    print(f"Using device: {device}")
    
    # Set up character-level encoding for Shakespeare dataset
    # In a real implementation, you would load this from the dataset
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?-_'\"\n "
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, ' ') for i in l])
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model configuration
    config = FHEGPTConfig(
        block_size=128,
        vocab_size=vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=128,
        bias=True,
        n_bits=8
    )
    
    # Create the FHE-compatible model
    print("Creating FHE-compatible model...")
    model = FHEGPT(config)
    
    # Load the model weights
    print(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    
    # Check if the state dict keys match the model
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    
    print(f"Model has {len(model_keys)} keys")
    print(f"State dict has {len(state_dict_keys)} keys")
    
    # Find missing keys
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    # Try to load the state dict
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Model loaded successfully (with some missing/unexpected keys)")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model.to(device)
    model.eval()
    
    # Encode the prompt
    print(f"\nPrompt: {args.prompt}")
    encoded_prompt = encode(args.prompt)
    x = torch.tensor(encoded_prompt, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    print("\nGenerating...")
    with torch.no_grad():
        output = model.generate(x, max_new_tokens=args.max_tokens, temperature=0.8, top_k=40)
        generated_text = decode(output[0].tolist())
    
    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
