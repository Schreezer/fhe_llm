"""
Inference script for nanoGPT model without FHE
"""
import os
import argparse
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPT, GPTConfig

def load_model(checkpoint_path=None, model_type=None, device='cpu'):
    """
    Load a trained model from checkpoint or initialize a pre-trained GPT-2 model

    Args:
        checkpoint_path: Path to the checkpoint file
        model_type: Type of GPT-2 model to load (e.g., 'gpt2', 'gpt2-medium')
        device: Device to load the model on

    Returns:
        model: The loaded model
    """
    if checkpoint_path:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model_args' in checkpoint:
            # This is a checkpoint saved by train.py
            config = GPTConfig(**checkpoint['model_args'])
            model = GPT(config)
            state_dict = checkpoint['model']

            # Handle potential prefix in state dict keys
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

            model.load_state_dict(state_dict)
        else:
            # This is a simple state dict
            config = GPTConfig()
            model = GPT(config)
            model.load_state_dict(checkpoint)

    elif model_type and model_type.startswith('gpt2'):
        print(f"Loading pre-trained model: {model_type}")
        model = GPT.from_pretrained(model_type, dict(dropout=0.0))

    else:
        raise ValueError("Either checkpoint_path or model_type must be provided")

    model.eval()
    model.to(device)
    return model

def setup_encoding(checkpoint=None, encoding_type='gpt2'):
    """
    Set up encoding and decoding functions

    Args:
        checkpoint: Checkpoint dictionary that might contain dataset info
        encoding_type: Type of encoding to use if no meta.pkl is found

    Returns:
        encode: Function to encode text to token IDs
        decode: Function to decode token IDs to text
    """
    # Try to load meta.pkl if available
    load_meta = False
    if checkpoint and 'config' in checkpoint and 'dataset' in checkpoint['config']:
        meta_path = os.path.join(os.path.dirname(__file__), 'data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)

    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # Default to GPT-2 encoding
        print(f"Using {encoding_type} encoding...")
        enc = tiktoken.get_encoding(encoding_type)
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)

    return encode, decode

def generate_text(model, prompt, encode_fn, decode_fn, max_tokens=100, temperature=0.8, top_k=40, device='cpu'):
    """
    Generate text using the model

    Args:
        model: The GPT model
        prompt: Text prompt to start generation
        encode_fn: Function to encode text to token IDs
        decode_fn: Function to decode token IDs to text
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        device: Device to run inference on

    Returns:
        generated_text: The generated text
    """
    # Encode the prompt
    prompt_ids = encode_fn(prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, ...]

    # Generate
    with torch.no_grad():
        y = model.generate(x, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)
        generated_text = decode_fn(y[0].tolist())

    return generated_text

def main():
    parser = argparse.ArgumentParser(description='Run inference with nanoGPT')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default=None,
                        help='Pre-trained model type (e.g., gpt2, gpt2-medium)')
    parser.add_argument('--prompt', type=str, default='Once upon a time',
                        help='Text prompt to start generation')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling parameter')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')

    args = parser.parse_args()

    # Validate arguments
    if args.checkpoint is None and args.model_type is None:
        parser.error("Either --checkpoint or --model_type must be provided")

    # Set up device
    device = args.device
    print(f"Using device: {device}")

    # Load model
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model = load_model(checkpoint_path=args.checkpoint, device=device)
    else:
        checkpoint = None
        model = load_model(model_type=args.model_type, device=device)

    # Set up encoding/decoding
    encode, decode = setup_encoding(checkpoint)

    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("\nGenerating...")

    generated_text = generate_text(
        model,
        args.prompt,
        encode,
        decode,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )

    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
