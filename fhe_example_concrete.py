"""
FHE Example using Concrete-ML

This script demonstrates how to use Concrete-ML to perform encrypted inference
with the nanoGPT model.
"""

import torch
import time
from transformers import AutoTokenizer
from concrete_ml_fhe import ConcreteMLFHE

def main():
    """Run an example of the Concrete-ML FHE model with nanoGPT."""
    print("Initializing Concrete-ML FHE model for nanoGPT...")
    
    # Path to the extracted weights
    weights_path = "ciphertext/data/extracted_weights_nanogpt.pth"
    
    # Initialize the ConcreteMLFHE class
    concrete_fhe = ConcreteMLFHE(model_path=weights_path, n_bits=4)
    
    # Quantize the model
    quantized_model = concrete_fhe.quantize_model()
    
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define a sample input
    input_text = "Encryption is"
    print(f"Input text: '{input_text}'")
    
    # Run inference in clear mode first
    print("\nRunning inference in clear mode...")
    start_time = time.time()
    clear_text = concrete_fhe.run_inference(
        input_text,
        tokenizer,
        max_new_tokens=5,
        use_fhe=False
    )
    clear_time = time.time() - start_time
    print(f"Clear inference time: {clear_time:.2f} seconds")
    print(f"Generated text (clear): {clear_text}")
    
    # Run inference in FHE simulation mode
    print("\nRunning inference in FHE simulation mode...")
    start_time = time.time()
    fhe_text = concrete_fhe.run_inference(
        input_text,
        tokenizer,
        max_new_tokens=5,
        use_fhe=True
    )
    fhe_time = time.time() - start_time
    print(f"FHE simulation time: {fhe_time:.2f} seconds")
    print(f"Generated text (FHE): {fhe_text}")
    
    # Compare results
    print("\nComparing results:")
    if clear_text == fhe_text:
        print("✅ FHE simulation produces identical results to clear inference!")
    else:
        print("❌ FHE simulation produces different results from clear inference.")
        print(f"Clear: {clear_text}")
        print(f"FHE:   {fhe_text}")

if __name__ == "__main__":
    main() 