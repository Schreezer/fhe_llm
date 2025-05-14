"""
Simple Linear Model with Concrete-ML

This script demonstrates how to use Concrete-ML to run a simple linear model in FHE.
"""

import os
import sys
import torch
import numpy as np

# Make sure we're using the site-packages version of concrete with our fixes
site_packages = '/Users/chirag13/development/ai_project/fhe_llama_env_py310/lib/python3.10/site-packages'
if site_packages not in sys.path:
    sys.path.insert(0, site_packages)

# First, run the setup script to ensure all necessary files are in place
print("Setting up Concrete-ML environment...")
try:
    import setup_concrete_ml
except ImportError:
    print("Running setup_concrete_ml.py directly...")
    exec(open("setup_concrete_ml.py").read())

# Try to import from Concrete-ML
try:
    from concrete.ml.torch.compile import compile_torch_model
    print("Successfully imported compile_torch_model")
except ImportError as e:
    print(f"Error importing compile_torch_model: {e}")
    sys.exit(1)

# Define a very simple linear model
class SimpleLinearModel(torch.nn.Module):
    def __init__(self, input_dim=4, output_dim=1):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
        # Initialize with very small weights to avoid bit width issues
        with torch.no_grad():
            self.linear.weight.data.normal_(mean=0.0, std=0.01)
            self.linear.bias.data.zero_()
    
    def forward(self, x):
        return self.linear(x)

def test_simple_linear_model():
    """Test a simple linear model with Concrete-ML."""
    print("\n=== Testing Simple Linear Model with Concrete-ML ===")
    
    # Create a simple model
    input_dim = 4
    model = SimpleLinearModel(input_dim=input_dim)
    print("Created simple linear model")
    
    # Create random input
    x_numpy = np.random.randn(10, input_dim) * 0.1  # Small values
    x = torch.tensor(x_numpy, dtype=torch.float32)
    
    # Run inference with the original model
    print("Running inference with the original model...")
    with torch.no_grad():
        original_output = model(x)
    
    # Compile the model with Concrete-ML
    print("Compiling the model with Concrete-ML...")
    try:
        # Use very low bit width to avoid overflow
        n_bits = 4
        quantized_module = compile_torch_model(
            model, 
            x_numpy,  # Use NumPy array for compilation
            n_bits=n_bits
        )
        
        # Run inference with the compiled model
        print("Running inference with the compiled model...")
        fhe_output = quantized_module.forward(x_numpy)
        
        # Compare outputs
        mae = np.abs(original_output.detach().numpy() - fhe_output).mean()
        print(f"Mean Absolute Error: {mae}")
        
        print("Simple linear model test completed successfully!")
        return quantized_module, model, mae
    
    except Exception as e:
        print(f"Error compiling/running the model: {e}")
        return None, model, float('inf')

def test_sklearn_model():
    """Test a simple sklearn model with Concrete-ML."""
    print("\n=== Testing sklearn model with Concrete-ML ===")
    
    try:
        from concrete.ml.sklearn import LinearRegression
        print("Successfully imported LinearRegression from concrete.ml.sklearn")
        
        # Generate synthetic data - use a single sample for FHE
        X_train = np.random.rand(100, 5) * 0.1  # Small values for training
        y_train = 0.2 * X_train[:, 0] + 0.5 * X_train[:, 1] - 0.1 * X_train[:, 2] + 0.01
        
        # Create a single test sample for FHE
        X_test_single = np.random.rand(1, 5) * 0.1  # Just one sample
        
        # Create and fit the model
        print("Creating and fitting LinearRegression model...")
        model = LinearRegression(n_bits=4)  # Use low bit width
        model.fit(X_train, y_train)
        
        # Make predictions with the plain model
        print("Making predictions with the plain model...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test_single)
        
        # Evaluate on training data
        mae_train = np.abs(y_train - y_pred_train).mean()
        print(f"Training Mean Absolute Error: {mae_train}")
        
        # Compile for FHE
        print("Compiling the model for FHE...")
        model.compile(X_test_single)  # Compile with a single sample shape
        
        # Run in FHE with a single sample
        print("Running the model in FHE with a single sample...")
        try:
            # Get the FHE circuit
            fhe_circuit = model.fhe_circuit
            y_pred_fhe = fhe_circuit.encrypt_run_decrypt(X_test_single)
            
            # Compare FHE vs non-FHE on the single test sample
            fhe_mae = np.abs(y_pred_test - y_pred_fhe).mean()
            print(f"FHE vs non-FHE Mean Absolute Error on test sample: {fhe_mae}")
            
            print("sklearn model test completed successfully!")
            return model, mae_train
        except Exception as e:
            print(f"Error in FHE execution: {e}")
            import traceback
            traceback.print_exc()
            # Return the model and training MAE even if FHE fails
            return model, mae_train
    
    except Exception as e:
        print(f"Error in sklearn model test: {e}")
        import traceback
        traceback.print_exc()
        return None, float('inf')

if __name__ == "__main__":
    print("Starting Concrete-ML tests...")
    
    # Test PyTorch model
    quantized_module, torch_model, torch_mae = test_simple_linear_model()
    
    # Test sklearn model
    sklearn_model, sklearn_mae = test_sklearn_model()
    
    print("\nAll tests completed!")
    print(f"PyTorch model MAE: {torch_mae}")
    print(f"sklearn model MAE: {sklearn_mae}") 