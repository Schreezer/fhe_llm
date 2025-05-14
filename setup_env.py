"""
Setup script to install necessary packages for FHE simulation
"""

import subprocess
import sys

def install_packages():
    """Install necessary packages for FHE simulation"""
    packages = [
        "torch",
        "numpy",
        "safetensors",
    ]
    
    print("Installing required packages...")
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All packages installed successfully!")

if __name__ == "__main__":
    install_packages() 