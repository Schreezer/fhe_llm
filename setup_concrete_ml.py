"""
Script to set up a proper Python package structure for Concrete-ML
"""

import os
import sys
import shutil
import importlib

# Path to the concrete-ml source directory
concrete_ml_src = '/Users/chirag13/development/ai_project/concrete-ml/src'

# Path to the site-packages directory
site_packages = '/Users/chirag13/development/ai_project/fhe_llama_env_py310/lib/python3.10/site-packages'

# Create __init__.py files for all directories in the concrete module
def create_init_files(directory):
    """Create __init__.py files for all directories in the given directory."""
    for root, dirs, files in os.walk(directory):
        # Skip __pycache__ directories
        if '__pycache__' in root:
            continue
            
        # Create __init__.py file if it doesn't exist
        init_file = os.path.join(root, '__init__.py')
        if not os.path.exists(init_file):
            print(f"Creating {init_file}...")
            with open(init_file, 'w') as f:
                f.write('# This file was created by setup_concrete_ml.py\n')

# Create __init__.py file for concrete directory
concrete_dir = os.path.join(site_packages, 'concrete')
init_file = os.path.join(concrete_dir, '__init__.py')
if not os.path.exists(init_file):
    print(f"Creating {init_file}...")
    with open(init_file, 'w') as f:
        f.write('# This file was created by setup_concrete_ml.py\n')
    print(f"File created successfully.")

# Create __init__.py files for all directories in the concrete module
create_init_files(concrete_dir)

# Copy a directory and its files from source to destination
def copy_directory(src_dir, dest_dir):
    """Copy a directory and its files from source to destination."""
    print(f"Copying {src_dir} to {dest_dir}...")
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dest_item = os.path.join(dest_dir, item)
        
        # Skip __pycache__ directories
        if item == '__pycache__':
            continue
            
        # Copy files
        if os.path.isfile(src_item):
            shutil.copy2(src_item, dest_item)
            print(f"Copied {src_item} to {dest_item}")
        # Recursively copy subdirectories
        elif os.path.isdir(src_item):
            copy_directory(src_item, dest_item)

# List of directories to copy
directories_to_copy = [
    "torch",
    "sklearn",
    "common",
    "deployment",
    "quantization",
    "onnx",
    "compilation",
    "pytest_plugin",
]

# Copy all required directories
for directory in directories_to_copy:
    src_dir = os.path.join(concrete_ml_src, 'concrete', 'ml', directory)
    dest_dir = os.path.join(site_packages, 'concrete', 'ml', directory)
    
    if os.path.exists(src_dir):
        copy_directory(src_dir, dest_dir)
    else:
        print(f"Warning: Directory {src_dir} does not exist, skipping...")

# Copy individual files at the concrete.ml level
individual_files = ["version.py"]
for file in individual_files:
    src_file = os.path.join(concrete_ml_src, 'concrete', 'ml', file)
    dest_file = os.path.join(site_packages, 'concrete', 'ml', file)
    
    if os.path.exists(src_file):
        shutil.copy2(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")
    else:
        print(f"Warning: File {src_file} does not exist, skipping...")

print("\nSetup complete. Now try importing the modules again.") 