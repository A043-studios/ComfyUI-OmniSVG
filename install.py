#!/usr/bin/env python3
"""
Installation script for ComfyUI-OmniSVG
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ])
            print("✓ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"✗ Error installing dependencies: {e}")
            return False
    else:
        print("✗ requirements.txt not found!")
        return False
    
    return True

def find_comfyui_models_dir():
    """Find ComfyUI models directory"""
    current_path = Path(__file__).parent
    
    # Walk up to find ComfyUI root
    while current_path.parent != current_path:
        models_dir = current_path / "models"
        if models_dir.exists():
            return models_dir
        current_path = current_path.parent
    
    # Fallback to creating models directory
    fallback_dir = Path(__file__).parent.parent.parent / "models"
    fallback_dir.mkdir(exist_ok=True)
    return fallback_dir

def setup_model_directory():
    """Setup OmniSVG models directory"""
    print("Setting up models directory...")
    
    models_dir = find_comfyui_models_dir()
    omnisvg_models_dir = models_dir / "omnisvg"
    omnisvg_models_dir.mkdir(exist_ok=True)
    
    print(f"✓ Models directory created at: {omnisvg_models_dir}")
    
    # Check if models already exist in storage
    storage_model_path = Path("/mnt/storage/OmniSVG-3B")
    if storage_model_path.exists():
        print(f"✓ Found existing OmniSVG model at: {storage_model_path}")
        print("  The nodes will automatically use this model.")
        return True
    
    # Check if models exist in the models directory
    model_path = omnisvg_models_dir / "OmniSVG-3B"
    if model_path.exists() and (model_path / "config.yaml").exists():
        print(f"✓ Found existing OmniSVG model at: {model_path}")
        return True
    
    print("⚠ OmniSVG model not found!")
    print("Please download the model using one of these methods:")
    print("")
    print("Method 1: Using huggingface-cli")
    print(f"  huggingface-cli download OmniSVG/OmniSVG --local-dir {model_path}")
    print("")
    print("Method 2: Manual download")
    print("  1. Go to https://huggingface.co/OmniSVG/OmniSVG")
    print(f"  2. Download all files to: {model_path}")
    print("")
    
    return False

def verify_installation():
    """Verify the installation"""
    print("Verifying installation...")
    
    # Check if core files exist
    core_dir = Path(__file__).parent / "omnisvg_core"
    required_files = ["decoder.py", "tokenizer.py", "deepsvg"]
    
    missing_files = []
    for file_name in required_files:
        file_path = core_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"✗ Missing core files: {missing_files}")
        print("Please ensure OmniSVG source files are available.")
        return False
    
    print("✓ Core files verified!")
    
    # Try importing key dependencies
    try:
        import torch
        import transformers
        import cairosvg
        print("✓ Key dependencies verified!")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True

def main():
    """Main installation function"""
    print("=" * 50)
    print("ComfyUI-OmniSVG Installation")
    print("=" * 50)
    
    success = True
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Setup model directory
    if not setup_model_directory():
        print("⚠ Model setup incomplete - you'll need to download models manually")
    
    # Verify installation
    if not verify_installation():
        success = False
    
    print("=" * 50)
    if success:
        print("✓ Installation completed successfully!")
        print("")
        print("Next steps:")
        print("1. Restart ComfyUI")
        print("2. Look for OmniSVG nodes in the node menu")
        print("3. Load the example workflow from workflow_example.json")
    else:
        print("✗ Installation completed with warnings")
        print("Please check the messages above and resolve any issues.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
