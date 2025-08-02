"""
ComfyUI OmniSVG Custom Nodes
Generate SVG graphics from text descriptions and images using OmniSVG
"""

import os
import sys
import shutil
from pathlib import Path

# Add the current directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def setup_omnisvg_core():
    """Setup OmniSVG core files on first import"""
    omnisvg_core_dir = current_dir / "omnisvg_core"
    
    if not omnisvg_core_dir.exists():
        print("Setting up OmniSVG core files...")
        omnisvg_core_dir.mkdir(exist_ok=True)
        
        # Copy essential files from the OmniSVG installation
        source_dir = Path("/root/ComfyUI/OmniSVG")
        
        if source_dir.exists():
            # Copy core files
            files_to_copy = [
                "decoder.py",
                "tokenizer.py",
                "deepsvg",
            ]
            
            for file_name in files_to_copy:
                source_path = source_dir / file_name
                dest_path = omnisvg_core_dir / file_name
                
                if source_path.exists():
                    if source_path.is_dir():
                        if dest_path.exists():
                            shutil.rmtree(dest_path)
                        shutil.copytree(source_path, dest_path)
                    else:
                        shutil.copy2(source_path, dest_path)
            
            # Create __init__.py for the core module
            init_file = omnisvg_core_dir / "__init__.py"
            with open(init_file, 'w') as f:
                f.write("# OmniSVG Core Module\n")
            
            print("OmniSVG core files setup complete!")
        else:
            print("Warning: OmniSVG source directory not found. Please ensure OmniSVG is installed.")

# Setup core files
setup_omnisvg_core()

# Import nodes
try:
    from .nodes import (
        OmniSVGModelLoader,
        OmniSVGTextToSVG,
        OmniSVGImageToSVG,
        SVGToImage,
        SVGSaver
    )
    
    # Node mappings for ComfyUI
    NODE_CLASS_MAPPINGS = {
        "OmniSVG Model Loader": OmniSVGModelLoader,
        "OmniSVG Text to SVG": OmniSVGTextToSVG,
        "OmniSVG Image to SVG": OmniSVGImageToSVG,
        "SVG to Image": SVGToImage,
        "SVG Saver": SVGSaver,
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "OmniSVG Model Loader": "🔧 OmniSVG Model Loader",
        "OmniSVG Text to SVG": "📝 OmniSVG Text to SVG",
        "OmniSVG Image to SVG": "🖼️ OmniSVG Image to SVG", 
        "SVG to Image": "🔄 SVG to Image",
        "SVG Saver": "💾 SVG Saver",
    }
    
    print("ComfyUI-OmniSVG nodes loaded successfully!")
    
except Exception as e:
    print(f"Error loading ComfyUI-OmniSVG nodes: {e}")
    import traceback
    traceback.print_exc()
    
    # Fallback empty mappings
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# Optional JavaScript extensions directory
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
