# OmniSVG Core Module
# Adapted for ComfyUI integration

import os
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import core components
try:
    from .decoder import SketchDecoder
    from .tokenizer import SVGTokenizer
    
    __all__ = ['SketchDecoder', 'SVGTokenizer']
    
except ImportError as e:
    print(f"Warning: Could not import OmniSVG core components: {e}")
    __all__ = []
