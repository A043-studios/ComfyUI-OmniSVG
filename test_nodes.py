#!/usr/bin/env python3
"""
Test script for ComfyUI-OmniSVG nodes
"""

import sys
import os
import torch
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def test_model_loader():
    """Test OmniSVG Model Loader"""
    print("Testing OmniSVG Model Loader...")
    
    try:
        from nodes import OmniSVGModelLoader
        
        loader = OmniSVGModelLoader()
        
        # Get available models
        input_types = loader.INPUT_TYPES()
        available_models = input_types["required"]["model_name"][0]
        print(f"Available models: {available_models}")
        
        if "Download Required" not in available_models:
            # Test loading a model
            model_name = available_models[0]
            print(f"Loading model: {model_name}")
            
            result = loader.load_model(model_name)
            model_bundle = result[0]
            
            print(f"✓ Model loaded successfully!")
            print(f"  Device: {model_bundle.device}")
            print(f"  Loaded: {model_bundle.loaded}")
            
            return model_bundle
        else:
            print("⚠ No models available for testing")
            return None
            
    except Exception as e:
        print(f"✗ Error testing model loader: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_text_to_svg(model_bundle):
    """Test Text to SVG generation"""
    if model_bundle is None:
        print("⚠ Skipping Text to SVG test - no model available")
        return
    
    print("\nTesting Text to SVG...")
    
    try:
        from nodes import OmniSVGTextToSVG
        
        text_node = OmniSVGTextToSVG()
        
        # Test with simple prompt
        prompt = "A simple red circle"
        print(f"Generating SVG for: '{prompt}'")
        
        result = text_node.generate_svg_from_text(
            omnisvg_model=model_bundle,
            text_prompt=prompt,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.05
        )
        
        svg_code, preview_image = result
        
        if svg_code and not svg_code.startswith("Error"):
            print("✓ Text to SVG generation successful!")
            print(f"  SVG length: {len(svg_code)} characters")
            print(f"  Preview image shape: {preview_image.shape}")
            
            # Save test SVG
            test_svg_path = current_dir / "test_output.svg"
            with open(test_svg_path, 'w') as f:
                f.write(svg_code)
            print(f"  Test SVG saved to: {test_svg_path}")
            
            return svg_code
        else:
            print(f"✗ Text to SVG failed: {svg_code}")
            return None
            
    except Exception as e:
        print(f"✗ Error testing text to SVG: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_svg_to_image(svg_code):
    """Test SVG to Image conversion"""
    if svg_code is None:
        print("\nTesting SVG to Image with sample SVG...")
        # Use a simple test SVG
        svg_code = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="40" fill="red" />
        </svg>'''
    else:
        print("\nTesting SVG to Image...")

    try:
        from nodes import SVGToImage

        svg_node = SVGToImage()

        result = svg_node.svg_to_image(
            svg_code=svg_code,
            width=512,
            height=512
        )

        image_tensor = result[0]

        print("✓ SVG to Image conversion successful!")
        print(f"  Image tensor shape: {image_tensor.shape}")
        print(f"  Image tensor range: {image_tensor.min():.3f} - {image_tensor.max():.3f}")

        return image_tensor

    except Exception as e:
        print(f"✗ Error testing SVG to Image: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_svg_saver(svg_code):
    """Test SVG Saver"""
    if svg_code is None:
        print("\nTesting SVG Saver with sample SVG...")
        # Use a simple test SVG
        svg_code = '''<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="40" fill="blue" />
        </svg>'''
    else:
        print("\nTesting SVG Saver...")

    try:
        from nodes import SVGSaver

        saver = SVGSaver()

        result = saver.save_svg(
            svg_code=svg_code,
            filename="test_saved_svg",
            output_dir=str(current_dir)
        )

        file_path = result[0]

        if os.path.exists(file_path):
            print("✓ SVG Saver successful!")
            print(f"  File saved to: {file_path}")
        else:
            print(f"✗ SVG file not found: {file_path}")

    except Exception as e:
        print(f"✗ Error testing SVG Saver: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("=" * 60)
    print("ComfyUI-OmniSVG Node Testing")
    print("=" * 60)
    
    # Test model loader
    model_bundle = test_model_loader()
    
    # Test text to SVG
    svg_code = test_text_to_svg(model_bundle)
    
    # Test SVG to image
    test_svg_to_image(svg_code)
    
    # Test SVG saver
    test_svg_saver(svg_code)
    
    # Cleanup
    if model_bundle:
        print("\nCleaning up...")
        model_bundle.unload_models()
        print("✓ Models unloaded")
    
    print("=" * 60)
    print("Testing completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
