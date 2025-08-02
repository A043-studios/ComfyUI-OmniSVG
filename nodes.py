"""
ComfyUI OmniSVG Nodes Implementation
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageOps
import cairosvg
import io
import yaml
import gc
from pathlib import Path
import tempfile
# Try to import folder_paths, create mock if not available
try:
    import folder_paths
except ImportError:
    # Create mock folder_paths for standalone testing
    class MockFolderPaths:
        def __init__(self):
            self.models_dir = "/root/ComfyUI/models"
            self.output_directory = "/root/ComfyUI/output"

    folder_paths = MockFolderPaths()

# Import OmniSVG components
SketchDecoder = None
SVGTokenizer = None

try:
    # Try relative import first (for ComfyUI)
    from .omnisvg_core.decoder import SketchDecoder
    from .omnisvg_core.tokenizer import SVGTokenizer
except ImportError:
    try:
        # Try absolute import (for standalone testing)
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'omnisvg_core'))
        from decoder import SketchDecoder
        from tokenizer import SVGTokenizer
    except ImportError as e:
        print(f"Warning: Could not import OmniSVG core components: {e}")
        print("Please ensure OmniSVG is properly installed and core files are copied.")

# Import transformers components
try:
    from transformers import AutoTokenizer, AutoProcessor
except ImportError as e:
    print(f"Warning: Could not import transformers: {e}")
    AutoTokenizer = None
    AutoProcessor = None

# Import qwen_vl_utils
try:
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    print(f"Warning: Could not import qwen_vl_utils: {e}")
    process_vision_info = None

# Custom data type for OmniSVG model bundle
class OmniSVGModelBundle:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.processor = None
        self.sketch_decoder = None
        self.svg_tokenizer = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
    
    def load_models(self):
        """Load all OmniSVG models"""
        if self.loaded:
            return

        print(f"Loading OmniSVG models from: {self.model_path}")

        # Check if required components are available
        if AutoTokenizer is None or AutoProcessor is None:
            raise ImportError("transformers library not available")
        if SketchDecoder is None or SVGTokenizer is None:
            raise ImportError("OmniSVG core components not available")

        # Load config
        config_path = os.path.join(self.model_path, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.yaml not found in {self.model_path}")

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Load Qwen models
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left")
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", padding_side="left")

        # Load sketch decoder
        self.sketch_decoder = SketchDecoder()
        sketch_weight_file = os.path.join(self.model_path, "pytorch_model.bin")
        if not os.path.exists(sketch_weight_file):
            raise FileNotFoundError(f"pytorch_model.bin not found in {self.model_path}")

        self.sketch_decoder.load_state_dict(torch.load(sketch_weight_file, map_location=self.device))
        self.sketch_decoder = self.sketch_decoder.to(self.device).eval()

        # Load SVG tokenizer
        self.svg_tokenizer = SVGTokenizer(config_path)

        self.loaded = True
        print("OmniSVG models loaded successfully!")
    
    def unload_models(self):
        """Unload models to free memory"""
        if hasattr(self, 'sketch_decoder') and self.sketch_decoder is not None:
            del self.sketch_decoder
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
        if hasattr(self, 'processor') and self.processor is not None:
            del self.processor
        if hasattr(self, 'svg_tokenizer') and self.svg_tokenizer is not None:
            del self.svg_tokenizer
        
        self.loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Utility functions
def get_omnisvg_models_dir():
    """Get the OmniSVG models directory"""
    models_dir = folder_paths.models_dir if hasattr(folder_paths, 'models_dir') else os.path.join(os.path.dirname(__file__), "..", "..", "models")
    omnisvg_dir = os.path.join(models_dir, "omnisvg")
    os.makedirs(omnisvg_dir, exist_ok=True)
    return omnisvg_dir

def find_omnisvg_models():
    """Find available OmniSVG models"""
    models_dir = get_omnisvg_models_dir()
    available_models = []
    
    if os.path.exists(models_dir):
        for model_dir in os.listdir(models_dir):
            model_path = os.path.join(models_dir, model_dir)
            if os.path.isdir(model_path):
                # Check for required files
                required_files = ["config.yaml", "pytorch_model.bin"]
                if all(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                    available_models.append(model_dir)
    
    # Fallback to storage location
    storage_path = "/mnt/storage/OmniSVG-3B"
    if not available_models and os.path.exists(storage_path):
        available_models.append("OmniSVG-3B (from storage)")
    
    return available_models if available_models else ["Download Required"]

def comfyui_to_pil(image_tensor):
    """Convert ComfyUI IMAGE tensor to PIL Image"""
    # ComfyUI format: [B,H,W,C] with values 0-1
    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor[0]  # Take first batch
    
    # Convert to numpy and scale to 0-255
    image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(image_np)

def pil_to_comfyui(pil_image):
    """Convert PIL Image to ComfyUI IMAGE tensor"""
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Convert to numpy array and normalize to 0-1
    image_np = np.array(pil_image).astype(np.float32) / 255.0
    
    # Add batch dimension: [H,W,C] -> [B,H,W,C]
    image_tensor = torch.from_numpy(image_np)[None,]
    
    return image_tensor

def svg_to_pil(svg_string):
    """Convert SVG string to PIL Image"""
    try:
        png_data = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
        return Image.open(io.BytesIO(png_data))
    except Exception as e:
        print(f"Error converting SVG to PIL: {e}")
        # Return a placeholder image
        placeholder = Image.new('RGB', (512, 512), color='white')
        return placeholder

# System prompt for OmniSVG
SYSTEM_PROMPT = "You are a multimodal SVG generation assistant capable of generating SVG code from both text descriptions and images."

class OmniSVGModelLoader:
    """Load OmniSVG models"""
    
    @classmethod
    def INPUT_TYPES(cls):
        available_models = find_omnisvg_models()
        
        return {
            "required": {
                "model_name": (available_models, {"default": available_models[0]}),
            }
        }
    
    RETURN_TYPES = ("OMNISVG_MODEL",)
    RETURN_NAMES = ("omnisvg_model",)
    CATEGORY = "OmniSVG"
    FUNCTION = "load_model"
    
    def load_model(self, model_name):
        """Load the OmniSVG model"""
        try:
            if model_name == "Download Required":
                raise ValueError("Please download OmniSVG models first. Place them in ComfyUI/models/omnisvg/")
            
            # Determine model path
            if "from storage" in model_name:
                model_path = "/mnt/storage/OmniSVG-3B"
            else:
                models_dir = get_omnisvg_models_dir()
                model_path = os.path.join(models_dir, model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at: {model_path}")
            
            # Create model bundle
            model_bundle = OmniSVGModelBundle(model_path)
            model_bundle.load_models()
            
            return (model_bundle,)
            
        except Exception as e:
            print(f"Error loading OmniSVG model: {e}")
            raise e


class OmniSVGTextToSVG:
    """Generate SVG from text description"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omnisvg_model": ("OMNISVG_MODEL",),
                "text_prompt": ("STRING", {
                    "default": "A simple icon of a house with a red roof and blue door",
                    "multiline": True
                }),
                "temperature": ("FLOAT", {
                    "default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.1, "max": 1.0, "step": 0.05
                }),
                "top_k": ("INT", {
                    "default": 50, "min": 1, "max": 100, "step": 1
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("SVG_STRING", "IMAGE")
    RETURN_NAMES = ("svg_code", "preview_image")
    CATEGORY = "OmniSVG"
    FUNCTION = "generate_svg_from_text"

    def generate_svg_from_text(self, omnisvg_model, text_prompt, temperature, top_p, top_k, repetition_penalty):
        """Generate SVG from text description"""
        try:
            if not omnisvg_model.loaded:
                omnisvg_model.load_models()

            # Process text input
            messages = [{
                "role": "system",
                "content": SYSTEM_PROMPT
            }, {
                "role": "user",
                "content": f"Task: text-to-svg\nGenerate SVG code for: {text_prompt}"
            }]

            text_input = omnisvg_model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = omnisvg_model.processor(
                text=[text_input],
                truncation=True,
                return_tensors="pt"
            )

            input_ids = inputs['input_ids'].to(omnisvg_model.device)
            attention_mask = inputs['attention_mask'].to(omnisvg_model.device)

            # Generate SVG
            svg_code, png_image = self._generate_svg(
                omnisvg_model, input_ids, attention_mask, None, None,
                temperature, top_p, top_k, repetition_penalty, "text-to-svg"
            )

            if svg_code and not svg_code.startswith("Error"):
                # Convert PNG to ComfyUI format
                comfyui_image = pil_to_comfyui(png_image)
                return (svg_code, comfyui_image)
            else:
                # Return error placeholder
                placeholder = Image.new('RGB', (512, 512), color='red')
                comfyui_image = pil_to_comfyui(placeholder)
                return (f"Error: {svg_code}", comfyui_image)

        except Exception as e:
            print(f"Error in text to SVG generation: {e}")
            import traceback
            traceback.print_exc()

            # Return error placeholder
            placeholder = Image.new('RGB', (512, 512), color='red')
            comfyui_image = pil_to_comfyui(placeholder)
            return (f"Error: {str(e)}", comfyui_image)

    def _generate_svg(self, model_bundle, input_ids, attention_mask, pixel_values, image_grid_thw,
                     temperature, top_p, top_k, repetition_penalty, task_type):
        """Core SVG generation function"""
        try:
            # Clean memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            print(f"Generating SVG for {task_type}...")

            # Generation configuration based on task type
            if task_type == "image-to-svg":
                gen_config = {
                    "do_sample": True,
                    "temperature": 0.1,
                    "top_p": 0.001,
                    "top_k": 1,
                    "repetition_penalty": 1.05,
                    "early_stopping": True,
                }
            else:
                gen_config = {
                    "do_sample": True,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
                    "early_stopping": True,
                }

            # Get model configuration
            model_config = model_bundle.config['model']
            max_length = model_config['max_length']

            # Initialize output_ids properly (like original implementation)
            output_ids = torch.ones(1, max_length + 1).long().to(model_bundle.device) * model_config['eos_token_id']

            # Generate tokens
            with torch.no_grad():
                results = model_bundle.sketch_decoder.transformer.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    bos_token_id=model_config['bos_token_id'],
                    eos_token_id=model_config['eos_token_id'],
                    pad_token_id=model_config['pad_token_id'],
                    use_cache=True,
                    **gen_config
                )

                results = results[:, :max_length]
                output_ids[:, :results.shape[1]] = results

                # Process generated tokens
                generated_xy, generated_colors = model_bundle.svg_tokenizer.process_generated_tokens(output_ids)
                print(f"Generated {len(generated_colors)} colors")

            print('Rendering...')
            # Convert to SVG tensors
            svg_tensors = model_bundle.svg_tokenizer.raster_svg(generated_xy)

            if not svg_tensors or not svg_tensors[0]:
                return "Error: No valid SVG paths generated", None

            print('Creating SVG...')
            # Apply colors and create SVG
            svg = model_bundle.svg_tokenizer.apply_colors_to_svg(svg_tensors[0], generated_colors)
            svg_str = svg.to_str()

            # Convert to PNG for preview
            png_image = svg_to_pil(svg_str)

            return svg_str, png_image

        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {e}", None


class OmniSVGImageToSVG:
    """Convert image to SVG"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "omnisvg_model": ("OMNISVG_MODEL",),
                "image": ("IMAGE",),
                "target_size": ("INT", {
                    "default": 200, "min": 64, "max": 512, "step": 8
                }),
                "temperature": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01
                }),
                "top_p": ("FLOAT", {
                    "default": 0.001, "min": 0.001, "max": 1.0, "step": 0.001
                }),
                "top_k": ("INT", {
                    "default": 1, "min": 1, "max": 100, "step": 1
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05
                }),
            }
        }

    RETURN_TYPES = ("SVG_STRING", "IMAGE")
    RETURN_NAMES = ("svg_code", "preview_image")
    CATEGORY = "OmniSVG"
    FUNCTION = "generate_svg_from_image"

    def generate_svg_from_image(self, omnisvg_model, image, target_size, temperature, top_p, top_k, repetition_penalty):
        """Generate SVG from image"""
        try:
            if not omnisvg_model.loaded:
                omnisvg_model.load_models()

            # Convert ComfyUI image to PIL and resize
            pil_image = comfyui_to_pil(image)
            pil_image = pil_image.resize((target_size, target_size), Image.Resampling.LANCZOS)

            # Save to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                pil_image.save(tmp_file.name)
                temp_image_path = tmp_file.name

            try:
                # Process image input
                messages = [{
                    "role": "system",
                    "content": SYSTEM_PROMPT
                }, {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Task: image-to-svg\nGenerate SVG code that accurately represents the following image."},
                        {"type": "image", "image": temp_image_path},
                    ]
                }]

                text_input = omnisvg_model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)

                inputs = omnisvg_model.processor(
                    text=[text_input],
                    images=image_inputs,
                    truncation=True,
                    return_tensors="pt"
                )

                input_ids = inputs['input_ids'].to(omnisvg_model.device)
                attention_mask = inputs['attention_mask'].to(omnisvg_model.device)
                pixel_values = inputs['pixel_values'].to(omnisvg_model.device) if 'pixel_values' in inputs else None
                image_grid_thw = inputs['image_grid_thw'].to(omnisvg_model.device) if 'image_grid_thw' in inputs else None

                # Generate SVG using the same method as text-to-svg
                text_to_svg_node = OmniSVGTextToSVG()
                svg_code, png_image = text_to_svg_node._generate_svg(
                    omnisvg_model, input_ids, attention_mask, pixel_values, image_grid_thw,
                    temperature, top_p, top_k, repetition_penalty, "image-to-svg"
                )

                if svg_code and not svg_code.startswith("Error"):
                    # Convert PNG to ComfyUI format
                    comfyui_image = pil_to_comfyui(png_image)
                    return (svg_code, comfyui_image)
                else:
                    # Return error placeholder
                    placeholder = Image.new('RGB', (512, 512), color='red')
                    comfyui_image = pil_to_comfyui(placeholder)
                    return (f"Error: {svg_code}", comfyui_image)

            finally:
                # Clean up temporary file
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)

        except Exception as e:
            print(f"Error in image to SVG generation: {e}")
            import traceback
            traceback.print_exc()

            # Return error placeholder
            placeholder = Image.new('RGB', (512, 512), color='red')
            comfyui_image = pil_to_comfyui(placeholder)
            return (f"Error: {str(e)}", comfyui_image)


class SVGToImage:
    """Convert SVG string to ComfyUI IMAGE"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_code": ("SVG_STRING",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "OmniSVG"
    FUNCTION = "svg_to_image"

    def svg_to_image(self, svg_code, width, height):
        """Convert SVG to ComfyUI IMAGE tensor"""
        try:
            # Convert SVG to PNG with specified dimensions
            png_data = cairosvg.svg2png(
                bytestring=svg_code.encode('utf-8'),
                output_width=width,
                output_height=height
            )

            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(png_data))

            # Convert to ComfyUI format
            comfyui_image = pil_to_comfyui(pil_image)

            return (comfyui_image,)

        except Exception as e:
            print(f"Error converting SVG to image: {e}")
            # Return placeholder image
            placeholder = Image.new('RGB', (width, height), color='white')
            comfyui_image = pil_to_comfyui(placeholder)
            return (comfyui_image,)


class SVGSaver:
    """Save SVG to file"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "svg_code": ("SVG_STRING",),
                "filename": ("STRING", {"default": "generated_svg"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    CATEGORY = "OmniSVG"
    FUNCTION = "save_svg"
    OUTPUT_NODE = True

    def save_svg(self, svg_code, filename, output_dir=""):
        """Save SVG code to file"""
        try:
            # Determine output directory
            if not output_dir:
                # Use ComfyUI output directory
                if hasattr(folder_paths, 'output_directory'):
                    output_dir = folder_paths.output_directory
                else:
                    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")

            os.makedirs(output_dir, exist_ok=True)

            # Ensure filename has .svg extension
            if not filename.endswith('.svg'):
                filename += '.svg'

            # Create full file path
            file_path = os.path.join(output_dir, filename)

            # Save SVG file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(svg_code)

            print(f"SVG saved to: {file_path}")
            return (file_path,)

        except Exception as e:
            print(f"Error saving SVG: {e}")
            return (f"Error: {str(e)}",)
