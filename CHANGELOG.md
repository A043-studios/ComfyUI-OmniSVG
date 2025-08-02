# Changelog

All notable changes to ComfyUI-OmniSVG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-02

### Added
- Initial release of ComfyUI-OmniSVG custom nodes
- **OmniSVG Model Loader** - Load and manage OmniSVG models
- **OmniSVG Text to SVG** - Generate SVG graphics from text descriptions
- **OmniSVG Image to SVG** - Convert images to SVG format
- **SVG to Image** - Convert SVG strings to ComfyUI IMAGE tensors
- **SVG Saver** - Save SVG files to disk
- Complete ComfyUI workflow integration
- Automated installation script
- Comprehensive documentation and examples
- Working example workflow (workflow_example.json)
- Full test suite for all nodes

### Features
- Support for OmniSVG-3B model (17GB VRAM)
- Text-to-SVG generation with configurable parameters
- Image-to-SVG conversion with preprocessing
- High-quality SVG rendering using CairoSVG
- Automatic model detection and loading
- Memory management and cleanup
- Error handling and validation
- ComfyUI-Manager compatibility

### Technical Details
- Custom data types: OMNISVG_MODEL, SVG_STRING
- Proper tensor format conversion for ComfyUI
- GPU memory optimization
- Fallback support for CPU inference
- Integration with Qwen2.5-VL-3B-Instruct base model

### Documentation
- Complete README with installation instructions
- Development status documentation
- Example workflows and usage guides
- Troubleshooting section
- API documentation for all nodes

## [Unreleased]

### Planned
- Support for OmniSVG-7B model when available
- Batch processing capabilities
- Additional SVG manipulation nodes
- Performance optimizations
- ComfyUI-Manager registry integration
