# Contributing to ComfyUI-OmniSVG

Thank you for your interest in contributing to ComfyUI-OmniSVG! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 2.3.0+
- CUDA 12.1+ (recommended)
- ComfyUI installation
- 17GB+ VRAM for model loading

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/ComfyUI-OmniSVG.git
   cd ComfyUI-OmniSVG
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run tests:
   ```bash
   python test_nodes.py
   ```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- **Clear description** of the problem
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, Python version, GPU, VRAM)
- **Error messages** and stack traces
- **ComfyUI version** and other relevant software versions

### Suggesting Features

Feature requests are welcome! Please include:

- **Clear description** of the feature
- **Use case** and motivation
- **Possible implementation** approach
- **Compatibility considerations** with ComfyUI

### Pull Requests

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards below

3. **Test your changes**:
   ```bash
   python test_nodes.py
   ```

4. **Update documentation** if needed

5. **Commit with clear messages**:
   ```bash
   git commit -m "Add: Brief description of changes"
   ```

6. **Push and create a pull request**

## Coding Standards

### Python Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings for all public functions and classes
- Keep functions focused and small
- Use type hints where appropriate

### ComfyUI Node Standards
- Follow ComfyUI node conventions
- Use proper INPUT_TYPES and RETURN_TYPES
- Include helpful parameter descriptions
- Add proper error handling
- Use appropriate CATEGORY names

### Example Node Structure
```python
class ExampleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "parameter": ("TYPE", {"default": value, "description": "Clear description"}),
            }
        }
    
    RETURN_TYPES = ("OUTPUT_TYPE",)
    RETURN_NAMES = ("output_name",)
    CATEGORY = "OmniSVG"
    FUNCTION = "process"
    
    def process(self, parameter):
        """Clear docstring describing the function"""
        try:
            # Implementation
            return (result,)
        except Exception as e:
            print(f"Error in {self.__class__.__name__}: {e}")
            raise e
```

## Testing

### Running Tests
```bash
# Run all tests
python test_nodes.py

# Test specific functionality
python -c "from test_nodes import test_model_loader; test_model_loader()"
```

### Adding Tests
When adding new features, include tests that cover:
- Normal operation
- Edge cases
- Error conditions
- Integration with other nodes

## Documentation

### Code Documentation
- Add docstrings to all public functions
- Include parameter descriptions
- Document return values
- Add usage examples for complex functions

### User Documentation
- Update README.md for new features
- Add examples to workflow_example.json
- Update CHANGELOG.md
- Include troubleshooting information

## Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Test thoroughly
4. Create release tag
5. Update documentation

## Community Guidelines

### Be Respectful
- Use welcoming and inclusive language
- Respect different viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what's best for the community

### Be Helpful
- Help newcomers get started
- Share knowledge and best practices
- Provide constructive feedback
- Collaborate effectively

## Questions?

If you have questions about contributing, please:
- Check existing documentation
- Search existing issues
- Create a new issue with the "question" label
- Join community discussions

Thank you for contributing to ComfyUI-OmniSVG!
