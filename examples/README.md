# ComfyUI-OmniSVG Examples

This directory contains example workflows and usage patterns for ComfyUI-OmniSVG.

## 📁 Files

### `basic_workflow.json`
Complete workflow demonstrating all OmniSVG nodes:
- Model loading
- Text-to-SVG generation
- Image-to-SVG conversion
- SVG processing and saving
- Preview integration

## 🚀 How to Use Examples

1. **Load Workflow in ComfyUI**
   - Open ComfyUI
   - Click "Load" button
   - Select the workflow JSON file
   - All nodes will be automatically arranged

2. **Configure Parameters**
   - Adjust text prompts for different SVG styles
   - Modify generation parameters (temperature, top_p, etc.)
   - Set output directories and filenames

3. **Run Generation**
   - Click "Queue Prompt" to start generation
   - Monitor progress in the console
   - View results in preview nodes

## 💡 Example Prompts

### Text-to-SVG Prompts
```
"A minimalist icon of a coffee cup with steam"
"Simple geometric pattern with triangles and circles"
"Logo design for a tech company with clean lines"
"Weather icon showing sun and clouds"
"Navigation arrow pointing right"
```

### Image-to-SVG Tips
- Use simple, high-contrast images for best results
- Logos and icons work particularly well
- Resize images to 200-400px for optimal processing
- Black and white images often produce cleaner SVGs

## 🎯 Workflow Patterns

### Pattern 1: Text-to-SVG Pipeline
```
Model Loader → Text to SVG → SVG to Image → Preview
                     ↓
                 SVG Saver
```

### Pattern 2: Image-to-SVG Pipeline
```
Load Image → Model Loader → Image to SVG → SVG to Image → Preview
                                  ↓
                              SVG Saver
```

### Pattern 3: Batch Processing
```
Model Loader (shared)
    ↓
Multiple Text/Image to SVG nodes
    ↓
Multiple SVG processors
```

## 🔧 Customization

### Generation Parameters
- **Temperature**: Controls randomness (0.1-2.0)
  - Lower = more consistent
  - Higher = more creative
- **Top-p**: Nucleus sampling (0.1-1.0)
- **Top-k**: Token selection limit (1-100)
- **Repetition Penalty**: Prevents repetition (1.0-2.0)

### Output Settings
- **SVG Dimensions**: Modify in SVG to Image node
- **File Naming**: Customize in SVG Saver node
- **Output Directory**: Set custom paths

## 📊 Performance Tips

1. **GPU Memory**: Ensure 17GB+ VRAM available
2. **Batch Size**: Process one at a time for stability
3. **Model Loading**: Reuse loaded models across generations
4. **Memory Cleanup**: Restart ComfyUI if memory issues occur

## 🐛 Troubleshooting Examples

### Common Issues
1. **"Model not found"**: Check model path in loader
2. **"CUDA out of memory"**: Free GPU memory or use CPU
3. **"Generation failed"**: Try different parameters
4. **"SVG invalid"**: Check input image quality

### Debug Workflow
1. Load basic_workflow.json
2. Use simple text prompt: "red circle"
3. Check each node output
4. Verify model loading first

## 🎨 Creative Ideas

### Icon Sets
Create consistent icon sets by using similar prompts:
- "Minimalist home icon"
- "Minimalist settings icon"
- "Minimalist user icon"

### Logo Variations
Generate logo variations:
- "Tech company logo with geometric shapes"
- "Same logo but with rounded corners"
- "Same logo but monochrome"

### Pattern Generation
Create repeatable patterns:
- "Seamless geometric pattern"
- "Art deco border design"
- "Simple repeating motif"

## 📝 Notes

- All examples are tested and working
- Modify parameters to suit your needs
- Share your own workflows with the community
- Report issues or improvements on GitHub
