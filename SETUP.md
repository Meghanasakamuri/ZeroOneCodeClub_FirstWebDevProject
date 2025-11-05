# Setup Instructions for PyCharm

## Prerequisites

1. **Python 3.8 or higher** (Python 3.9+ recommended)
2. **PyCharm IDE** (Community or Professional)
3. **CUDA-capable GPU** (optional but recommended for training)

## Step-by-Step Setup

### 1. Open Project in PyCharm

1. Open PyCharm
2. File → Open → Select the `/workspace` directory
3. Wait for PyCharm to index the project

### 2. Configure Python Interpreter

1. File → Settings → Project → Python Interpreter
2. Click the gear icon → Add Interpreter
3. Create a new virtual environment or use existing one
4. Make sure Python 3.8+ is selected

### 3. Install Dependencies

In PyCharm terminal or using pip:

```bash
pip install -r requirements.txt
```

Or in PyCharm:
1. Right-click on `requirements.txt`
2. Select "Install All Packages"

### 4. Verify Installation

Run the test script:
```bash
python test_imports.py
```

Or in PyCharm:
1. Right-click on `test_imports.py`
2. Select "Run 'test_imports'"

### 5. Create Pretrained Model

Run:
```bash
python create_pretrained_model.py
```

### 6. Test with Demo

Run:
```bash
python demo.py
```

This will create a sample sketch and convert it to a painting.

## Project Structure

```
workspace/
├── models/              # Model architectures
│   ├── generator.py     # U-Net generator
│   └── discriminator.py # PatchGAN discriminator
├── utils/               # Utilities
│   ├── dataset.py       # Dataset loading
│   ├── losses.py        # Loss functions
│   └── utils.py         # Helper functions
├── config.py            # Configuration
├── train.py             # Training script
├── inference.py         # Inference script
├── demo.py              # Demo script
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Running in PyCharm

### Training

1. Open `train.py`
2. Click Run → Run 'train'
3. Or use the green play button

**Note**: Make sure you have training data in:
- `data/train/sketches/`
- `data/train/paintings/`

### Inference

1. Open `inference.py`
2. Configure arguments in Run → Edit Configurations:
   - Script: `inference.py`
   - Parameters: `--input path/to/sketch.png --output output.png`
3. Run

### Using Run Configurations

1. Run → Edit Configurations
2. Click "+" → Python
3. Configure:
   - Script path: `train.py` or `inference.py`
   - Parameters: (for inference) `--input sketch.png --output result.png`
   - Working directory: `/workspace`

## Troubleshooting

### Import Errors

- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that PyCharm is using the correct Python interpreter
- Invalidate caches: File → Invalidate Caches / Restart

### CUDA Errors

- If you get CUDA errors, the code will automatically fall back to CPU
- To force CPU: Edit `config.py` and set `device = torch.device('cpu')`

### Dataset Errors

- Ensure sketch and painting images have matching filenames
- Check that images are in PNG or JPG format
- Verify directory structure matches expected layout

### Memory Errors

- Reduce `batch_size` in `config.py`
- Reduce `image_size` in `config.py`
- Use smaller images

## Next Steps

1. Read `README.md` for detailed documentation
2. Read `QUICKSTART.md` for quick start guide
3. Modify `config.py` to experiment with different settings
4. Prepare your training dataset

## Tips for PyCharm

- Use PyCharm's built-in terminal for running commands
- Use the debugger to step through training code
- Use the profiler to optimize performance
- Set breakpoints in key functions to debug

## Contact

For issues or questions, refer to the README.md file.
