# Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Test Installation

```bash
python test_imports.py
```

This should print "All tests passed!" if everything is set up correctly.

## Step 3: Create Pretrained Model

```bash
python create_pretrained_model.py
```

This creates a pretrained model checkpoint (with random weights) for immediate testing.

## Step 4: Run Demo

```bash
python demo.py
```

This will:
1. Create a sample cat sketch
2. Convert it to a painting using the pretrained model
3. Save the result as `demo_output.png`

## Step 5: Train Your Own Model (Optional)

1. Prepare your dataset:
   ```
   data/
     train/
       sketches/    # Your grayscale sketches
       paintings/   # Your colored paintings (paired)
   ```

2. Start training:
   ```bash
   python train.py
   ```

3. Checkpoints will be saved in `checkpoints/` directory
4. Sample outputs will be saved in `outputs/` directory

## Step 6: Use Your Own Images

```bash
python inference.py --input your_sketch.png --output result.png
```

## Troubleshooting

- **CUDA errors**: Make sure you have PyTorch with CUDA support installed
- **Import errors**: Run `pip install -r requirements.txt` again
- **Dataset errors**: Ensure sketch and painting images have matching filenames

## Next Steps

- Read `README.md` for detailed documentation
- Modify `config.py` to adjust training parameters
- Experiment with different loss functions (L1 vs Perceptual)
