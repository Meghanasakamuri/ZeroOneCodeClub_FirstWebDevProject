# Setup Instructions for PyCharm

## 📋 Prerequisites

- **PyCharm** (Community or Professional Edition)
- **Python 3.8+** installed on your system
- **Git** (optional, for version control)
- **CUDA-capable GPU** (optional, but highly recommended)

---

## 🚀 Step-by-Step Setup in PyCharm

### Step 1: Open Project in PyCharm

1. Launch PyCharm
2. Click **"Open"** or **"File → Open"**
3. Navigate to `/workspace` (or wherever you saved the AutoPainter files)
4. Click **"OK"**

### Step 2: Configure Python Interpreter

1. Go to **"File → Settings"** (Windows/Linux) or **"PyCharm → Preferences"** (Mac)
2. Navigate to **"Project → Python Interpreter"**
3. Click the gear icon ⚙️ → **"Add"**
4. Choose **"Virtualenv Environment"** → **"New environment"**
5. Set location to: `workspace/venv`
6. Base interpreter: Select your Python 3.8+ installation
7. Click **"OK"**

### Step 3: Install Dependencies

**Option A: Using PyCharm Terminal**

1. Open terminal in PyCharm (**"View → Tool Windows → Terminal"**)
2. Run:
```bash
pip install -r requirements.txt
```

**Option B: Using PyCharm Package Manager**

1. Go to **"File → Settings → Project → Python Interpreter"**
2. Click **"+"** to add packages
3. Manually install each package from `requirements.txt`:
   - torch
   - torchvision
   - numpy
   - Pillow
   - matplotlib
   - tqdm
   - opencv-python
   - scikit-image
   - tensorboard
   - wget
   - gdown

### Step 4: Verify Installation

In PyCharm Terminal, run:
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.x.x
CUDA: True  (or False if no GPU)
```

---

## 🎯 Running the Project in PyCharm

### Quick Start (Recommended for First Time)

1. **Right-click on `run.py`** in Project Explorer
2. Select **"Run 'run'"**
3. In the interactive menu, choose:
   - Option 1: Setup
   - Option 2: Quick Train

### Prepare Dataset

1. **Right-click on `download_dataset.py`**
2. Select **"Run 'download_dataset'"**
3. When prompted, select **option 3** (Create synthetic dataset)
4. Enter `100` for number of samples
5. Wait for completion (~1-2 minutes)

### Train the Model

**Option A: Interactive Menu**
1. **Right-click on `run.py`** → **"Run 'run'"**
2. Choose option 2: Quick Train
3. Confirm when prompted

**Option B: Direct Training**
1. **Right-click on `train.py`** → **"Run 'train'"**
2. Training will start with default settings
3. Monitor progress in the console

**Option C: Configure Then Run**
1. Open `config.py`
2. Modify settings (e.g., `NUM_EPOCHS = 10` for quick test)
3. **Right-click on `train.py`** → **"Run 'train'"**

### Generate Paintings

**Option A: Create Sample Sketch**
1. **Right-click on `inference.py`**
2. Select **"Modify Run Configuration"**
3. In **"Parameters"** field, enter:
   ```
   --create-sample --checkpoint checkpoints/best_model.pth
   ```
4. Click **"Run"**

**Option B: Convert Your Own Sketch**
1. Place your sketch image in the workspace (e.g., `my_sketch.jpg`)
2. **Right-click on `inference.py`** → **"Modify Run Configuration"**
3. In **"Parameters"** field, enter:
   ```
   --input my_sketch.jpg --output my_painting.png
   ```
4. Click **"Run"**

**Option C: Batch Processing**
1. Create a folder with multiple sketches (e.g., `sketches/`)
2. **Right-click on `inference.py`** → **"Modify Run Configuration"**
3. In **"Parameters"** field, enter:
   ```
   --input sketches/ --output paintings/ --batch
   ```
4. Click **"Run"**

### Run Research Experiments

1. **Right-click on `experiments.py`** → **"Run 'experiments'"**
2. Choose experiment from menu:
   - Option 1: Quick test
   - Option 2: Loss comparison
   - Option 3: Data requirements
   - Option 4: Discriminator comparison

---

## 🔧 PyCharm Configuration Tips

### Enable Code Completion

1. PyCharm should automatically index the project
2. If not, go to **"File → Invalidate Caches → Invalidate and Restart"**

### Set Up Run Configurations

Create saved run configurations for common tasks:

**Quick Train Configuration:**
1. **"Run → Edit Configurations"** → **"+"** → **"Python"**
2. Name: `Quick Train`
3. Script path: `/workspace/train.py`
4. Before launch: Add custom config modification
5. Click **"OK"**

**Inference Configuration:**
1. **"Run → Edit Configurations"** → **"+"** → **"Python"**
2. Name: `Generate Painting`
3. Script path: `/workspace/inference.py`
4. Parameters: `--create-sample --checkpoint checkpoints/best_model.pth`
5. Click **"OK"**

### GPU Memory Issues

If you get CUDA out of memory errors:

1. Open `config.py`
2. Change `BATCH_SIZE = 16` to `BATCH_SIZE = 4` or `8`
3. Save and re-run

### View Training Progress

**Loss Plots:**
1. Navigate to: `logs/[experiment_name]/plots/`
2. Right-click on `training_losses.png`
3. Select **"Open with → Image Viewer"**

**Sample Images:**
1. Navigate to: `logs/[experiment_name]/samples/`
2. Browse generated images during training

**Checkpoints:**
1. Navigate to: `checkpoints/[experiment_name]/`
2. Find `best_model.pth` for inference

---

## 📁 Project Structure in PyCharm

```
workspace/
├── 📄 Core Implementation
│   ├── config.py          ⚙️ Configuration
│   ├── model.py           🧠 Generator & Discriminator
│   ├── dataset.py         📦 Data loading
│   ├── utils.py           🔧 Utilities
│   ├── train.py           🏋️ Training script
│   ├── inference.py       🎨 Generate paintings
│   ├── download_dataset.py 📥 Dataset preparation
│   ├── experiments.py     🔬 Research experiments
│   └── run.py             🚀 Interactive menu
│
├── 📚 Documentation
│   ├── README_AUTOPAINTER.md  📖 Full documentation
│   ├── QUICKSTART.md          ⚡ Quick start guide
│   ├── PROJECT_SUMMARY.md     📊 Project overview
│   └── SETUP_INSTRUCTIONS.md  🛠️ This file
│
├── 📦 Dependencies
│   └── requirements.txt       📝 Package list
│
└── 🗂️ Generated (after running)
    ├── data/              💾 Dataset
    ├── checkpoints/       💿 Saved models
    ├── logs/              📈 Training logs
    └── experiments/       🧪 Experiment results
```

---

## 🐛 Troubleshooting in PyCharm

### Issue: "No module named 'torch'"

**Solution:**
1. Check Python interpreter: **"File → Settings → Project → Python Interpreter"**
2. Verify packages are installed
3. Reinstall: `pip install -r requirements.txt` in PyCharm Terminal

### Issue: "Cannot find config module"

**Solution:**
1. Right-click on workspace folder
2. Select **"Mark Directory as → Sources Root"**

### Issue: Script runs but no output

**Solution:**
1. Check **"Run"** window at bottom of PyCharm
2. Look for error messages
3. Enable **"Run → Edit Configurations → Emulate terminal"**

### Issue: Import errors

**Solution:**
1. Ensure working directory is `/workspace`
2. Go to **"Run → Edit Configurations"**
3. Set **"Working directory"** to `/workspace`

### Issue: CUDA not detected

**Solution:**
1. Install CUDA toolkit from NVIDIA website
2. Reinstall PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

---

## ✅ Verification Checklist

Before training, verify in PyCharm:

- [ ] Python interpreter configured (3.8+)
- [ ] All packages from requirements.txt installed
- [ ] Working directory set to `/workspace`
- [ ] `data/monet_sketches/` exists with images
- [ ] No import errors when running `config.py`
- [ ] GPU detected (if available): Run test script

**Test Script** (Run in PyCharm Terminal):
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

from config import Config
print(f"\nConfig loaded: {Config.get_experiment_name()}")

from model import Generator
gen = Generator()
print(f"Generator loaded: {sum(p.numel() for p in gen.parameters()):,} params")

print("\n✅ All checks passed!")
```

---

## 🎯 Recommended Workflow in PyCharm

### First Time Setup (15 minutes)

1. ✅ Open project in PyCharm
2. ✅ Configure Python interpreter + virtualenv
3. ✅ Install requirements.txt
4. ✅ Run verification test
5. ✅ Run `download_dataset.py` (option 3, 100 samples)
6. ✅ Run quick train (10 epochs)
7. ✅ Generate sample painting

### Regular Development Workflow

**Morning: Start Training**
1. Modify `config.py` if needed
2. Run `train.py`
3. Monitor progress in console
4. Check samples in `logs/` folder

**Afternoon: Evaluate Results**
1. Run `inference.py` on test sketches
2. Review generated paintings
3. Analyze loss plots
4. Adjust hyperparameters if needed

**Evening: Experiments**
1. Run `experiments.py`
2. Compare different configurations
3. Document findings
4. Save best models

### Research Paper Workflow

**Week 1: Setup & Baseline**
- Day 1-2: Setup, prepare dataset
- Day 3-5: Train baseline model (L1 loss, PatchGAN)
- Day 6-7: Evaluate baseline, collect metrics

**Week 2: Experiments**
- Day 1-2: Loss comparison experiments
- Day 3-4: Data requirements study
- Day 5-6: Discriminator comparison
- Day 7: Compile results

**Week 3: Analysis**
- Day 1-3: Analyze all results
- Day 4-5: Generate figures and tables
- Day 6-7: Write research findings

---

## 🔍 Debugging in PyCharm

### Set Breakpoints

1. Click in left margin of code editor (next to line numbers)
2. Red dot appears = breakpoint set
3. **Right-click on script → "Debug 'script'"**
4. Execution pauses at breakpoint
5. Inspect variables in **"Variables"** panel

**Useful Places to Set Breakpoints:**
- `train.py` line ~100: Inside training loop
- `model.py` line ~50: Generator forward pass
- `dataset.py` line ~80: Dataset loading
- `inference.py` line ~60: Image generation

### Watch Variables

1. **"Run → View Breakpoints"**
2. Select breakpoint
3. Add **"Watches"** for specific variables
4. View values during debugging

### Profile Performance

1. **"Run → Profile 'script'"**
2. PyCharm shows performance metrics
3. Identify bottlenecks
4. Optimize slow functions

---

## 💻 Terminal Commands Quick Reference

All commands should be run in PyCharm Terminal:

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare dataset
python download_dataset.py

# Quick train (10 epochs)
python run.py --quick-train

# Full train (200 epochs)
python train.py

# Generate painting
python inference.py --create-sample

# Run experiments
python experiments.py

# Check GPU
python -c "import torch; print(torch.cuda.is_available())"

# List checkpoints
ls checkpoints/*/

# View training logs
cat logs/*/training.log
```

---

## 🎓 Learning Resources

**PyCharm Tutorials:**
- PyCharm Quick Start Guide: https://www.jetbrains.com/help/pycharm/quick-start-guide.html
- Python Debugger: https://www.jetbrains.com/help/pycharm/debugging-code.html

**PyTorch Resources:**
- PyTorch Tutorials: https://pytorch.org/tutorials/
- GANs in PyTorch: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

**AutoPainter Documentation:**
- Full Documentation: `README_AUTOPAINTER.md`
- Quick Start: `QUICKSTART.md`
- Project Overview: `PROJECT_SUMMARY.md`

---

## 🆘 Getting Help

### In Order of Preference:

1. **Check documentation** in this folder (README files)
2. **Review code comments** in Python files
3. **Check PyCharm built-in help**: **"Help → Find Action"**
4. **Examine error messages** in Run/Console window
5. **Use PyCharm debugger** to step through code

---

## ✨ You're Ready!

Everything is set up. Now you can:

1. **Train** your sketch-to-painting model
2. **Generate** beautiful Monet-style art
3. **Experiment** with research questions
4. **Deploy** for production use

**Start with**: `python run.py` in PyCharm Terminal

---

**Happy Coding in PyCharm! 🎨💻**
