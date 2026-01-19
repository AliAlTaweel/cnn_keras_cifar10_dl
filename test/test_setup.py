"""
Test if everything is set up correctly
Run this before training to check your setup
"""

import sys

print("=" * 50)
print("CIFAR-10 Setup Checker")
print("=" * 50)
print()

# Check Python version
print("1. Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor} - OK")
else:
    print(f"   ✗ Python {sys.version_info.major}.{sys.version_info.minor} - Need 3.8+")
    sys.exit(1)

# Check packages
print("\n2. Checking required packages...")
packages = {
    'tensorflow': 'TensorFlow',
    'matplotlib': 'Matplotlib',
    'numpy': 'NumPy',
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'PIL': 'Pillow'
}

missing = []
for package, name in packages.items():
    try:
        __import__(package)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")
        missing.append(name)

if missing:
    print("\n⚠ Missing packages!")
    print("\nInstall them with:")
    print("   pip3 install -r requirements.txt")
    sys.exit(1)

# Check TensorFlow version
print("\n3. Checking TensorFlow...")
import tensorflow as tf
print(f"   ✓ TensorFlow version: {tf.__version__}")

# Check if GPU is available (optional)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✓ GPU detected: {len(gpus)} GPU(s) available")
else:
    print("   ℹ No GPU detected - will use CPU (slower but works fine)")

# Check file structure
print("\n4. Checking file structure...")
import os
if os.path.exists('src'):
    print("   ✓ src/ folder exists")
else:
    print("   ✗ src/ folder missing - create it: mkdir src")
    
if os.path.exists('src/train_simple.py'):
    print("   ✓ src/train_simple.py exists")
else:
    print("   ⚠ src/train_simple.py not found - you need to create it")

if os.path.exists('src/api.py'):
    print("   ✓ src/api.py exists")
else:
    print("   ⚠ src/api.py not found - you need to create it")

# Check disk space
print("\n5. Checking disk space...")
import shutil
stat = shutil.disk_usage('.')
free_gb = stat.free / (1024**3)
if free_gb >= 2:
    print(f"   ✓ Free space: {free_gb:.1f} GB")
else:
    print(f"   ⚠ Only {free_gb:.1f} GB free - you need at least 2 GB")

print("\n" + "=" * 50)
if not missing:
    print("✓ All checks passed!")
    print("\nYou're ready to go!")
    print("\nNext steps:")
    print("   1. Train: python3 src/train_simple.py")
    print("   2. API:   python3 src/api.py")
else:
    print("⚠ Please fix the issues above first")
print("=" * 50)