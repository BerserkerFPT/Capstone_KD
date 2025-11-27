"""
Script để setup môi trường Kaggle trước khi chạy training
Chạy script này đầu tiên trong Kaggle notebook
"""
import subprocess
import sys

def install_dependencies():
    """Install required packages on Kaggle"""
    print("=" * 70)
    print("INSTALLING DEPENDENCIES FOR KAGGLE")
    print("=" * 70)
    
    packages = [
        'timm',           # PyTorch Image Models
        'openpyxl',       # For Excel export
        'seaborn',        # For visualization
    ]
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    print("\n✓ All dependencies installed successfully!")


def check_environment():
    """Check Kaggle environment"""
    import torch
    import os
    
    print("\n" + "=" * 70)
    print("KAGGLE ENVIRONMENT CHECK")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("⚠ WARNING: GPU not available! Make sure to enable GPU in Kaggle settings.")
        print("  Go to Settings (right panel) → Accelerator → Select 'GPU T4 x2' or 'GPU P100'")
    
    # Check PyTorch version
    print(f"\n✓ PyTorch Version: {torch.__version__}")
    
    # Check Kaggle paths
    print(f"\n✓ Kaggle Working Directory: /kaggle/working")
    print(f"✓ Kaggle Input Directory: /kaggle/input")
    
    # List available datasets
    if os.path.exists('/kaggle/input'):
        datasets = os.listdir('/kaggle/input')
        if datasets:
            print(f"\n✓ Available Datasets:")
            for dataset in datasets:
                print(f"  - /kaggle/input/{dataset}")
        else:
            print("\n⚠ No datasets found in /kaggle/input")
            print("  Make sure to add your dataset in Kaggle notebook settings (Add Data)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n🚀 Setting up Kaggle environment...\n")
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Check environment
    check_environment()
    
    print("\n✅ Setup complete! You can now run main.py")
    print("\nNext steps:")
    print("  1. Make sure GPU is enabled (Settings → Accelerator → GPU)")
    print("  2. Add your dataset (Settings → Add Data)")
    print("  3. Update DATASET_PATH in config.py if needed")
    print("  4. Run: python main.py")
    print("\n" + "=" * 70 + "\n")
