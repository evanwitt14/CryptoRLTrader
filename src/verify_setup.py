import importlib
import sys

required_packages = [
    'tensorflow',
    'torch',
    'pandas',
    'numpy',
    'scikit-learn',
    'plotly',
    'stable_baselines3',
    'python-binance',
    'transformers',
    'ta'
]

def check_packages():
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} is missing")
    
    if missing_packages:
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)
    else:
        print("\nAll required packages are installed!")

if __name__ == "__main__":
    check_packages() 