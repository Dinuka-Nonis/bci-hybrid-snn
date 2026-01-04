"""
Verify all dependencies are installed correctly
"""

import sys
import platform

def check_package(package_name, import_name=None):
    """Try importing a package and report version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:20s} version {version}")
        return True
    except ImportError as e:
        print(f"✗ {package_name:20s} FAILED: {e}")
        return False

print("=" * 60)
print("CHECKING CONDA ENVIRONMENT")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Architecture: {platform.machine()}")
print()

print("=" * 60)
print("PACKAGE VERSIONS")
print("=" * 60)

packages = [
    ('numpy', 'numpy'),
    ('scipy', 'scipy'),
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
    ('mne', 'mne'),
    ('moabb', 'moabb'),
    ('torch', 'torch'),
    ('snntorch', 'snntorch'),
    ('sklearn', 'sklearn'),
    ('optuna', 'optuna'),
    ('jupyter', 'jupyter'),
]

all_good = True
for pkg_name, import_name in packages:
    if not check_package(pkg_name, import_name):
        all_good = False

print("\n" + "=" * 60)

if all_good:
    print("✓ ALL PACKAGES INSTALLED SUCCESSFULLY!")
    print("=" * 60)
    
    # Check CUDA availability
    import torch
    print("\nGPU Check:")
    if torch.cuda.is_available():
        print(f"✓ CUDA available! GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("⚠ CUDA not available")
        print("  Training will use CPU (slower but still works)")
        print("  This is fine for learning!")
    
else:
    print("✗ SOME PACKAGES FAILED TO INSTALL")
    print("=" * 60)
    print("\nTroubleshooting:")
    print("1. Make sure conda environment is activated: conda activate bci_env")
    print("2. Try reinstalling failed packages")
    print("3. Check conda install commands above")

print("\n" + "=" * 60)