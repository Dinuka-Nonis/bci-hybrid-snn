"""
Download BNCI2014001 dataset for motor imagery BCI
This will download ~500MB of EEG data to your home directory
"""

import os
from moabb.datasets import BNCI2014001

print("=" * 60)
print("DOWNLOADING BNCI2014001 DATASET")
print("=" * 60)
print("\nDataset Info:")
print("- 9 subjects")
print("- 4 classes: left hand, right hand, feet, tongue")
print("- 288 trials per subject")
print("- 22 EEG channels")
print("- Sampling rate: 250 Hz")
print(f"\nDownload location: {os.path.expanduser('~')}\\mne_data\\")
print("\nThis will take 5-15 minutes depending on internet speed...")
print("=" * 60)

# Initialize dataset
dataset = BNCI2014001()

# Download all subjects
try:
    print("\nStarting download...")
    print("(You'll see progress bars for each file)\n")
    
    dataset.download(subject_list=None)  # None = all subjects
    
    print("\n" + "=" * 60)
    print("✓ DOWNLOAD COMPLETE!")
    print("=" * 60)
    
    # Verify download by checking available subjects
    print("\nVerifying data...")
    subjects = dataset.subject_list
    print(f"Available subjects: {subjects}")
    print(f"Total: {len(subjects)} subjects")
    
    # Try loading one subject to verify everything works
    print("\nTesting data load (Subject 1)...")
    data = dataset.get_data(subjects=[1])
    print(f"✓ Successfully loaded Subject 1")
    print(f"  Number of sessions: {len(data[1])}")
    
    print("\n" + "=" * 60)
    print("✓ READY FOR DATA EXPLORATION!")
    print("=" * 60)
    print("\nNext step: Run the data loader to verify everything works")
    print("  python src\\data_processing\\load_data.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print(f"✗ ERROR during download: {e}")
    print("=" * 60)
    print("\nTroubleshooting:")
    print("1. Check your internet connection")
    print("2. Make sure you have ~1GB free space")
    print("3. Try running the script again (it will resume)")
    print("4. Check Windows firewall/antivirus isn't blocking")
    import traceback
    print("\nFull error:")
    traceback.print_exc()