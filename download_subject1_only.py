"""
Download only Subject 1 from BNCI2014001 dataset
This is sufficient for Week 1 of the project
"""

import os
import time
from moabb.datasets import BNCI2014001

print("=" * 60)
print("DOWNLOADING SUBJECT 1 ONLY (WEEK 1)")
print("=" * 60)
print("\nDataset Info:")
print("- Subject 1 only (sufficient for Week 1)")
print("- 4 classes: left hand, right hand, feet, tongue")
print("- 288 trials")
print("- 22 EEG channels")
print("- Sampling rate: 250 Hz")
print(f"\nDownload location: {os.path.expanduser('~')}\\mne_data\\")
print("\nThis will take 2-5 minutes...")
print("=" * 60)

# Initialize dataset
dataset = BNCI2014001()

# Download only subject 1 with retry logic
max_retries = 3
retry_delay = 5  # seconds

for attempt in range(max_retries):
    try:
        print(f"\nAttempt {attempt + 1}/{max_retries}...")
        print("Downloading Subject 1...\n")
        
        # Download only subject 1
        dataset.download(subject_list=[1])
        
        print("\n" + "=" * 60)
        print("✓ DOWNLOAD COMPLETE!")
        print("=" * 60)
        
        # Verify download
        print("\nVerifying data...")
        data = dataset.get_data(subjects=[1])
        print(f"✓ Successfully loaded Subject 1")
        print(f"  Number of sessions: {len(data[1])}")
        
        print("\n" + "=" * 60)
        print("✓ READY FOR WEEK 1!")
        print("=" * 60)
        print("\nNext step: Test the data loader")
        print("  python src\\data_processing\\load_data.py")
        
        break  # Success! Exit the retry loop
        
    except Exception as e:
        print(f"\n✗ Attempt {attempt + 1} failed: {e}")
        
        if attempt < max_retries - 1:
            print(f"\nRetrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("\n" + "=" * 60)
            print("✗ ALL ATTEMPTS FAILED")
            print("=" * 60)
            print("\nAlternative solutions:")
            print("1. Check your internet connection stability")
            print("2. Try using a VPN or different network")
            print("3. Disable Windows firewall temporarily")
            print("4. Try again later (server might be overloaded)")
            print("\nNote: Files already downloaded are saved and won't be")
            print("      re-downloaded on the next attempt.")
            
            import traceback
            print("\nFull error:")
            traceback.print_exc()