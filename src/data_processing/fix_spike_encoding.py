"""
Re-encode with optimal max_rate for target firing rate
"""

import numpy as np
import sys
sys.path.append('../..')

from src.data_processing.spike_encoding import SpikeEncoder

print("=" * 60)
print("FIXING SPIKE ENCODING - OPTIMAL FIRING RATE")
print("=" * 60)

# Load preprocessed data
print("\nLoading preprocessed data...")
X_clean = np.load('../../data/processed/X_subject1_clean.npy')
y = np.load('../../data/processed/y_subject1.npy')
print(f"✓ Loaded: {X_clean.shape}")

# Initialize encoder
encoder = SpikeEncoder(dt=0.004)

# Find optimal max_rate
print("\n" + "=" * 60)
print("FINDING OPTIMAL MAX_RATE")
print("=" * 60)

test_rates = [15, 20, 25, 30]
optimal_rate = None

for test_rate in test_rates:
    print(f"\nTesting max_rate={test_rate}:")
    
    # Test on 20 trials
    test_spikes = encoder.encode_dataset(
        X_clean[:20], 
        method='rate_coding',
        max_rate=test_rate,
        verbose=False
    )
    
    avg_firing_rate = test_spikes.mean() / encoder.dt
    print(f"  → Average firing rate: {avg_firing_rate:.2f} Hz")
    
    if 5 <= avg_firing_rate <= 15:
        print(f"  ✓ IN TARGET RANGE!")
        optimal_rate = test_rate
        break
    elif avg_firing_rate < 5:
        print(f"  ✗ Too low, need higher max_rate")
    else:
        print(f"  ✗ Too high, need lower max_rate")

if optimal_rate is None:
    print("\n⚠ No perfect rate found, using max_rate=20 as best compromise")
    optimal_rate = 20

# Encode full dataset with optimal rate
print("\n" + "=" * 60)
print(f"ENCODING FULL DATASET WITH max_rate={optimal_rate}")
print("=" * 60)

spike_trains = encoder.encode_dataset(
    X_clean, 
    method='rate_coding',
    max_rate=optimal_rate,
    verbose=True
)

# Verify
avg_rate = spike_trains.mean() / encoder.dt
print("\n" + "=" * 60)
print("FINAL VERIFICATION")
print("=" * 60)
print(f"Average firing rate: {avg_rate:.2f} Hz")

if 5 <= avg_rate <= 15:
    print("✓✓ PERFECT! Firing rate in target range (5-15 Hz)")
    quality = "GOOD"
elif 4 <= avg_rate <= 17:
    print("✓ ACCEPTABLE: Firing rate close to target")
    quality = "ACCEPTABLE"
else:
    print("⚠ WARNING: Firing rate outside ideal range")
    quality = "SUBOPTIMAL"

# Save with quality tag
print(f"\nSaving spike trains (quality: {quality})...")
np.save('../../data/processed/spikes_subject1.npy', spike_trains)
np.save('../../data/processed/spike_params.npy', 
        {'max_rate': optimal_rate, 'avg_firing_rate': avg_rate, 'quality': quality})

print("✓ Saved to data/processed/")

print("\n" + "=" * 60)
print("✓ ENCODING FIXED!")
print("=" * 60)
print(f"\nFinal parameters:")
print(f"  max_rate: {optimal_rate}")
print(f"  Average firing rate: {avg_rate:.2f} Hz")
print(f"  Quality: {quality}")
print(f"  Total spikes per trial: {spike_trains.sum(axis=(1,2)).mean():.0f}")

print("\n✓ Ready for Day 5: Train/Test Split & Baseline Model")
print("=" * 60)