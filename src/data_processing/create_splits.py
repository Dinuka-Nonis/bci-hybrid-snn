"""
Create Train/Validation/Test splits for Subject 1
70% train, 15% validation, 15% test
"""

import numpy as np
from sklearn.model_selection import train_test_split

print("=" * 60)
print("CREATING TRAIN/VAL/TEST SPLITS")
print("=" * 60)

# Load preprocessed data and spike trains
print("\nLoading data...")
X_clean = np.load('../../data/processed/X_subject1_clean.npy')
spikes = np.load('../../data/processed/spikes_subject1.npy')
y = np.load('../../data/processed/y_subject1.npy')

print(f"✓ Loaded preprocessed: {X_clean.shape}")
print(f"✓ Loaded spike trains: {spikes.shape}")
print(f"✓ Loaded labels: {y.shape}")

# Verify class distribution
print("\nOriginal class distribution:")
unique, counts = np.unique(y, return_counts=True)
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} trials ({count/len(y)*100:.1f}%)")

print("\n" + "=" * 60)
print("SPLITTING DATA")
print("=" * 60)

# First split: 70% train, 30% temp (for val+test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_clean, y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y  # Keep class balance
)

spikes_train, spikes_temp = train_test_split(
    spikes,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: Split temp into 50/50 for val and test (15% each of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

spikes_val, spikes_test = train_test_split(
    spikes_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train)} trials ({len(X_train)/len(X_clean)*100:.1f}%)")
print(f"  Val:   {len(X_val)} trials ({len(X_val)/len(X_clean)*100:.1f}%)")
print(f"  Test:  {len(X_test)} trials ({len(X_test)/len(X_clean)*100:.1f}%)")

# Verify stratification (class balance maintained)
print("\n" + "=" * 60)
print("VERIFYING CLASS BALANCE")
print("=" * 60)

for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
    print(f"\n{split_name} split:")
    unique, counts = np.unique(split_y, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} trials ({count/len(split_y)*100:.1f}%)")

# Save splits
print("\n" + "=" * 60)
print("SAVING SPLITS")
print("=" * 60)

# Preprocessed data splits
np.save('../../data/splits/X_train_s1.npy', X_train)
np.save('../../data/splits/X_val_s1.npy', X_val)
np.save('../../data/splits/X_test_s1.npy', X_test)

np.save('../../data/splits/y_train_s1.npy', y_train)
np.save('../../data/splits/y_val_s1.npy', y_val)
np.save('../../data/splits/y_test_s1.npy', y_test)

print("✓ Saved preprocessed splits")

# Spike train splits
np.save('../../data/splits/spikes_train_s1.npy', spikes_train)
np.save('../../data/splits/spikes_val_s1.npy', spikes_val)
np.save('../../data/splits/spikes_test_s1.npy', spikes_test)

print("✓ Saved spike train splits")

print("\n" + "=" * 60)
print("✓ SPLITS CREATED SUCCESSFULLY!")
print("=" * 60)

print("\nSummary:")
print(f"  Training set:   {X_train.shape}")
print(f"  Validation set: {X_val.shape}")
print(f"  Test set:       {X_test.shape}")
print(f"  All classes balanced: ✓")

print("\n✓ Ready for baseline model training!")
print("=" * 60)