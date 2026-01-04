"""
Data loader for BNCI2014001 dataset
Handles loading, extracting, and organizing EEG data
"""

import numpy as np
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
import warnings
warnings.filterwarnings('ignore')


class BCIDataLoader:
    """
    Loads and prepares BNCI2014001 motor imagery data
    
    The dataset contains:
    - 9 subjects
    - 4 classes: left hand (0), right hand (1), feet (2), tongue (3)
    - 22 EEG channels
    - 250 Hz sampling rate
    - ~3-second trials (750 samples after extraction)
    """
    
    def __init__(self, subject_ids=None):
        """
        Args:
            subject_ids: List of subject IDs to load (default: [1])
                        None will load all available subjects
        """
        self.dataset = BNCI2014001()
        
        if subject_ids is None:
            subject_ids = [1]  # Default to subject 1 for Week 1
        
        self.subject_ids = subject_ids
        
        # Motor imagery paradigm (handles event extraction)
        self.paradigm = MotorImagery(
            events=['left_hand', 'right_hand', 'feet', 'tongue'],
            n_classes=4,
            fmin=8,   # Filter: 8-30 Hz (motor imagery relevant frequencies)
            fmax=30,
            tmin=0.5, # Extract 0.5s to 3.5s after cue (most informative period)
            tmax=3.5,
            baseline=None,  # We'll handle baseline correction ourselves
            channels=None    # Use all available channels
        )
    
    def load_subject(self, subject_id):
        """
        Load data for a single subject
        
        Returns:
            dict: {
                'X': numpy array [n_trials, n_channels, n_timepoints]
                'y': numpy array [n_trials] - class labels (0, 1, 2, 3)
                'info': dict with metadata
            }
        """
        print(f"Loading Subject {subject_id}...")
        
        try:
            # Get data using MOABB paradigm
            X, y, metadata = self.paradigm.get_data(
                dataset=self.dataset,
                subjects=[subject_id],
                return_epochs=False  # Return numpy arrays directly
            )
            
            # Handle different MOABB return formats
            # Newer versions return dict, older return arrays directly
            if isinstance(X, dict):
                # Dictionary format: extract first session
                sessions = list(X.keys())
                session_key = sessions[0]
                X_data = X[session_key]
                y_data = y[session_key]
            else:
                # Direct array format
                X_data = X
                y_data = y
            
            # Remap labels to 0, 1, 2, 3
            # Handle both string and numeric labels
            if isinstance(y_data[0], str):
                label_map = {
                    'left_hand': 0,
                    'right_hand': 1,
                    'feet': 2,
                    'tongue': 3
                }
                y_numeric = np.array([label_map[label] for label in y_data])
            else:
                # Already numeric, just ensure correct mapping
                y_numeric = y_data.astype(int)
            
            # Get metadata
            info = {
                'subject_id': subject_id,
                'n_trials': X_data.shape[0],
                'n_channels': X_data.shape[1],
                'n_timepoints': X_data.shape[2],
                'sfreq': 250,  # Sampling frequency
                'duration': X_data.shape[2] / 250,  # Trial duration in seconds
                'channel_names': ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                                 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
                                 'P2', 'POz'],
                'class_names': ['left_hand', 'right_hand', 'feet', 'tongue']
            }
            
            print(f"  ✓ Shape: {X_data.shape}")
            print(f"  ✓ Classes: {np.unique(y_numeric, return_counts=True)}")
            
            return {
                'X': X_data,
                'y': y_numeric,
                'info': info
            }
            
        except Exception as e:
            print(f"  ✗ ERROR loading subject {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_all_subjects(self):
        """
        Load data for all specified subjects
        
        Returns:
            dict: {subject_id: {'X': ..., 'y': ..., 'info': ...}}
        """
        print("=" * 60)
        print(f"LOADING {len(self.subject_ids)} SUBJECT(S)")
        print("=" * 60)
        
        all_data = {}
        
        for subject_id in self.subject_ids:
            subject_data = self.load_subject(subject_id)
            if subject_data is not None:
                all_data[subject_id] = subject_data
            print()
        
        print("=" * 60)
        print(f"✓ {len(all_data)} SUBJECT(S) LOADED SUCCESSFULLY")
        print("=" * 60)
        
        return all_data


# Quick test function
if __name__ == "__main__":
    """Test the loader with Subject 1"""
    
    print("\n" + "=" * 60)
    print("TESTING DATA LOADER WITH SUBJECT 1")
    print("=" * 60)
    
    loader = BCIDataLoader(subject_ids=[1])
    data = loader.load_all_subjects()
    
    if 1 in data:
        # Print summary
        subject_data = data[1]
        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"X shape: {subject_data['X'].shape}")
        print(f"y shape: {subject_data['y'].shape}")
        print(f"Sampling frequency: {subject_data['info']['sfreq']} Hz")
        print(f"Trial duration: {subject_data['info']['duration']:.2f} seconds")
        print(f"Number of channels: {subject_data['info']['n_channels']}")
        
        # Check data quality
        X = subject_data['X']
        y = subject_data['y']
        
        print("\n" + "=" * 60)
        print("DATA QUALITY CHECK")
        print("=" * 60)
        print(f"NaN values: {np.isnan(X).sum()}")
        print(f"Infinite values: {np.isinf(X).sum()}")
        print(f"Min value: {X.min():.2e}")
        print(f"Max value: {X.max():.2e}")
        print(f"Mean: {X.mean():.2e}")
        print(f"Std: {X.std():.2e}")
        
        print("\n" + "=" * 60)
        print("CLASS DISTRIBUTION")
        print("=" * 60)
        unique, counts = np.unique(y, return_counts=True)
        class_names = subject_data['info']['class_names']
        for cls, count in zip(unique, counts):
            print(f"Class {cls} ({class_names[cls]:12s}): {count} trials")
        
        print("\n" + "=" * 60)
        print("✓ ALL CHECKS PASSED! DATA IS READY!")
        print("=" * 60)
        print("\nWhat this means:")
        print(f"- You have {subject_data['X'].shape[0]} trials of brain activity")
        print(f"- Each trial has {subject_data['X'].shape[1]} EEG channels")
        print(f"- Each trial is {subject_data['info']['duration']:.1f} seconds long")
        print(f"- Data is balanced across all 4 movement types")
        print("\n✓ Ready for Day 2: Data Exploration & Visualization!")
        
    else:
        print("\n✗ Failed to load Subject 1")
        print("Check the error messages above")