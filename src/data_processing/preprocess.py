"""
EEG Preprocessing Pipeline
Cleans and prepares EEG data for neural network training
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


class EEGPreprocessor:
    """
    Preprocessing pipeline for motor imagery EEG data
    
    Steps:
    1. Notch filter (remove 50/60 Hz power line noise)
    2. Bandpass filter (8-30 Hz for motor imagery)
    3. Baseline correction (remove DC offset)
    4. Normalization (z-score)
    """
    
    def __init__(self, sfreq=250, lowcut=8, highcut=30, notch_freq=50):
        """
        Args:
            sfreq: Sampling frequency (Hz)
            lowcut: Lower cutoff for bandpass (Hz)
            highcut: Upper cutoff for bandpass (Hz)
            notch_freq: Frequency to notch filter (50 Hz Europe, 60 Hz US)
        """
        self.sfreq = sfreq
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        
        print(f"Initialized EEG Preprocessor:")
        print(f"  Sampling rate: {sfreq} Hz")
        print(f"  Bandpass: {lowcut}-{highcut} Hz")
        print(f"  Notch filter: {notch_freq} Hz")
    
    def notch_filter(self, data, quality_factor=30):
        """
        Remove power line noise at 50 or 60 Hz
        
        Args:
            data: [n_trials, n_channels, n_timepoints] or [n_channels, n_timepoints]
            quality_factor: Q factor for notch filter (higher = narrower)
        
        Returns:
            Filtered data (same shape as input)
        """
        nyquist = 0.5 * self.sfreq
        freq = self.notch_freq / nyquist
        b, a = iirnotch(freq, quality_factor)
        
        if data.ndim == 3:
            # Multiple trials
            filtered = np.zeros_like(data)
            for i in range(data.shape[0]):
                for ch in range(data.shape[1]):
                    filtered[i, ch, :] = filtfilt(b, a, data[i, ch, :])
        elif data.ndim == 2:
            # Single trial
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch, :] = filtfilt(b, a, data[ch, :])
        else:
            raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
        
        return filtered
    
    def bandpass_filter(self, data, order=5):
        """
        Bandpass filter to extract motor imagery frequencies (8-30 Hz)
        
        Why 8-30 Hz?
        - Alpha band (8-13 Hz): Motor cortex idle state
        - Beta band (13-30 Hz): Active motor control
        - Motor imagery shows desynchronization in these bands
        
        Args:
            data: [n_trials, n_channels, n_timepoints] or [n_channels, n_timepoints]
            order: Filter order (higher = sharper cutoff, but slower)
        
        Returns:
            Filtered data (same shape as input)
        """
        nyquist = 0.5 * self.sfreq
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        
        if data.ndim == 3:
            # Multiple trials
            filtered = np.zeros_like(data)
            for i in range(data.shape[0]):
                for ch in range(data.shape[1]):
                    filtered[i, ch, :] = filtfilt(b, a, data[i, ch, :])
        elif data.ndim == 2:
            # Single trial
            filtered = np.zeros_like(data)
            for ch in range(data.shape[0]):
                filtered[ch, :] = filtfilt(b, a, data[ch, :])
        else:
            raise ValueError(f"Expected 2D or 3D data, got {data.ndim}D")
        
        return filtered
    
    def remove_baseline(self, data, baseline_end=0.5):
        """
        Remove baseline (mean during first 0.5 seconds)
        
        This corrects for DC offset and slow drifts
        
        Args:
            data: [n_trials, n_channels, n_timepoints]
            baseline_end: Time (seconds) to use for baseline calculation
        
        Returns:
            Baseline-corrected data
        """
        if data.ndim != 3:
            raise ValueError("remove_baseline expects 3D data [trials, channels, time]")
        
        baseline_idx = int(baseline_end * self.sfreq)
        
        # Calculate mean during baseline period
        baseline_mean = data[:, :, :baseline_idx].mean(axis=2, keepdims=True)
        
        # Subtract baseline from entire trial
        return data - baseline_mean
    
    def normalize(self, data, method='zscore'):
        """
        Normalize the data
        
        Args:
            data: [n_trials, n_channels, n_timepoints]
            method: 'zscore' (default) or 'minmax'
        
        Returns:
            Normalized data
        """
        if data.ndim != 3:
            raise ValueError("normalize expects 3D data [trials, channels, time]")
        
        if method == 'zscore':
            # Z-score normalization per channel (across all trials and time)
            mean = data.mean(axis=(0, 2), keepdims=True)
            std = data.std(axis=(0, 2), keepdims=True) + 1e-8  # Avoid division by zero
            normalized = (data - mean) / std
            
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = data.min(axis=(0, 2), keepdims=True)
            max_val = data.max(axis=(0, 2), keepdims=True)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        return normalized
    
    def preprocess(self, X, verbose=True):
        """
        Apply full preprocessing pipeline
        
        Args:
            X: [n_trials, n_channels, n_timepoints] - raw EEG data
            verbose: Print progress
        
        Returns:
            X_clean: Preprocessed data (same shape)
        """
        if verbose:
            print("\n" + "=" * 60)
            print("PREPROCESSING PIPELINE")
            print("=" * 60)
            print(f"Input shape: {X.shape}")
        
        # Step 1: Notch filter
        if verbose:
            print(f"\n1. Notch filtering ({self.notch_freq} Hz)...")
        X = self.notch_filter(X)
        if verbose:
            print(f"   ✓ Removed power line noise")
        
        # Step 2: Bandpass filter
        if verbose:
            print(f"\n2. Bandpass filtering ({self.lowcut}-{self.highcut} Hz)...")
        X = self.bandpass_filter(X)
        if verbose:
            print(f"   ✓ Extracted motor imagery frequencies")
        
        # Step 3: Baseline correction
        if verbose:
            print(f"\n3. Baseline correction...")
        X = self.remove_baseline(X)
        if verbose:
            print(f"   ✓ Removed DC offset")
        
        # Step 4: Normalization
        if verbose:
            print(f"\n4. Z-score normalization...")
        X = self.normalize(X, method='zscore')
        if verbose:
            print(f"   ✓ Normalized to zero mean, unit variance")
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"✓ PREPROCESSING COMPLETE")
            print("=" * 60)
            print(f"Output shape: {X.shape}")
            print(f"Mean: {X.mean():.4f} (should be ~0)")
            print(f"Std: {X.std():.4f} (should be ~1)")
            print(f"Range: [{X.min():.2f}, {X.max():.2f}]")
        
        return X


# Test the preprocessor
if __name__ == "__main__":
    """Test preprocessing on Subject 1"""
    
    import sys
    sys.path.append('../..')
    from src.data_processing.load_data import BCIDataLoader
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("TESTING PREPROCESSOR")
    print("=" * 60)
    
    # Load data
    print("\nLoading Subject 1...")
    loader = BCIDataLoader(subject_ids=[1])
    data = loader.load_all_subjects()
    
    X_raw = data[1]['X']
    y = data[1]['y']
    
    print(f"✓ Loaded: {X_raw.shape}")
    
    # Preprocess
    preprocessor = EEGPreprocessor(sfreq=250, lowcut=8, highcut=30, notch_freq=50)
    X_clean = preprocessor.preprocess(X_raw.copy())
    
    # Visualize before/after
    print("\nCreating before/after comparison plot...")
    
    trial_idx = 0
    channel_idx = 7  # C3 channel
    time = np.arange(X_raw.shape[2]) / 250
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    axes[0].plot(time, X_raw[trial_idx, channel_idx, :], 'b-', linewidth=0.8)
    axes[0].set_ylabel('Raw Signal (µV)', fontsize=12)
    axes[0].set_title(f'Before Preprocessing - Trial {trial_idx}, Channel C3', 
                      fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time, X_clean[trial_idx, channel_idx, :], 'r-', linewidth=0.8)
    axes[1].set_ylabel('Processed Signal (σ)', fontsize=12)
    axes[1].set_xlabel('Time (seconds)', fontsize=12)
    axes[1].set_title('After Preprocessing', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../results/preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/preprocessing_comparison.png")
    plt.show()
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    np.save('../../data/processed/X_subject1_clean.npy', X_clean)
    np.save('../../data/processed/y_subject1.npy', y)
    print("✓ Saved to data/processed/")
    
    print("\n" + "=" * 60)
    print("✓ PREPROCESSING TEST COMPLETE!")
    print("=" * 60)
