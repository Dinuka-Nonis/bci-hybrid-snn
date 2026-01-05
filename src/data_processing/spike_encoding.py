"""
Spike Encoding for EEG Signals
Converts continuous EEG values into discrete spike trains
"""

import numpy as np
import matplotlib.pyplot as plt


class SpikeEncoder:
    """
    Converts continuous EEG signals to spike trains using rate coding
    
    Rate Coding: Higher signal amplitude → Higher firing rate
    - Strong positive signal → Many spikes
    - Weak signal → Few spikes
    - Like how neurons encode stimulus intensity!
    """
    
    def __init__(self, dt=0.004, method='rate_coding'):
        """
        Args:
            dt: Time step in seconds (default: 4ms → 250 Hz)
            method: Encoding method ('rate_coding' for now)
        """
        self.dt = dt
        self.method = method
        
        print(f"Initialized Spike Encoder:")
        print(f"  Method: {method}")
        print(f"  Time step: {dt*1000:.1f} ms")
    
    def rate_coding(self, signal, max_rate=50):
        """
        Rate coding: Signal amplitude → Spike probability
        
        How it works:
        1. Normalize signal to [0, 1]
        2. Higher values → Higher spike probability
        3. Generate spikes via Bernoulli process
        
        Args:
            signal: [n_channels, n_timepoints] - single trial
            max_rate: Maximum firing rate in Hz
        
        Returns:
            spikes: [n_channels, n_timepoints] - binary spike train
        """
        n_channels, n_timepoints = signal.shape
        spikes = np.zeros_like(signal, dtype=np.uint8)
        
        for ch in range(n_channels):
            s = signal[ch]
            
            # Normalize to [0, 1]
            s_min = s.min()
            s_max = s.max()
            if s_max - s_min > 1e-8:
                s_norm = (s - s_min) / (s_max - s_min)
            else:
                s_norm = np.zeros_like(s)
            
            # Spike probability = normalized signal * max_rate * dt
            spike_prob = s_norm * max_rate * self.dt
            spike_prob = np.clip(spike_prob, 0, 1)  # Ensure [0, 1]
            
            # Generate spikes (Bernoulli process)
            spikes[ch] = (np.random.rand(n_timepoints) < spike_prob).astype(np.uint8)
        
        return spikes
    
    def encode_dataset(self, X, method='rate_coding', max_rate=50, verbose=True):
        """
        Encode entire dataset
        
        Args:
            X: [n_trials, n_channels, n_timepoints]
            method: Encoding method
            max_rate: Maximum firing rate (Hz)
            verbose: Print progress
        
        Returns:
            spikes: [n_trials, n_channels, n_timepoints] - binary
        """
        n_trials = X.shape[0]
        spike_trains = []
        
        if verbose:
            print("\n" + "=" * 60)
            print("SPIKE ENCODING")
            print("=" * 60)
            print(f"Input shape: {X.shape}")
            print(f"Method: {method}")
            print(f"Max firing rate: {max_rate} Hz")
            print(f"Time step: {self.dt*1000:.1f} ms")
        
        for i in range(n_trials):
            if method == 'rate_coding':
                spikes = self.rate_coding(X[i], max_rate=max_rate)
            else:
                raise ValueError(f"Unknown encoding method: {method}")
            
            spike_trains.append(spikes)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Encoded {i+1}/{n_trials} trials...")
        
        spike_trains = np.array(spike_trains)
        
        if verbose:
            print(f"\n✓ Encoding complete!")
            print("=" * 60)
            print(f"Output shape: {spike_trains.shape}")
            print(f"Spike density: {spike_trains.mean():.4f}")
            print(f"Avg firing rate: {spike_trains.mean() / self.dt:.2f} Hz")
            print(f"Expected range: 5-15 Hz for max_rate={max_rate}")
        
        return spike_trains
    
    def visualize_encoding(self, signal, spikes, channel_idx=0, save_path=None):
        """
        Visualize original signal vs spike train
        
        Args:
            signal: [n_channels, n_timepoints] - original signal
            spikes: [n_channels, n_timepoints] - spike train
            channel_idx: Which channel to plot
            save_path: Path to save figure
        """
        n_timepoints = signal.shape[1]
        time = np.arange(n_timepoints) * self.dt
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Original signal
        axes[0].plot(time, signal[channel_idx], 'b-', linewidth=1)
        axes[0].set_ylabel('Signal Amplitude (σ)', fontsize=12)
        axes[0].set_title(f'Original Signal - Channel {channel_idx}', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Spike train (raster plot)
        spike_times = time[spikes[channel_idx] == 1]
        axes[1].eventplot(spike_times, colors='red', linewidths=2)
        axes[1].set_ylabel('Spikes', fontsize=12)
        axes[1].set_xlabel('Time (seconds)', fontsize=12)
        axes[1].set_title(f'Spike Train - {len(spike_times)} spikes', 
                         fontsize=14, fontweight='bold')
        axes[1].set_ylim([-0.5, 1.5])
        axes[1].set_yticks([])
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {save_path}")
        
        plt.show()
        
        # Print statistics
        print("\nSpike Statistics:")
        print(f"  Total spikes: {spikes[channel_idx].sum()}")
        print(f"  Duration: {time[-1]:.2f} seconds")
        print(f"  Firing rate: {spikes[channel_idx].sum() / time[-1]:.2f} Hz")


# Test the encoder
if __name__ == "__main__":
    """Test spike encoding on preprocessed data"""
    
    import sys
    sys.path.append('../..')
    
    print("=" * 60)
    print("TESTING SPIKE ENCODER")
    print("=" * 60)
    
    # Load preprocessed data
    print("\nLoading preprocessed data...")
    X_clean = np.load('../../data/processed/X_subject1_clean.npy')
    y = np.load('../../data/processed/y_subject1.npy')
    print(f"✓ Loaded: {X_clean.shape}")
    
    # Initialize encoder
    encoder = SpikeEncoder(dt=0.004)
    
    # Test on a single trial first
    print("\n" + "=" * 60)
    print("TEST 1: Single Trial Encoding")
    print("=" * 60)
    
    trial_idx = 0
    signal = X_clean[trial_idx]  # Shape: [22, 751]
    
    spikes = encoder.rate_coding(signal, max_rate=50)
    
    print(f"\nSingle trial results:")
    print(f"  Signal shape: {signal.shape}")
    print(f"  Spikes shape: {spikes.shape}")
    print(f"  Total spikes: {spikes.sum()}")
    print(f"  Spike density: {spikes.mean():.4f}")
    
    # Visualize one channel
    encoder.visualize_encoding(
        signal, 
        spikes, 
        channel_idx=7,  # C3 channel
        save_path='../../results/spike_encoding_example.png'
    )
    
    # Encode entire dataset
    print("\n" + "=" * 60)
    print("TEST 2: Full Dataset Encoding")
    print("=" * 60)
    
    # Try different max_rates to find optimal
    print("\nTesting different max_rate values...")
    test_rates = [20, 25, 30]
    
    for test_rate in test_rates:
        print(f"\nTrying max_rate={test_rate}:")
        test_spikes = encoder.encode_dataset(
            X_clean[:10],  # Just 10 trials for testing
            method='rate_coding',
            max_rate=test_rate,
            verbose=False
        )
        avg_rate = test_spikes.mean() / encoder.dt
        print(f"  Average firing rate: {avg_rate:.2f} Hz")
        
        if 5 <= avg_rate <= 15:
            print(f"  ✓ OPTIMAL: Using max_rate={test_rate}")
            chosen_rate = test_rate
            break
    else:
        chosen_rate = 25  # Default if none work
    
    print(f"\n{'='*60}")
    print(f"Encoding full dataset with max_rate={chosen_rate}...")
    print(f"{'='*60}")
    
    spike_trains = encoder.encode_dataset(
        X_clean, 
        method='rate_coding',
        max_rate=chosen_rate,
        verbose=True
    )
    
    # Verify encoding quality
    print("\n" + "=" * 60)
    print("ENCODING QUALITY CHECK")
    print("=" * 60)
    
    avg_rate = spike_trains.mean() / encoder.dt
    print(f"Average firing rate: {avg_rate:.2f} Hz")
    
    if 5 <= avg_rate <= 15:
        print("✓ GOOD: Firing rate in target range (5-15 Hz)")
    elif avg_rate < 5:
        print("⚠ WARNING: Firing rate too low, try increasing max_rate")
    else:
        print("⚠ WARNING: Firing rate too high, try decreasing max_rate")
    
    # Check spike distribution across trials
    spikes_per_trial = spike_trains.sum(axis=(1, 2))
    print(f"\nSpikes per trial:")
    print(f"  Mean: {spikes_per_trial.mean():.1f}")
    print(f"  Std: {spikes_per_trial.std():.1f}")
    print(f"  Range: [{spikes_per_trial.min()}, {spikes_per_trial.max()}]")
    
    # Visualize firing rates across channels
    print("\nCreating firing rate heatmap...")
    
    firing_rates = spike_trains.mean(axis=(0, 2)) / encoder.dt  # [n_channels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    channel_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz',
                    'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
                    'P2', 'POz']
    
    bars = ax.bar(range(22), firing_rates, color='steelblue', alpha=0.7)
    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Firing Rate (Hz)', fontsize=12)
    ax.set_title('Average Firing Rate per Channel', fontsize=14, fontweight='bold')
    ax.set_xticks(range(22))
    ax.set_xticklabels(channel_names, rotation=45, ha='right')
    ax.axhline(y=avg_rate, color='r', linestyle='--', label=f'Mean: {avg_rate:.2f} Hz')
    ax.axhspan(5, 15, alpha=0.1, color='green', label='Target range')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('../../results/firing_rates_per_channel.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: results/firing_rates_per_channel.png")
    plt.show()
    
    # Save encoded data
    print("\nSaving spike trains...")
    np.save('../../data/processed/spikes_subject1.npy', spike_trains)
    print("✓ Saved to data/processed/spikes_subject1.npy")
    
    print("\n" + "=" * 60)
    print("✓ SPIKE ENCODING TEST COMPLETE!")
    print("=" * 60)
    print("\nKey Results:")
    print(f"  ✓ Encoded {spike_trains.shape[0]} trials")
    print(f"  ✓ Average firing rate: {avg_rate:.2f} Hz")
    print(f"  ✓ Spike trains saved and ready for SNN training")
