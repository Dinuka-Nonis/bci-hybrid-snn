"""
Baseline ANN (Artificial Neural Network) for BCI Classification
Simple feedforward network using spike count features
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# PATH SETUP (FIX)
# ------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..')
)

DATA_SPLITS_DIR = os.path.join(PROJECT_ROOT, 'data', 'splits')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------
class SimpleANN(nn.Module):
    """
    Simple feedforward neural network
    Uses spike counts (sum over time) as features
    """

    def __init__(self, input_dim=22, hidden_dim=128, output_dim=4):
        super(SimpleANN, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


# ------------------------------------------------------------------
# TRAINING FUNCTION
# ------------------------------------------------------------------
def train_baseline(
    spikes_train, y_train,
    spikes_val, y_val,
    spikes_test, y_test,
    epochs=30,
    batch_size=32,
    lr=0.001
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Convert spike trains to spike counts
    print("\nConverting spike trains to spike counts...")
    train_counts = spikes_train.sum(axis=2)
    val_counts = spikes_val.sum(axis=2)
    test_counts = spikes_test.sum(axis=2)

    print(f"  Train counts: {train_counts.shape}")
    print(f"  Val counts:   {val_counts.shape}")
    print(f"  Test counts:  {test_counts.shape}")

    # Datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(train_counts),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_counts),
        torch.LongTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(test_counts),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model
    model = SimpleANN(input_dim=22, hidden_dim=128, output_dim=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0

    print("\n" + "=" * 60)
    print("TRAINING BASELINE ANN")
    print("=" * 60)

    for epoch in range(epochs):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                os.path.join(RESULTS_DIR, 'baseline_ann_best.pt')
            )

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.3f}")

    # ---------------- TEST ----------------
    model.load_state_dict(
        torch.load(os.path.join(RESULTS_DIR, 'baseline_ann_best.pt'))
    )

    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)

            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = test_correct / test_total

    print("\n" + "=" * 60)
    print("BASELINE ANN RESULTS")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.3f}")
    print(f"Test Accuracy:           {test_acc:.3f}")

    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\nPer-Class Test Accuracy:")
    for cls in range(4):
        mask = all_labels == cls
        acc = (all_preds[mask] == all_labels[mask]).mean()
        print(f"  {class_names[cls]:12s}: {acc:.3f}")

    return model, history, test_acc


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("BASELINE ANN TRAINING")
    print("=" * 60)

    print("\nLoading data splits...")
    spikes_train = np.load(os.path.join(DATA_SPLITS_DIR, 'spikes_train_s1.npy'))
    spikes_val   = np.load(os.path.join(DATA_SPLITS_DIR, 'spikes_val_s1.npy'))
    spikes_test  = np.load(os.path.join(DATA_SPLITS_DIR, 'spikes_test_s1.npy'))

    y_train = np.load(os.path.join(DATA_SPLITS_DIR, 'y_train_s1.npy'))
    y_val   = np.load(os.path.join(DATA_SPLITS_DIR, 'y_val_s1.npy'))
    y_test  = np.load(os.path.join(DATA_SPLITS_DIR, 'y_test_s1.npy'))

    print(f"✓ Train: {spikes_train.shape}, {y_train.shape}")
    print(f"✓ Val:   {spikes_val.shape}, {y_val.shape}")
    print(f"✓ Test:  {spikes_test.shape}, {y_test.shape}")

    model, history, test_acc = train_baseline(
        spikes_train, y_train,
        spikes_val, y_val,
        spikes_test, y_test,
        epochs=30,
        batch_size=32,
        lr=0.001
    )

    print("\nPlotting training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].axhline(0.25, linestyle='--', alpha=0.5, label='Chance (25%)')
    axes[1].set_title('Training & Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, 'baseline_ann_training.png'),
        dpi=150,
        bbox_inches='tight'
    )
    plt.show()

    np.save(
        os.path.join(RESULTS_DIR, 'baseline_results.npy'),
        {'test_acc': test_acc, 'history': history}
    )

    print("\n" + "=" * 60)
    print("✓ BASELINE TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Baseline Test Accuracy: {test_acc:.3f}")
    print(f"Target for Hybrid SNN: > {test_acc:.3f}")
