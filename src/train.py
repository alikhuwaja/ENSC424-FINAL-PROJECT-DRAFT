import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

from sklearn.metrics import f1_score, balanced_accuracy_score

from .dataset import SERDataset
from .models import CRNN, SERTransformer
from . import config


def collate_fn(batch):
    """
    Pads/collates Mel spectrograms into (B, 1, NUM_MEL, T) tensor.
    """
    xs, ys = zip(*batch)  # xs are (NUM_MEL, T) tensors from SERDataset

    # Ensure xs are tensors (in case you ever pass numpy arrays)
    xs = [
        x if isinstance(x, torch.Tensor) else torch.from_numpy(x)
        for x in xs
    ]

    # Pad time dimension
    max_T = max(x.shape[1] for x in xs)
    padded = []
    for x in xs:
        pad = max_T - x.shape[1]
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))  # pad along time dim
        padded.append(x)

    X = torch.stack(padded)      # (B, NUM_MEL, T)
    X = X.unsqueeze(1)           # (B, 1, NUM_MEL, T) for your CNN

    # ys are 0-D tensors or ints; stack them cleanly
    if isinstance(ys[0], torch.Tensor):
        Y = torch.stack(ys)      # (B,)
    else:
        Y = torch.tensor(ys)

    return X.float(), Y.long()


# --------------------------
#   SIMPLE DATA AUGMENTATIONS
# --------------------------
def apply_augmentations(x):
    """
    x: (B, 1, mel, T)
    """
    B, _, M, T = x.shape

    # Gaussian noise
    if config.AUG_NOISE:
        noise = torch.randn_like(x) * 0.01
        x = x + noise

    # Frequency masking
    if config.AUG_FREQ_MASK:
        f = np.random.randint(0, M // 6)
        f0 = np.random.randint(0, M - f)
        x[:, :, f0:f0 + f, :] = 0

    # Time masking
    if config.AUG_TIME_MASK:
        t = np.random.randint(0, T // 6)
        t0 = np.random.randint(0, T - t)
        x[:, :, :, t0:t0 + t] = 0

    return x


# --------------------------
#   TRAINING LOOP
# --------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    running_loss = 0

    for X, y in tqdm(loader, desc="Training"):
        X, y = X.to(device), y.to(device)

        # Apply augmentations ONLY during training
        if config.USE_AUGMENTATIONS:
            X = apply_augmentations(X)

        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X.size(0)

    return running_loss / len(loader.dataset)


# --------------------------
#   VALIDATION (with Macro-F1 & UAR)
# --------------------------
def validate(model, loader, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    val_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = loss_fn(logits, y)
            val_loss += loss.item() * X.size(0)

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Metrics
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    uar = balanced_accuracy_score(all_labels, all_preds)

    return (
        val_loss / len(loader.dataset),
        acc,
        macro_f1,
        uar
    )


# --------------------------
#   MAIN
# --------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    dataset = SERDataset()
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --------------------------
    #  SWITCH MODELS HERE
    # --------------------------
    if config.MODEL_TYPE == "crnn":
        model = CRNN().to(device)
        model_name = "best_model_crnn.pth"
    else:
        model = SERTransformer().to(device)
        model_name = "best_model_transformer.pth"

    optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_val_loss = float("inf")

    for epoch in range(1, config.N_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.N_EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, acc, macro_f1, uar = validate(model, val_loader, device)

        print(
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  |  "
            f"Acc: {acc:.4f}  |  "
            f"Macro-F1: {macro_f1:.4f}  |  "
            f"UAR: {uar:.4f}"
        )

        # -----------------------
        #  SAVE BEST MODEL
        # -----------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_name)
            print(f"Saved new best model to {model_name}!")

    print("\nTraining complete.")

if __name__ == "__main__":
    main()
    