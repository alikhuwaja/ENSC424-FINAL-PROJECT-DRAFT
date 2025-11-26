import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score

from .dataset import SERDataset
from .models import CRNN, SERTransformer
from .train import collate_fn
from . import config, features


# --------------------------
#  MODEL LOADER
# --------------------------
def load_model(model_type, device):
    """
    Load the best model for a given model_type ('crnn' or 'transformer').
    Returns: (model, model_name, ckpt_name).
    """
    if model_type == "crnn":
        model = CRNN().to(device)
        ckpt = "best_model_crnn.pth"
        model_name = "CRNN"
    elif model_type == "transformer":
        model = SERTransformer().to(device)
        ckpt = "best_model_transformer.pth"
        model_name = "SERTransformer"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    return model, model_name, ckpt


# --------------------------
#  GENERIC EVAL ON DATALOADER
# --------------------------
def evaluate_loader(model, loader, device):
    """
    Generic evaluation over a DataLoader.
    Returns: accuracy, macro_f1, uar, confusion_matrix
    """
    all_true, all_pred = [], []

    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_true.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    acc = (all_true == all_pred).mean()
    macro_f1 = f1_score(all_true, all_pred, average="macro")
    uar = balanced_accuracy_score(all_true, all_pred)
    cm = confusion_matrix(all_true, all_pred)

    return acc, macro_f1, uar, cm


# --------------------------
#  BASELINE EVALUATION (original WAV)
# --------------------------
def evaluate_baseline(model, device, batch_size=None):
    """
    Evaluate model on original WAVs (no compression).
    Uses SERDataset + DataLoader.
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    dataset = SERDataset()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return evaluate_loader(model, loader, device)


# --------------------------
#  COMPRESSION EVALUATION (precompressed *_decoded.wav)
# --------------------------
def evaluate_compression_for_bitrate(model, dataset, device, bitrate="64k"):
    """
    Evaluate model when using pre-compressed+decoded WAVs.

    For each original path '.../file.wav', we expect:
        '.../file_{bitrate}_decoded.wav'
    already created by precompress_all.py.

    Returns: accuracy, macro_f1, uar
    """
    model.eval()
    all_true, all_pred = [], []

    n = len(dataset)
    for idx in range(n):
        meta = dataset.get_audio_metadata(idx)
        original_path = meta["path"]
        _, y = dataset[idx]  # label tensor

        base, ext = os.path.splitext(original_path)
        wav_path = f"{base}_{bitrate}_decoded.wav"

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Missing precompressed file: {wav_path}")

        # Extract features from decoded WAV using SAME pipeline as training
        mel = features.extract_features_from_path(wav_path)  # (NUM_MEL, T)
        x_comp = torch.from_numpy(mel)

        # Use same collate_fn to get (B, 1, NUM_MEL, T)
        batch_x, batch_y = collate_fn([(x_comp, y)])
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        with torch.no_grad():
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)

        all_true.append(int(batch_y.item()))
        all_pred.append(int(preds.item()))

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    acc = (all_true == all_pred).mean()
    macro_f1 = f1_score(all_true, all_pred, average="macro")
    uar = balanced_accuracy_score(all_true, all_pred)

    return acc, macro_f1, uar


# --------------------------
#  MAIN
# --------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model_types = ["crnn", "transformer"]
    bitrates = ["128k", "64k", "32k"]

    for model_type in model_types:
        print("\n" + "=" * 60)
        print(f"Evaluating model_type = '{model_type}'")

        # 1) Load best model for this type
        model, model_name, ckpt = load_model(model_type, device)
        print(f"Using model: {model_name} (checkpoint='{ckpt}')")

        # 2) Baseline on original WAVs
        print("\nBaseline ")
        base_acc, base_f1, base_uar, base_cm = evaluate_baseline(model, device)
        print(f"Baseline Accuracy : {base_acc:.4f}")
        print(f"Baseline Macro-F1 : {base_f1:.4f}")
        print(f"Baseline UAR      : {base_uar:.4f}")
        print("Baseline confusion matrix:\n", base_cm)

        # 3) Compression evaluation
        dataset = SERDataset()
        print("\nCompression robustness ")
        for br in bitrates:
            acc_c, f1_c, uar_c = evaluate_compression_for_bitrate(model, dataset, device, bitrate=br)
            print(f"{br}: Acc={acc_c:.4f} | Macro-F1={f1_c:.4f} | UAR={uar_c:.4f}")


if __name__ == "__main__":
    main()




#python -m src.evaluate
