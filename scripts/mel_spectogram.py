import pandas as pd
import matplotlib.pyplot as plt
import librosa.display

from src import config
from src import features


def plot_example(csv_path, title_prefix):
    df = pd.read_csv(csv_path)
    row = df.iloc[868]
    #how many we want to read? change the index number for df.iloc[num]
    audio_path = row["path"]
    emotion = row.get("emotion", "unknown")

    print(f"Plotting {title_prefix} example:")
    print("  path   :", audio_path)
    print("  emotion:", emotion)

    logmel = features.extract_features_from_path(audio_path)

    plt.figure(figsize=(8, 4))
    librosa.display.specshow(
        logmel,
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{title_prefix} - {emotion}")
    plt.tight_layout()
    plt.show()


def main():
    plot_example(config.RAVDESS_CSV, "RAVDESS (audio)")
    plot_example(config.CREMAD_CSV, "CREMA-D (audio)")


if __name__ == "__main__":
    main()
