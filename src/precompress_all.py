import os
from pydub import AudioSegment
from pydub.utils import which

from .dataset import SERDataset

# OPTIONAL: force pydub to use the system ffmpeg (adjust path if needed)
# If this path matches your install, uncomment the next line:
AudioSegment.converter = r"C:\Program Files\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

BITRATES = ["128k", "64k", "32k"]


def compress_one(in_path, bitrate):
    base, ext = os.path.splitext(in_path)

    mp3_path = f"{base}_{bitrate}.mp3"
    wav_path = f"{base}_{bitrate}_decoded.wav"

    # If final decoded WAV already exists, skip (idempotent)
    if os.path.exists(wav_path):
        print(f"[{bitrate}] Skipping existing: {wav_path}")
        return

    # Clean up any stale MP3 from previous crashed runs
    if os.path.exists(mp3_path):
        os.remove(mp3_path)

    try:
        # Original WAV -> MP3
        audio = AudioSegment.from_file(in_path)  # input is RAVDESS/CREMAD WAV
        audio.export(mp3_path, format="mp3", bitrate=bitrate)

        # MP3 -> WAV (explicitly tell pydub it's MP3)
        compressed = AudioSegment.from_file(mp3_path, format="mp3")
        compressed.export(wav_path, format="wav")

        # Remove intermediate MP3 to save space
        os.remove(mp3_path)

        print(f"[{bitrate}] OK: {wav_path}")
    except Exception as e:
        print(f"[{bitrate}] ERROR on {in_path}: {e}")
        # If something went wrong, clean partial files
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)


def main():
    ds = SERDataset()
    n = len(ds)
    print(f"Total samples: {n}")

    for idx in range(n):
        meta = ds.get_audio_metadata(idx)
        in_path = meta["path"]

        for br in BITRATES:
            compress_one(in_path, br)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{n} files")


if __name__ == "__main__":
    main()
