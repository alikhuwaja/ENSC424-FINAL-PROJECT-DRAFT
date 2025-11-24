from pathlib import Path     
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config, features


# ------------------------------------------------------------
# Emotion ID remapping:
#   Original dataset IDs: {1,3,4,5,6,7}
#   Model class indices : {0,1,2,3,4,5}
# ------------------------------------------------------------
EMOTION_MAP = {
    1: 0,   # neutral
    3: 1,   # happy
    4: 2,   # sad
    5: 3,   # angry
    6: 4,   # fearful
    7: 5,   # disgust
}


class SERDataset(Dataset):
    """
    Speech Emotion Recognition dataset 

    Loads ALL THREE CSVs:
      - config.RAVDESS_CSV
      - config.CREMAD_CSV
      - config.RAVDESS_VIDEO_CSV

    Training uses only the audio rows (RAVDESS + CREMAD).
    Video is loaded and available via helper functions.
    """

    def __init__(self):
        self.ravdess_df = pd.read_csv(config.RAVDESS_CSV)
        self.cremad_df = pd.read_csv(config.CREMAD_CSV)
        self.df = pd.concat([self.ravdess_df, self.cremad_df], ignore_index=True)

        try:
            self.video_df = pd.read_csv(config.RAVDESS_VIDEO_CSV)
        except FileNotFoundError:
            self.video_df = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns:
          x: log-Mel features as torch.FloatTensor (NUM_MEL, T)
          y: emotion label mapped to 0–5
        """
        row = self.df.iloc[idx]
        audio_path = row["path"]

        # Audio → Mel Spectrogram
        feats = features.extract_features_from_path(audio_path)
        x = torch.from_numpy(feats)

        # Map emotion_id → 0..5 (CRUCIAL for CrossEntropyLoss)
        raw_id = int(row["emotion_id"])
        y = torch.tensor(EMOTION_MAP[raw_id])

        return x, y
    
    def get_audio_metadata(self, idx):
        """
        Return all columns for this audio sample as a dataset row dictionary.
        """
        row = self.df.iloc[idx]
        return row.to_dict()
    
    def get_matching_video_rows(self, audio_idx):
        """
        Given an audio sample index, return a DataFrame of matching video rows.
        If no video index is loaded, returns None.
        """
        if self.video_df is None:
            return None

        audio_row = self.df.iloc[audio_idx]
        audio_path = audio_row["path"]
        audio_fname = Path(audio_path).stem  

        matching_rows = self.video_df[self.video_df["path"].apply(
            lambda vp: Path(vp).stem == audio_fname
        )]

        return matching_rows


    def get_video_features(self, video_path):
        """
        Extract log-Mel features from the video's audio track.
        """
        feats = features.extract_features_from_path(video_path)
        x = torch.from_numpy(feats)
        return x
    
    def get_video_metadata(self, video_path):
        """
        Return metadata row for given video path.
        """
        if self.video_df is None:
            return None

        matching_rows = self.video_df[self.video_df["path"] == str(video_path)]
        if matching_rows.empty:
            return None

        row = matching_rows.iloc[0]
        return row.to_dict()
    

class SERVideoIndex(Dataset):
    """
    Simple dataset for the RAVDESS VIDEO index.
    Returns each row of ravdess_video_index.csv as a dict.
    """

    def __init__(self):
        self.df = pd.read_csv(config.RAVDESS_VIDEO_CSV)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row.to_dict()


#if __name__ == "__main__":
 #   from torch.utils.data import DataLoader

  #  ds = SERDataset()
   # print("Total samples:", len(ds))

    # Inspect one sample
    #x, y = ds[0]
    #print("Feature shape:", x.shape)
    #print("Label:", y.item())