# Speech Emotion Recognition (SER) — ENSC 424 Final Project

A Machine Learning project that predicts human emotion from speech audio.  
This repo preprocesses audio into **log-Mel spectrograms**, trains deep models (e.g., **CRNN** and **Transformer**), evaluates performance (Accuracy / Macro-F1 / Confusion Matrix), and optionally serves the model via a **FastAPI** inference app.

---

## Project Goal
Given an input audio clip (speech), classify it into one of the emotion categories:

- `neutral`
- `happy`
- `sad`
- `angry`
- `fearful`
- `disgust`

> Labels may be adjusted depending on dataset availability & mapping.

---

## How It Works (Pipeline)

1. **Dataset**
   - Speech emotion datasets (e.g., RAVDESS / CREMA-D or course-provided datasets).
2. **Preprocessing**
   - Load `.wav` audio
   - Resample to a fixed sample rate (e.g., 16 kHz)
   - Normalize audio / trim silence (optional)
3. **Feature Extraction**
   - Convert audio → **log-Mel spectrogram**: `(n_mels × time)`
   - (Optional) augmentations (noise, time masking, freq masking)
4. **Model Training**
   - Train deep learning models:
     - **CRNN**: CNN for feature extraction + RNN/GRU for temporal modeling
     - **Transformer**: Self-attention over time frames to learn long-range dependencies
5. **Evaluation**
   - Accuracy, Macro-F1 (balanced measure across classes)
   - Confusion matrix for error analysis
6. **Inference**
   - Load trained checkpoint
   - Run prediction on audio file (CLI) or via **FastAPI** endpoint

---

## Repository Structure 

```text
ser-project/
├── app.py                 # FastAPI inference server
├── requirements.txt
├── README.md
├── src/
│   ├── config.py          # sample rate, n_mels, labels, paths
│   ├── features.py        # mel/log-mel extraction utilities
│   ├── dataset.py         # PyTorch Dataset + label mapping
│   ├── train.py           # training loop
│   ├── eval.py            # evaluation script
│   └── models/
│       ├── crnn.py        # CRNN model
│       └── transformer.py # Transformer model
├── notebooks/             # experiments / EDA / training notebooks
├── data/
│   ├── raw/               # original dataset files
│   └── processed/         # cached features/splits
└── checkpoints/           # saved models














