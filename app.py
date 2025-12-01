import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["OMP_NUM_THREADS"] = "1"

import subprocess
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse

from src import config, features
from src.models import CRNN, SERTransformer

app = FastAPI(title="SER FastAPI", version="1.3")

EMO_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_CACHE: Dict[str, torch.nn.Module] = {}

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")


def _ckpt_for(model_type: str) -> str:
    mt = model_type.lower().strip()
    if mt == "crnn":
        return "best_model_crnn.pth"
    if mt == "transformer":
        return "best_model_transformer.pth"
    raise ValueError("model_type must be 'crnn' or 'transformer'")


def _build_model(model_type: str) -> torch.nn.Module:
    mt = model_type.lower().strip()
    if mt == "crnn":
        return CRNN()
    if mt == "transformer":
        return SERTransformer()
    raise ValueError("model_type must be 'crnn' or 'transformer'")


def load_model(model_type: str) -> torch.nn.Module:
    mt = model_type.lower().strip()
    if mt in _MODEL_CACHE:
        return _MODEL_CACHE[mt]

    model = _build_model(mt).to(DEVICE)
    ckpt = _ckpt_for(mt)

    # Safer load (new torch), fallback for older torch
    try:
        state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=DEVICE)

    model.load_state_dict(state)
    model.eval()
    _MODEL_CACHE[mt] = model
    return model


def convert_any_audio_to_wav(in_path: str, out_wav_path: str) -> None:
    """
    Convert any input audio to mono 16k WAV using ffmpeg.
    """
    cmd = [
        FFMPEG_BIN, "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", in_path,
        "-ac", "1",
        "-ar", str(config.SAMPLE_RATE),
        "-f", "wav",
        out_wav_path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.strip() or p.stdout.strip()}")


@torch.inference_mode()
def predict_from_any_path(audio_path: str, model_type: str):
    """
    Accepts any audio type (wav/mp3/webm/m4a/ogg...). Converts to wav then runs model.
    """
    model = load_model(model_type)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpwav:
        wav_path = tmpwav.name

    try:
        convert_any_audio_to_wav(audio_path, wav_path)

        mel = features.extract_features_from_path(wav_path)  # (NUM_MEL, T)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(1).float().to(DEVICE)  # (1,1,mel,T)

        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        pred_idx = int(np.argmax(probs))

        return {
            "device": DEVICE,
            "model_type": model_type.lower().strip(),
            "predicted_class": pred_idx,
            "predicted_label": EMO_LABELS[pred_idx],
            "probabilities": [float(p) for p in probs],
        }
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass


@app.get("/", response_class=JSONResponse)
def root():
    return {
        "status": "ok",
        "device": DEVICE,
        "endpoints": {
            "demo_ui": "/demo",
            "predict_api": "/predict?model_type=crnn|transformer",
            "docs": "/docs",
        },
    }


@app.get("/demo", response_class=HTMLResponse)
def demo():
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>SER Demo (Media Upload + Real time Mic input)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body {{ font-family: system-ui, Segoe UI, Arial; max-width: 980px; margin: 28px auto; padding: 0 14px; }}
    .card {{ border:1px solid #e5e7eb; border-radius:16px; padding:16px; box-shadow:0 10px 30px rgba(0,0,0,0.06); }}
    h1 {{ margin: 4px 0 10px; font-size: 22px; }}
    .row {{ display:flex; gap:12px; flex-wrap:wrap; align-items:center; }}
    select, input[type=file], button {{ padding:10px 12px; border-radius:10px; border:1px solid #d1d5db; background:#fff; font-size:14px; }}
    button {{ cursor:pointer; border:none; background:#111827; color:#fff; }}
    button:disabled {{ opacity:0.55; cursor:not-allowed; }}
    .muted {{ color:#6b7280; font-size:13px; }}
    .pill {{ display:inline-block; padding: 4px 10px; border-radius: 999px; background:#eef2ff; color:#3730a3; font-weight:600; font-size:13px; }}
    pre {{ background:#0b1020; color:#e5e7eb; padding:12px; border-radius:12px; overflow:auto; }}
    hr {{ border:none; border-top:1px solid #e5e7eb; margin:16px 0; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Speech Emotion Recognition : Media Upload + Real time Mic input</h1>
    <div class="muted">
      Labels: <span class="pill">{", ".join(EMO_LABELS)}</span> &nbsp; | &nbsp;
      Mic mode sends a <b>5s</b> segment and updates prediction.
    </div>

    <br/>
    <div class="row">
      <label>Model:</label>
      <select id="modelType">
        <option value="crnn" selected>CRNN</option>
        <option value="transformer">Transformer</option>
      </select>
    </div>

    <hr/>

    <h3>1) Upload audio file</h3>
    <div class="row">
      <input id="fileInput" type="file" accept="audio/*" />
      <button id="btnUpload">Upload</button>
    </div>

    <hr/>

    <h3>2) Microphone (real-time segments)</h3>
    <div class="row">
      <button id="btnMicStart">Start Mic</button>
      <button id="btnMicStop" disabled>Stop Mic</button>
      <span id="micStatus" class="muted">Mic: idle</span>
    </div>
    <div class="muted" style="margin-top:8px;">
      If Mic fails: allow permissions (Chrome lock icon), and ensure no other app is blocking the microphone.
    </div>

    <h3>Result</h3>
    <pre id="out">Ready.</pre>
  </div>

<script>
  const out = document.getElementById("out");
  const modelType = document.getElementById("modelType");

  // Upload
  document.getElementById("btnUpload").addEventListener("click", async () => {{
    const f = document.getElementById("fileInput").files?.[0];
    if (!f) {{
      out.textContent = "Please choose an audio file first.";
      return;
    }}
    out.textContent = "Running inference (upload)...";
    try {{
      const fd = new FormData();
      fd.append("file", f);
      const url = `/predict?model_type=${{encodeURIComponent(modelType.value)}}`;
      const res = await fetch(url, {{ method:"POST", body: fd }});
      out.textContent = JSON.stringify(await res.json(), null, 2);
    }} catch (e) {{
      out.textContent = "Error: " + e;
    }}
  }});

  // Mic recording in FULL 3s segments (stop -> upload -> restart)
  let stream = null;
  let running = false;

  const micStatus = document.getElementById("micStatus");
  const btnStart = document.getElementById("btnMicStart");
  const btnStop  = document.getElementById("btnMicStop");

  function setStatus(msg) {{ micStatus.textContent = msg; }}

  function pickMimeType() {{
    const tryTypes = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
      "audio/ogg"
    ];
    for (const t of tryTypes) {{
      if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) return t;
    }}
    return "";
  }}

  async function sendBlob(blob, filename) {{
    if (!blob || blob.size < 2000) {{
      out.textContent = "Mic chunk too small/empty. Try again.";
      return;
    }}
    out.textContent = "Running inference (mic)...";
    const fd = new FormData();
    fd.append("file", blob, filename);
    const url = `/predict?model_type=${{encodeURIComponent(modelType.value)}}`;
    const res = await fetch(url, {{ method:"POST", body: fd }});
    out.textContent = JSON.stringify(await res.json(), null, 2);
  }}

  function recordOneSegment(seconds=5) {{
    if (!running || !stream) return;

    const mimeType = pickMimeType();
    const chunks = [];
    const rec = new MediaRecorder(stream, mimeType ? {{ mimeType }} : undefined);

    rec.ondataavailable = (e) => {{
      if (e.data && e.data.size > 0) chunks.push(e.data);
    }};

    rec.onstop = async () => {{
      const type = rec.mimeType || "audio/webm";
      const blob = new Blob(chunks, {{ type }});
      const ext = type.includes("ogg") ? "ogg" : "webm";
      await sendBlob(blob, `mic.${{ext}}`);
      if (running) recordOneSegment(seconds);
    }};

    rec.start(); // IMPORTANT: no timeslice => complete container output
    setTimeout(() => {{ try {{ rec.stop(); }} catch {{}} }}, seconds * 1000);
  }}

  btnStart.addEventListener("click", async () => {{
    out.textContent = "Requesting microphone...";
    try {{
      stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
      running = true;
      setStatus("Mic: recording (3s segments)...");
      btnStart.disabled = true;
      btnStop.disabled = false;
      recordOneSegment(5);
    }} catch (e) {{
      running = false;
      setStatus("Mic: error");
      out.textContent = "Microphone error: " + e;
    }}
  }});

  btnStop.addEventListener("click", () => {{
    running = false;
    try {{
      if (stream) stream.getTracks().forEach(t => t.stop());
    }} catch {{}}
    stream = null;
    setStatus("Mic: idle");
    btnStart.disabled = false;
    btnStop.disabled = true;
    out.textContent = "Mic stopped.";
  }});
</script>
</body>
</html>
"""
    return HTMLResponse(html)


@app.post("/predict")
async def predict(file: UploadFile = File(...), model_type: str = "crnn"):
    suffix = Path(file.filename).suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict_from_any_path(tmp_path, model_type=model_type)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# Run:
# python -m uvicorn app:app --host 127.0.0.1 --port 9000
# Demo:
# http://127.0.0.1:9000/demo

#activating conda
#conda activate "D:\Study Materials\ENSC 424\Project\ser-project\.conda"
