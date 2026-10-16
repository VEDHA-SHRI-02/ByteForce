# QueueWise AI — Intelligent Queue Monitoring Dashboard

A hackathon starter project using a pretrained Ultralytics YOLO model and Streamlit.

## Features
- Upload a recorded queue video and analyse sampled frames with YOLO person detection
- View annotated detections and a crowd-density overlay
- Queue count snapshot, configurable crowd threshold, simple wait-time estimate
- Smart action recommendations and scenario planner
- Interactive charts and CSV export
- Save completed analyses locally and review them in Saved reports
- Download a formatted PDF report with summary metrics and frame-by-frame observations
- Compare each new clip with a baseline calculated from previously saved runs (analytics baseline, not model fine-tuning)
- Warm sage / coral / cream visual theme

## Run on Windows / VS Code

1. Install Python 3.10 or 3.11.
2. Extract this folder and open it in VS Code.
3. Open Terminal in the project folder.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run:
   `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
   and activate again.

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

7. Open the local URL shown in the terminal (usually http://localhost:8501).

## YOLO model
The app defaults to `yolo26n.pt`. On first run, Ultralytics may download the model weights, so internet access is required. If that weight is not available in your installed Ultralytics version, choose `yolov8n.pt` in the sidebar.

## Current CCTV support
This starter processes uploaded video files. The webcam option captures one still frame only. For real live CCTV, add an RTSP source using the camera/NVR's documented RTSP URL and ensure the computer has network access and permission to view the stream. Do not expose the CCTV stream publicly.

## Important prototype limitations
- It counts all detected people in sampled frames, not only people inside a queue boundary.
- It does not yet assign tracking IDs.
- The wait-time number is a rough estimate: detected count divided by the configured assumed service rate. It is not a guaranteed or individually tracked wait time.
- Peak-hour analytics from one video show observations within that clip; reliable forecasting requires stored data across multiple days.
- Use only video you have permission to process. No face recognition is implemented.


## Saved analytics and learning-style baseline
Completed video analyses can be saved from the Live monitor. Saved summaries and frame observations are stored locally in `queuewise_data/saved_runs.json` beside `app.py`; the Saved reports page provides history, charts, CSV downloads, and PDF reports. The dashboard calculates a historical baseline from saved runs to compare later videos. This is **not** retraining or fine-tuning the YOLO model: the detector continues to use pretrained weights. On hosted services, local files may not persist across redeploys unless persistent storage is configured.
