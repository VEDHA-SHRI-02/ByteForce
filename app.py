import os
import time
import tempfile
import json
import io
from datetime import datetime
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ultralytics import YOLO

st.set_page_config(
    page_title="QueueWise AI | Intelligent Queue Operations",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual system ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {
  --ink:#25332f; --muted:#6e7b75; --paper:#f7f5ef; --card:#fffefa;
  --coral:#e87961; --mint:#a9cbb8; --lime:#d9e7a7; --line:#e6e4da;
}
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stSidebar"] { background:#e8eee4; border-right:1px solid #d8e0d5; }
[data-testid="stSidebar"] * { color:var(--ink); }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink); }
.hero {
  background:linear-gradient(115deg,#e3eee4 0%,#f8e6d9 58%,#f6d5c8 100%);
  border:1px solid #e3ded2; border-radius:26px; padding:28px 30px; margin-bottom:18px;
  position:relative; overflow:hidden;
}
.hero:after { content:""; position:absolute; width:220px;height:220px;border-radius:50%;
  right:-50px;top:-80px;background:rgba(255,255,255,.36); }
.eyebrow { text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:700; color:#62756a; }
.hero h1 { font-size:36px; margin:7px 0 8px 0; line-height:1.08; }
.hero p { max-width:700px; color:#53665d; font-size:15px; margin:0; }
.section-label { font-size:12px; font-weight:700; letter-spacing:1.5px; color:#738078; text-transform:uppercase; margin:10px 0 12px; }
.metric-card {
  background:var(--card); border:1px solid var(--line); border-radius:20px; padding:18px 20px;
  min-height:125px; box-shadow:0 5px 18px rgba(51,63,55,.035);
}
.metric-label { color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.7px; text-transform:uppercase; }
.metric-value { font-family:'Space Grotesk',sans-serif; font-size:31px; font-weight:700; margin:6px 0 2px; color:var(--ink); }
.metric-foot { color:var(--muted); font-size:12px; }
.pill { display:inline-block; padding:5px 10px; border-radius:99px; background:#e5f0e7; color:#3e7255; font-size:11px; font-weight:700; }
.insight {
  background:#fffefa; border:1px solid #e6e4da; border-left:4px solid #e87961;
  border-radius:14px; padding:15px 17px; margin:8px 0;
}
.insight strong { font-family:'Space Grotesk',sans-serif; }
.small-muted { color:#77847d; font-size:12px; }
div[data-testid="stFileUploader"] { background:#fffefa; border:1px dashed #c7d4c8; border-radius:16px; padding:8px; }
.stButton > button, .stDownloadButton > button { border-radius:12px; border:0; background:#263d35; color:#fffefa; font-weight:700; padding:.65rem 1rem; }
.stButton > button:hover, .stDownloadButton > button:hover { background:#3e6656; color:white; }
div[data-testid="stTabs"] button { font-weight:700; }
hr { border-color:#e6e4da; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def metric_card(label, value, foot, emoji):
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">{emoji} &nbsp; {label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-foot">{foot}</div>
    </div>""", unsafe_allow_html=True)

def get_model(model_name):
    # Ultralytics downloads the pretrained weights on first run if not cached.
    return YOLO(model_name)

def make_heatmap(frame, boxes, size=(8, 5)):
    h, w = frame.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        cv2.circle(heat, (cx, cy), max(18, min(h,w)//12), 1, -1)
    if heat.max() > 0:
        heat = cv2.GaussianBlur(heat, (0, 0), 25)
        heat = heat / heat.max()
    colored = cv2.applyColorMap(np.uint8(heat*255), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(frame, .58, colored, .42, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

def estimate_wait(queue_count, service_rate_per_min):
    if queue_count <= 0:
        return 0
    if service_rate_per_min <= 0:
        return None
    return int(np.ceil(queue_count / service_rate_per_min))

def status_for(count, threshold):
    if count >= threshold:
        return "High congestion", "🔴", "high"
    if count >= max(1, int(threshold * .6)):
        return "Busy", "🟠", "medium"
    return "Flowing", "🟢", "low"

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🧭 QueueWise AI")
    st.caption("Queue intelligence • privacy-first")
    st.divider()
    page = st.radio("WORKSPACE", ["Live monitor", "Insights & action advisor", "Saved reports", "About the system"], label_visibility="visible")
    st.divider()
    st.markdown("### ⚙️ Monitor settings")
    model_name = st.selectbox("Detection model", ["yolo26n.pt", "yolov8n.pt"], help="A pretrained lightweight model. The first run may download weights.")
    queue_threshold = st.slider("Crowd alert threshold", 5, 60, 20, 1)
    service_rate = st.slider("People served per minute", 0.5, 10.0, 2.0, 0.5)
    st.caption("Waiting time is an estimate based on the service rate you set.")
    st.divider()
    st.markdown("**Privacy note**")
    st.caption("This prototype uses person detection only. It does not identify faces or infer age/disability.")

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <div class="eyebrow">AI-powered queue operations</div>
  <h1>Less waiting.<br>More thoughtful service.</h1>
  <p>Turn a queue video into actionable insights — detect people, estimate wait time, spot congestion and help staff decide what to do next.</p>
</div>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "last_count" not in st.session_state:
    st.session_state.last_count = 0
if "processed_rgb" not in st.session_state:
    st.session_state.processed_rgb = None
if "heat_rgb" not in st.session_state:
    st.session_state.heat_rgb = None
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
if "detection_summary" not in st.session_state:
    st.session_state.detection_summary = {}
if "analysis_saved" not in st.session_state:
    st.session_state.analysis_saved = False

# Local persistent history. This file is created beside app.py when a user saves a run.
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queuewise_data")
RUNS_FILE = os.path.join(DATA_DIR, "saved_runs.json")
def load_saved_runs():
    try:
        with open(RUNS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []
def persist_run(run):
    os.makedirs(DATA_DIR, exist_ok=True)
    runs = load_saved_runs()
    runs.append(run)
    with open(RUNS_FILE, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)
    return runs

if page == "Live monitor":
    left, right = st.columns([1.08, .92], gap="large")
    with left:
        st.markdown('<div class="section-label">01 / Video input</div>', unsafe_allow_html=True)
        source_mode = st.radio("Choose a video source", ["Upload recorded video", "Laptop webcam snapshot"], horizontal=True)
        uploaded = None
        if source_mode == "Upload recorded video":
            uploaded = st.file_uploader("Drop an MP4 / MOV / AVI queue video here", type=["mp4", "mov", "avi", "m4v"])
            st.caption("For a reliable hackathon demo, begin with a short recorded queue clip.")
        else:
            cam_img = st.camera_input("Capture a frame from your laptop camera")
            if cam_img:
                st.info("This option captures a still frame, not continuous live CCTV. Use an RTSP stream for true live camera processing.")
                uploaded = cam_img
        queue_roi = st.checkbox("Count people in the whole frame", value=True)
        st.caption("For a fixed queue area, define a polygon/ROI in a future iteration; this starter counts all detected people in the frame.")
        process_btn = st.button("▶  Analyse queue video", use_container_width=True)

        if process_btn:
            if uploaded is None:
                st.warning("Upload a video first.")
            else:
                suffix = ".mp4"
                if hasattr(uploaded, "name"):
                    ext = os.path.splitext(uploaded.name)[1].lower()
                    if ext in [".mp4", ".mov", ".avi", ".m4v"]:
                        suffix = ext
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    input_path = tmp.name
                try:
                    model = get_model(model_name)
                    cap = cv2.VideoCapture(input_path)
                    if not cap.isOpened():
                        # Could be a still image from camera_input
                        image_bytes = uploaded.getvalue()
                        arr = np.frombuffer(image_bytes, np.uint8)
                        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if image is None:
                            st.error("Could not read this video. Please try a standard MP4 file.")
                        else:
                            result = model.predict(image, classes=[0], verbose=False)[0]
                            boxes = []
                            if result.boxes is not None:
                                for b in result.boxes.xyxy.cpu().numpy():
                                    boxes.append(tuple(map(int, b[:4])))
                            annotated = result.plot()
                            st.session_state.processed_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            st.session_state.heat_rgb = make_heatmap(image, boxes)
                            st.session_state.last_count = len(boxes)
                            st.session_state.history = [{"sample": "Captured frame", "people": len(boxes)}]
                            st.session_state.detection_summary = {"frames": 1, "max_count": len(boxes), "avg_count": len(boxes), "video": "Captured frame"}
                            st.session_state.analysis_saved = False
                    else:
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                        stride = max(1, int(fps // 3))  # sample at roughly 3 frames/second
                        counts = []
                        sampled_frames = []
                        latest_frame = None
                        latest_annotated = None
                        latest_boxes = []
                        progress = st.progress(0, text="Loading video and running YOLO detection…")
                        live_status = st.empty()
                        idx = 0
                        while True:
                            ok, frame = cap.read()
                            if not ok:
                                break
                            if idx % stride == 0:
                                result = model.predict(frame, classes=[0], verbose=False, conf=0.35)[0]
                                boxes = []
                                if result.boxes is not None:
                                    for b in result.boxes.xyxy.cpu().numpy():
                                        boxes.append(tuple(map(int, b[:4])))
                                count = len(boxes)
                                counts.append(count)
                                sampled_frames.append({"sample": f"{idx/fps:.1f}s", "people": count})
                                live_status.caption(f"Live analysis: {len(counts)} sampled frames · latest count {count} · peak so far {max(counts)} people")
                                latest_frame = frame.copy()
                                latest_annotated = result.plot()
                                latest_boxes = boxes
                                if frame_count > 0:
                                    progress.progress(min(.99, idx / frame_count), text=f"Analysing frame {idx}/{frame_count}…")
                            idx += 1
                        cap.release()
                        progress.progress(1.0, text="Analysis complete.")
                        if not counts:
                            st.error("No frames could be processed from this video.")
                        else:
                            st.session_state.processed_rgb = cv2.cvtColor(latest_annotated, cv2.COLOR_BGR2RGB)
                            st.session_state.heat_rgb = make_heatmap(latest_frame, latest_boxes)
                            st.session_state.last_count = int(counts[-1])
                            st.session_state.history = sampled_frames
                            st.session_state.analysis_saved = False
                            st.session_state.detection_summary = {
                                "frames": len(counts),
                                "max_count": int(max(counts)),
                                "avg_count": round(float(np.mean(counts)), 1),
                                "video": getattr(uploaded, "name", "Uploaded video"),
                            }
                    cap.release()
                except Exception as e:
                    st.error(f"Model/video processing failed: {e}")
                    st.info("Check your internet connection on first run, installed packages, and video format.")
                finally:
                    try:
                        os.remove(input_path)
                    except Exception:
                        pass

    with right:
        st.markdown('<div class="section-label">02 / Detection preview</div>', unsafe_allow_html=True)
        if st.session_state.processed_rgb is not None:
            tab1, tab2 = st.tabs(["Detected people", "Congestion heatmap"])
            with tab1:
                st.image(st.session_state.processed_rgb, use_container_width=True, caption="Latest sampled frame with YOLO detections")
            with tab2:
                st.image(st.session_state.heat_rgb, use_container_width=True, caption="Illustrative density overlay based on detected person locations")
        else:
            st.markdown("""
            <div style="background:#fffefa;border:1px dashed #cbd7cc;border-radius:20px;padding:30px;text-align:center;min-height:280px">
              <div style="font-size:50px">📹</div>
              <h3>Detection preview appears here</h3>
              <p style="color:#78847d">Upload a queue clip, then run analysis to see person boxes and the crowd-density overlay.</p>
              <div style="font-size:30px;margin-top:16px">🧍 🧍 🧍 → 🧠 → 📊</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">03 / Queue snapshot</div>', unsafe_allow_html=True)
    count = int(st.session_state.last_count)
    wait = estimate_wait(count, service_rate)
    status, emoji, level = status_for(count, queue_threshold)
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1: metric_card("People detected", str(count), "Latest analysed frame", "👥")
    with m2: metric_card("Estimated wait", f"{wait} min" if wait is not None else "—", f"At {service_rate:g} people/min", "⏱️")
    with m3: metric_card("Queue status", status, f"Alert threshold: {queue_threshold}", emoji)
    with m4:
        peak = st.session_state.detection_summary.get("max_count", count)
        metric_card("Peak observed", str(peak), "Highest sampled frame count", "📈")

    if count >= queue_threshold:
        st.error("Crowd threshold reached. Consider opening an additional counter or guiding arrivals to another available service point.")
    elif count >= max(1, int(queue_threshold * .6)):
        st.warning("Queue is getting busy. Monitor service speed and prepare staff support.")
    else:
        st.success("Queue is currently below the configured crowd threshold.")

    if st.session_state.history:
        summary = st.session_state.detection_summary
        st.markdown("### 🧠 Adaptive learning snapshot")
        prior_runs = load_saved_runs()
        if prior_runs:
            baseline = round(sum(float(r.get("avg_count", 0)) for r in prior_runs) / len(prior_runs), 1)
            current_avg = float(summary.get("avg_count", count))
            delta = round(current_avg - baseline, 1)
            st.info(f"Saved-data baseline: {baseline} people on average across {len(prior_runs)} saved run(s). This clip averages {current_avg}; difference from baseline: {delta:+.1f}. This is a learned historical baseline, not fine-tuning of the YOLO model.")
        else:
            st.info("This is the first run in this workspace. Save the analysis to start building a historical baseline across videos.")
        df = pd.DataFrame(st.session_state.history)
        if "people" in df.columns:
            fig = px.area(df, x="sample", y="people", title="Queue size across sampled video frames",
                          color_discrete_sequence=["#e87961"])
            fig.update_layout(paper_bgcolor="#fffefa", plot_bgcolor="#fffefa", font_color="#25332f",
                              margin=dict(l=15,r=15,t=55,b=15), height=300)
            fig.update_traces(line=dict(width=3), fillcolor="rgba(232,121,97,.18)")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇ Download frame analytics CSV", df.to_csv(index=False).encode("utf-8"),
                               file_name="queuewise_frame_analytics.csv", mime="text/csv")
            save_col, report_col = st.columns(2)
            with save_col:
                if st.button("💾 Save this analysis to dashboard", use_container_width=True, disabled=st.session_state.analysis_saved):
                    saved = {
                        "saved_at": datetime.now().isoformat(timespec="seconds"),
                        "video": str(summary.get("video", "Uploaded video")),
                        "frames": int(summary.get("frames", len(df))),
                        "latest_count": int(count),
                        "max_count": int(summary.get("max_count", count)),
                        "avg_count": float(summary.get("avg_count", count)),
                        "estimated_wait_min": wait,
                        "status": status,
                        "threshold": int(queue_threshold),
                        "service_rate_per_min": float(service_rate),
                        "history": df.to_dict(orient="records"),
                    }
                    persist_run(saved)
                    st.session_state.analysis_saved = True
                    st.success("Analysis saved locally. Open Saved reports to review or download it.")
                    st.rerun()
            with report_col:
                try:
                    from reportlab.lib import colors
                    from reportlab.lib.pagesizes import letter, landscape
                    from reportlab.lib.styles import getSampleStyleSheet
                    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
                    pdf_buffer = io.BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
                    styles = getSampleStyleSheet()
                    story = [Paragraph("QueueWise AI — Queue Analysis Report", styles["Title"]), Spacer(1, 10)]
                    details = [["Video", str(summary.get("video", "Uploaded video"))], ["Generated", datetime.now().strftime("%Y-%m-%d %H:%M")], ["Sampled frames", str(summary.get("frames", len(df)))], ["Latest people count", str(count)], ["Peak people count", str(summary.get("max_count", count))], ["Average people count", str(summary.get("avg_count", count))], ["Estimated wait", f"{wait} min" if wait is not None else "Unavailable"], ["Queue status", status], ["Alert threshold", str(queue_threshold)]]
                    table = Table([["Metric", "Result"]] + details, colWidths=[150, 520])
                    table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#263d35")), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("GRID", (0,0), (-1,-1), .4, colors.HexColor("#d8ded8")), ("BACKGROUND", (0,1), (-1,-1), colors.HexColor("#fffefa")), ("PADDING", (0,0), (-1,-1), 7)]))
                    story.extend([table, Spacer(1, 14), Paragraph("Frame-by-frame observations", styles["Heading2"])])
                    frame_rows = [["Timestamp", "People detected"]] + [[str(r.get("sample", "")), str(r.get("people", ""))] for r in df.to_dict(orient="records")]
                    frame_table = Table(frame_rows, repeatRows=1, colWidths=[220, 150])
                    frame_table.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#dce9dc")), ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#d8ded8")), ("PADDING", (0,0), (-1,-1), 5)]))
                    story.append(frame_table)
                    story.append(Spacer(1, 10))
                    story.append(Paragraph("Note: YOLO uses pretrained weights. Historical baseline comparisons are analytics, not model retraining. Wait time is an estimate based on the configured service rate.", styles["BodyText"]))
                    doc.build(story)
                    st.download_button("📄 Download clear PDF report", pdf_buffer.getvalue(), file_name="queuewise_analysis_report.pdf", mime="application/pdf", use_container_width=True)
                except ImportError:
                    st.warning("Install reportlab to enable PDF reports: pip install reportlab")

elif page == "Insights & action advisor":
    st.markdown('<div class="section-label">Decision support workspace</div>', unsafe_allow_html=True)
    count = int(st.session_state.last_count)
    wait = estimate_wait(count, service_rate)
    status, emoji, level = status_for(count, queue_threshold)
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.markdown("### 🧠 Smart Queue Action Advisor")
        if count >= queue_threshold:
            st.markdown(f"""<div class="insight"><strong>Open / activate another counter</strong><br><span class="small-muted">Detected {count} people, above your {queue_threshold}-person threshold. If staffing permits, open an additional counter.</span></div>""", unsafe_allow_html=True)
            st.markdown("""<div class="insight"><strong>Guide new arrivals</strong><br><span class="small-muted">Use clear signage or a staff member to direct visitors to an available service point.</span></div>""", unsafe_allow_html=True)
        elif count >= max(1, int(queue_threshold*.6)):
            st.markdown("""<div class="insight"><strong>Prepare staff support</strong><br><span class="small-muted">Queue is trending toward the configured alert level. Review service speed and counter availability.</span></div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="insight"><strong>Maintain current operations</strong><br><span class="small-muted">The latest detected count is below the configured threshold. Continue monitoring.</span></div>""", unsafe_allow_html=True)
        st.markdown("### 🌿 Accessibility & service support")
        st.write("Provide a manual assistance-request option for visitors who need accessible service. This prototype does not infer anyone's age or disability from video.")
        assist = st.checkbox("Log an assistance request for staff")
        if assist:
            st.success("Assistance request noted in this session. Connect to a real notification service for operational alerts.")
    with b:
        st.markdown("### 📊 Queue pulse")
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=count,
            title={"text": "People in latest sampled frame"},
            gauge={"axis":{"range":[0, max(queue_threshold*1.5, 30)]},
                   "bar":{"color":"#e87961"},
                   "steps":[{"range":[0,queue_threshold*.6],"color":"#e4efe5"},
                            {"range":[queue_threshold*.6,queue_threshold],"color":"#f7e7c6"},
                            {"range":[queue_threshold,max(queue_threshold*1.5,30)],"color":"#f4d2c8"}],
                   "threshold":{"line":{"color":"#263d35","width":4},"thickness":.75,"value":queue_threshold}}
        ))
        fig.update_layout(paper_bgcolor="#fffefa", font_color="#25332f", height=300, margin=dict(l=20,r=20,t=55,b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("### ⏳ Waiting-time scenario planner")
        scenario_count = st.slider("What if the queue grows to… people?", 0, 100, max(count, 25))
        scenario_rate = st.slider("Service rate for scenario (people/min)", .5, 10.0, service_rate, .5)
        scenario_wait = estimate_wait(scenario_count, scenario_rate)
        st.metric("Scenario wait estimate", f"{scenario_wait} minutes" if scenario_wait is not None else "Unavailable")
        st.caption("Simple planning estimate: queue count ÷ assumed service rate. Real wait time varies with service complexity and arrivals.")

    st.markdown("### 🕰️ Peak-hour analytics")
    hist = st.session_state.history
    if hist and len(hist) > 1:
        d = pd.DataFrame(hist)
        fig = px.bar(d, x="sample", y="people", title="Observed people count by sampled timestamp",
                     color="people", color_continuous_scale=["#dce9dc", "#e87961"])
        fig.update_layout(paper_bgcolor="#fffefa", plot_bgcolor="#fffefa", font_color="#25332f",
                          margin=dict(l=15,r=15,t=55,b=15), height=300, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Analyse a video in Live monitor to populate historical queue observations. A single clip can show observed peaks, but reliable peak-hour forecasting needs data collected across multiple days.")

elif page == "Saved reports":
    st.markdown('<div class="section-label">Your saved queue intelligence</div>', unsafe_allow_html=True)
    saved_runs = load_saved_runs()
    if not saved_runs:
        st.info("No saved analyses yet. Run a video in Live monitor and choose ‘Save this analysis to dashboard’ after it finishes.")
    else:
        overview = pd.DataFrame([{ "Saved at": r.get("saved_at", ""), "Video": r.get("video", ""), "Latest": r.get("latest_count", 0), "Peak": r.get("max_count", 0), "Average": r.get("avg_count", 0), "Wait (min)": r.get("estimated_wait_min", "—"), "Status": r.get("status", "") } for r in saved_runs])
        st.metric("Saved analyses", len(saved_runs))
        st.dataframe(overview, use_container_width=True, hide_index=True)
        st.download_button("⬇ Download all saved summaries (CSV)", overview.to_csv(index=False).encode("utf-8"), file_name="queuewise_saved_summaries.csv", mime="text/csv")
        selected_idx = st.selectbox("Choose a saved analysis", list(range(len(saved_runs))), format_func=lambda i: f"{i+1}. {saved_runs[i].get('video','Video')} — {saved_runs[i].get('saved_at','')}")
        selected = saved_runs[selected_idx]
        st.markdown("#### Selected report details")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Latest count", selected.get("latest_count", 0))
        c2.metric("Peak count", selected.get("max_count", 0))
        c3.metric("Average count", selected.get("avg_count", 0))
        c4.metric("Estimated wait", f"{selected.get('estimated_wait_min')} min" if selected.get("estimated_wait_min") is not None else "—")
        detail_df = pd.DataFrame(selected.get("history", []))
        if not detail_df.empty:
            st.plotly_chart(px.line(detail_df, x="sample", y="people", markers=True, title="Saved video: queue count over time"), use_container_width=True)
            st.download_button("⬇ Download selected frame data", detail_df.to_csv(index=False).encode("utf-8"), file_name=f"queuewise_saved_run_{selected_idx+1}.csv", mime="text/csv")
        st.caption("Saved reports are stored in queuewise_data/saved_runs.json on the computer/server running Streamlit. They are not automatically synced between devices.")

else:
    st.markdown("### What QueueWise AI does")
    st.write("QueueWise AI is a prototype for intelligent queue monitoring. It combines pretrained YOLO person detection with queue analytics and an interactive Streamlit dashboard.")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Included in this starter")
        st.markdown("- Pretrained YOLO person detection\n- Video upload and sampled-frame analysis\n- Detection preview and density overlay\n- Queue count and configurable crowd threshold\n- Approximate waiting-time calculator\n- Action recommendations\n- Interactive charts and CSV export")
    with col2:
        st.markdown("#### Important limitations")
        st.markdown("- Counts all visible detected people; it does not yet distinguish queue members from passers-by.\n- Person tracking IDs are not yet enabled in this starter.\n- Wait time is a simple estimate based on a manually configured service rate.\n- True live CCTV requires an accessible RTSP stream and network permissions.\n- Peak-hour forecasting requires multiple days of saved observations.")
    st.markdown("#### Recommended next build")
    st.write("Add a queue-region selector, ByteTrack IDs, true RTSP stream handling, session/database storage, and validated waiting-time estimation. Keep camera credentials private and test with consent-approved footage.")

st.markdown("---")
st.markdown('<div style="display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#7a867f;font-size:12px"><span>QUEUEWISE AI · HACKATHON PROTOTYPE</span><span>Detect thoughtfully. Serve better.</span></div>', unsafe_allow_html=True)
