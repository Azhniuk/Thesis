import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="Sperm cell detection", page_icon="🔬", layout="wide")

st.title("Sperm cell detection")
st.markdown("Upload a video to detect and count sperm cells")
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
model_path = "best.pt"

conf_threshold = 0.7

def process_video_realtime(video_path, model_path, conf_threshold):
    model = YOLO(model_path)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        st.error("Error opening video file")
        return None
    stframe = st.empty()
    cell_count_display = st.empty()
    frame_cell_counts = []
    frame_idx = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        results = model(frame, conf=conf_threshold, verbose=False)
        num_cells = len(results[0].boxes)
        frame_cell_counts.append(num_cells)
        annotated_frame = frame.copy()
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls.cpu().numpy()[0])
            color = colors.get(cls_id, (255, 0, 0))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        stframe.image(annotated_frame, channels="BGR")
        cell_count_display.metric("## Current cell count", num_cells)
        frame_idx += 1
    video.release()
    return frame_cell_counts

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    if st.button("Press to process video"):
        st.subheader("The result is below")
        col1, col2 = st.columns([3, 1])
        with col1:
            video_display = st.empty()
        with col2:
            count_display = st.empty()
            count_display.markdown("## Cell Count")
            current_count = st.empty()
            st.markdown("### Class Legend")
            class_counts = {}
            class_names = {
                0: "Normal sperm",
                1: "Sperm clusters",
                2: "Small or pinhead cells"
            }
            class_metrics = {}
            for cls_id, name in class_names.items():
                class_metrics[cls_id] = st.empty()
        def process_with_layout(video_path, model_path, conf_threshold):
            try:
                model = YOLO(model_path)
                st.success("Model loaded successfully")
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                st.error("Error opening video file")
                return None
            frame_cell_counts = []
            frame_idx = 0
            colors = {
                0: (0, 255, 0),
                1: (0, 255, 255),
                2: (0, 0, 255)
            }
            class_names = {
                0: "Normal sperm",
                1: "Sperm clusters",
                2: "Small or pinhead cells"
            }
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break
                results = model(frame, conf=conf_threshold, verbose=False)
                class_counts = {0: 0, 1: 0, 2: 0}
                annotated_frame = frame.copy()
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls.cpu().numpy()[0])
                    class_counts[cls_id] += 1
                    color = colors.get(cls_id, (255, 0, 0))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                total_cells = sum(class_counts.values())
                frame_cell_counts.append(total_cells)
                video_display.image(annotated_frame, channels="BGR")
                current_count.metric("Total Cells Detected", total_cells)
                for cls_id, count in class_counts.items():
                    color_names = {0: "🟢", 1: "🟡", 2: "🔴"}
                    color_emoji = color_names.get(cls_id, "⚪")
                    class_metrics[cls_id].metric(
                        f"{color_emoji} {class_names[cls_id]}",
                        count
                    )
                frame_idx += 1
            video.release()
            return frame_cell_counts
        frame_counts = process_with_layout(video_path, model_path, conf_threshold)
        if frame_counts:
            st.subheader("Cell Count Statistics")
            st.metric("Total Frames Analyzed", len(frame_counts))
            st.metric("Max Cells Detected in One Frame", max(frame_counts))
            st.metric("Average Cells Per Frame", round(sum(frame_counts)/len(frame_counts), 2))
            st.line_chart(frame_counts)
        os.remove(video_path)
