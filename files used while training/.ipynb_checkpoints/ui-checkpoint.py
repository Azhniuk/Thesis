import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="Sperm Cell Detection", page_icon="🔬",layout="wide")
st.title("Sperm Cell Detection")
st.markdown("Upload a video to detect and count spe8rm cells")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
model_path = "best.pt"
conf_threshold = 0.5

def process_video_realtime(video_path, model_path, conf_threshold):
    model = YOLO(model_path)
    st.success("Model loaded successfully")

        
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        st.error("Error opening video file")
        return None
        
    stframe = st.empty()  
    cell_count_display = st.empty()
    
    frame_cell_counts = []
    frame_idx = 0
    
    colors = {
        0: (0, 255, 255),    # Yellow for class 0
        1: (0, 0, 255),      # Red for class 1
        2: (255, 0, 0)       # Blue for class 2
    }
    
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
        cell_count_display.metric("## Current Cell Count", num_cells)
        frame_idx += 1
    
    video.release()
    return frame_cell_counts

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
        
    if st.button("Press to process video"):
        st.subheader("Live Detection Output")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            video_display = st.empty()
        
        with col2:
            count_display = st.empty()
            count_display.markdown("## Cell Count")
            current_count = st.empty()
        
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
                0: (0, 255, 255),    # Cyan for class 0
                1: (0, 0, 255),      # Red for class 1
                2: (255, 0, 0)       # Blue for class 2
            }
            
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
                    
                    # Get class ID
                    cls_id = int(box.cls.cpu().numpy()[0])
                    
                    color = colors.get(cls_id, (255, 0, 0))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                video_display.image(annotated_frame, channels="BGR")
                current_count.metric("Cells Detected", num_cells)                
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