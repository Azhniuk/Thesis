# Automated Sperm Cell Analysis Using YOLOv10

[![YOLOv10](https://img.shields.io/badge/YOLOv10-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-red.svg)](https://streamlit.io)

**Bachelor's Thesis** | Riga Technical University | 2025

Automated computer vision system for detecting, counting, and classifying sperm cells in microscopic videos in IVF.

## Results

- **97% Detection Accuracy** (mAP50-95) using YOLOv10-Medium
- **744× Faster** than manual counting by experts
- **Real-time Processing** at 41.77 FPS
- **Web application** with an integrated model

## Problem & Solution

**Problem**: Manual sperm analysis in IVF is time-consuming (30-60 minutes), subjective, and requires expensive equipment 

**Solution**: Automated detection and classification of three sperm cell types:
- **Normal Sperm** (optimal for fertilization)
- **Sperm Clusters** (multiple cells grouped together)  
- **Small/Pinhead Sperm** (morphologically abnormal)


### Start
```bash
git clone https://github.com/yourusername/sperm-analysis.git
pip install ultralytics streamlit opencv-python pandas numpy matplotlib
```

### Run Web Application
```bash
streamlit run final_thesis_results/ui-final.py
```

## 📊 Technical Details

- **Model**: YOLOv10-Medium (16.5M parameters, 31.98 MB)
- **Dataset**: VISEM-Tracking (29,196 images)
- **Training**: 100 epochs, batch size 16, 1024px resolution
- **Hardware**: NVIDIA H100 (Digital Ocean cloud)


## 📁 Repository Structure

```
├── final_thesis_results/
│   ├── best.pt              # Trained model weights
│   ├── final_code.py        # Training script
│   └── ui-final.py          # Streamlit web app
├── validation_fps/          # Model validation
└── Azhniuk_Smite.docx      # Complete thesis document
```


**Author**: Sofiia Azhniuk  
**Institution**: Riga Technical University  
**Supervisor**: Dr.sc.ing. Katrīna Šmite

---

*Making IVF more accessible through AI-powered automation*
