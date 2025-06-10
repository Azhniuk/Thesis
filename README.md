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

[![Video](https://img.youtube.com/vi/lV3IrCqE9bE/0.jpg)](https://www.youtube.com/watch?v=lV3IrCqE9bE)

## Problem & Solution

**Problem**: Manual sperm analysis in IVF is time-consuming (30-60 minutes), subjective, and requires expensive equipment 

**Solution**: Automated detection and classification of three sperm cell types:
- **Normal Sperm** (optimal for fertilization)
- **Sperm Clusters** (multiple cells)  
- **Small/Pinhead Sperm** (morphologically abnormal)


### Start
To use the developed model, you first need to clone the repository and install the Ultralytics library.
```bash
git clone https://github.com/yourusername/sperm-analysis.git
pip install ultralytics streamlit opencv-python pandas numpy matplotlib
```

### Run Web Application
Then, open the terminal from the folder where you cloned the repository and run the following code.
```bash
streamlit run final_thesis_results/ui-final.py
```
Make sure that you have downloaded not only the UI file but also the weights of the final model. These two files are essential for the program to work properly.

### At that stage, the program should be ready to use.
You could upload the video and test the performance of the developed model

## 📊 Technical Details

- **Model**: YOLOv10-Medium (16.5M parameters, 31.98 MB)
- **Dataset**: VISEM-Tracking (29,196 images)
- **Training**: 100 epochs, batch size 16, 1024px resolution
- **Hardware**: NVIDIA H100 (Digital Ocean cloud)


## 📁 Repository Structure

```
├── all final model metrics, weights, etc
│   ├── F1_curve
│   ├── confusion matrix
│   ├── many other files/metrics/results of the final model performance
├── files used while training
│   ├── lr0-plots.ipynb
│   ├── model-version-selection.ipynb
│   ├── and other: directory management, graphs
├── final_thesis_results, selected/
│   ├── best.pt              # Trained model weights
│   ├── final_code.py        
│   └── ui-final.py          # Streamlit web app    
└── validation_fps/   
```


**Author**: Sofiia Azhniuk  
**Institution**: Riga Technical University  
**Supervisor**: Dr.sc.ing. Katrīna Šmite

---

*You were once the quickest sperm cell*
