from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO("yolov10m.pt")  
    
    model.train(
        data="data.yaml", 
        epochs=100,
        batch=16, 
        imgsz = 1024,
        patience=3,  
        device=0,  # Use GPU 0
        name="16-1024-YOLO10m", 
        lr0=0.005,  
        mosaic=1,  
        iou=0.6,  
        dropout=0.1,  
        
    )

if __name__ == "__main__":
    main()
