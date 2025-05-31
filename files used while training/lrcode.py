from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO("yolov8n.pt")
    
    model.train(
        data="data.yaml", 
        epochs=10,
        batch=8, 
        imgsz=640,
        device=0,
        fraction=0.0043,  
        name="lr0-001",
        lr0 = 0.001
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()