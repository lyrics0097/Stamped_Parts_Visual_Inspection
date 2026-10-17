import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO

if __name__ == '__main__':
    # 所有训练代码全部写在这里面
    model = YOLO("yolov8n.pt")
    model.train(
        data="my_data.yaml",
        epochs=30,
        batch=4,
        imgsz=640,
        workers=0   # Windows直接设置workers=0，关闭多进程，根治这个报错
    )
