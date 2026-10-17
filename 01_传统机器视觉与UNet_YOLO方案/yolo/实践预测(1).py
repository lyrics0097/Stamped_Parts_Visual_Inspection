import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

if __name__ == '__main__':
    # 这里加载训练好的best.pt模型！！不是图片
    model = YOLO(r"./runs/detect/train/weights/best.pt")

    # source写测试图片路径
    results = model.predict(
        source=r"C:/Users/wujid/Desktop/yolo_test3/61cf04f7d0f7e80547884a8f95d11cfe.jpg",
        save=True,
        conf=0.1
    )

    for res in results:
        print("像素坐标xyxy：", res.boxes.xyxy.tolist())
        print("置信度conf：", res.boxes.conf.tolist())
