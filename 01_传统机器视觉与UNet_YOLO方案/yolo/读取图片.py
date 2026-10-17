import cv2
import os

# =========修改这两个路径========
input_folder = r"C:/Users/wujid/Desktop/yolo_test3/img_input"   #原始图片文件夹
output_folder = r"C:/Users/wujid/Desktop/yolo_test3/img_output" #输出保存文件夹
# ===============================

# 输出文件夹不存在就自动创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹内图片
img_list = os.listdir(input_folder)

count = 0
for filename in img_list:
    # 只处理图片格式
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    full_in_path = os.path.join(input_folder, filename)
    frame = cv2.imread(full_in_path)

    if frame is None:
        print(f"读取失败：{filename}")
        continue

    # 如果需要BGR转RGB就打开下面这行；YOLO训练不需要转，注释掉即可
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    full_out_path = os.path.join(output_folder, filename)
    cv2.imwrite(full_out_path, frame)
    count += 1

print(f"批量处理完成，一共处理 {count} 张图片，输出目录：{output_folder}")
