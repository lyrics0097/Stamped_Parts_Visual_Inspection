import os

# 修改为你的标签文件夹路径
label_dir = r"./datasets/labels/train"
label_dir_val = r"./datasets/labels/val"

def fix_labels(folder):
    for txt in os.listdir(folder):
        if not txt.endswith(".txt"):
            continue
        path = os.path.join(folder, txt)
        with open(path,"r",encoding="utf-8") as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts)>0:
                parts[0] = "0"   # 将类别强制改为0
                new_lines.append(" ".join(parts)+"\n")
        with open(path,"w",encoding="utf-8") as f:
            f.writelines(new_lines)

fix_labels(label_dir)
fix_labels(label_dir_val)
print("全部标签类别号已修复为0")
