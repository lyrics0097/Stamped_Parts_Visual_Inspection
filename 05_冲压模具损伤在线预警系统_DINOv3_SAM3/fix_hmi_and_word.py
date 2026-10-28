# -*- coding: utf-8 -*-
"""
1. 重新生成《00_一图看懂监控大屏指南.png》(hmi_single_explained.png)：
   直接从已修正曲线的《08_佳能1080p连续36冲次四分屏预警演示视频.mp4》(或新渲染的 LEVEL 2 帧) 中提取单帧，
   确保右下角第④格的蓝线（重现率）和红线（划痕长度）是平滑上升的正确版本（与图2完全一致！），并加上清晰的四格中文说明框。
2. 修正 Word 生成脚本：
   - 修复图 1（使用修正后的 rising curve 版 00_一图看懂监控大屏指南.png）
   - 修复图 3（正确指向 02_监控大屏四阶段状态全览.png，彻底消除“[图片未找到]”空白问题）
   - 精简第三部分新相机介绍：去掉还没实际开展的复杂标定大图与参数表，重点说明“目前刚拿到新相机，下一步需要用它来干哪三件事”。
"""
import os
import io
import shutil
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = r"c:\Users\lyz\Desktop\sam3结合dinov3方案"
SHOWCASE_DIR = os.path.join(PROJECT_ROOT, "临时展示")
SYS_DIR = os.path.join(PROJECT_ROOT, "冲压模具损伤在线预警系统")
ARTIFACT_DIR = r"C:\Users\lyz\.gemini\antigravity\brain\899b1654-5eda-497b-83bb-bfd5ea1ce2ef"
SCRATCH_DIR = os.path.join(ARTIFACT_DIR, "scratch")


def imread_u(path: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(raw, flags)


def copy_file_bin(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as fsrc:
        data = fsrc.read()
    with open(dst, "wb") as fdst:
        fdst.write(data)


def regenerate_00_hmi_single_explained():
    """从最新的 02_监控大屏四阶段状态全览.png (右上/左下 LEVEL 2 帧) 或 08_佳能视频中提取带上升曲线的单帧，绘制 4 个通俗气泡框"""
    mp4_path = os.path.join(SHOWCASE_DIR, "08_佳能1080p连续36冲次四分屏预警演示视频.mp4")
    cap = cv2.VideoCapture(mp4_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 取第 18 冲次对应的帧 (大约在 50% 进度处，此时 LEVEL 2 橙色预警已触发，右下角蓝线从 0.1 爬升到 0.9，红线从 9.2mm 爬升到 15.4mm)
    target_f = int(n_frames * 0.52)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        raise RuntimeError("无法从 08 视频读取单帧")

    # 使用 PIL 绘制圆角半透明说明标签框，保证不遮挡右下角上升曲线主体
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGBA")
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_path = r"C:\Windows\Fonts\msyhbd.ttc"
    if not os.path.exists(font_path):
        font_path = r"C:\Windows\Fonts\simhei.ttf"
    font_title = ImageFont.truetype(font_path, 19)
    font_body = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc" if os.path.exists(r"C:\Windows\Fonts\msyh.ttc") else font_path, 16)

    def draw_callout(x0, y0, x1, y1, bg_rgba, border_rgb, title_str, lines):
        draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill=bg_rgba, outline=border_rgb + (255,), width=3)
        draw.text((x0 + 12, y0 + 8), title_str, font=font_title, fill=(255, 255, 255, 255))
        cy = y0 + 36
        for ln in lines:
            draw.text((x0 + 12, cy), ln, font=font_body, fill=(240, 248, 255, 255))
            cy += 23

    # 1. 左上格说明框 (放在左下角空白皮带区)
    draw_callout(
        28, 345, 495, 455,
        bg_rgba=(13, 110, 110, 220),
        border_rgb=(0, 230, 210),
        title_str="【左上格①：拍标准照 + 算冲压件倾斜角度】",
        lines=[
            "• 绿色外框：自动抓拍走到画面正中央的运动冲压件",
            "• 右上小字：实时算出工件表面倾斜角与距离，修正透视变形",
        ],
    )

    # 2. 右上格说明框 (放在右上格左下方)
    draw_callout(
        825, 345, 1345, 455,
        bg_rgba=(153, 27, 27, 220),
        border_rgb=(255, 100, 100),
        title_str="【右上格②：AI 抓出微细划痕 + 当场量出毫米长宽】",
        lines=[
            "• 红/橙色框：DINOv3+SAM3 自动框出冲压件表面的拉毛划痕",
            "• 框旁文字：直接标出这道划痕的真实物理长度 L 与宽度 W (mm)",
        ],
    )

    # 3. 左下格说明框 (放在棋盘格左上方空白格处)
    draw_callout(
        28, 525, 555, 658,
        bg_rgba=(30, 58, 138, 225),
        border_rgb=(96, 165, 250),
        title_str="【左下格③：模具表面的 12×12 电子地图（核心发明点！）】",
        lines=[
            "• 偶发绿圈（下方两处）：第2、6件偶发磕碰（只闪1次，自动过滤不报警）",
            "• 红色高亮格（R5,C6）：连续每件都在同一位置刮伤 → 实锤模具该点位拉毛！",
            "• 物理绑定：12×12 网格与现场蓝体检具 #4~#23 号检测点位一一对应",
        ],
    )

    # 4. 右下格说明框 (放在折线图顶部上方，完全不遮挡下方爬升的蓝线与红线)
    draw_callout(
        825, 505, 1375, 615,
        bg_rgba=(88, 28, 135, 225),
        border_rgb=(192, 132, 252),
        title_str="【右下格④：双线爬升趋势 + 四级报警信号灯】",
        lines=[
            "• 蓝线（左轴）：同位置重现率（偶发仅0.1；模具真坏了平滑爬升至 0.9~1.0）",
            "• 红线（右轴mm）：同位置连续≥2次才激活，随模具拉毛恶化从 9mm 涨到 16mm",
        ],
    )

    out_pil = Image.alpha_composite(pil_img, overlay).convert("RGB")
    out_bgr = cv2.cvtColor(np.array(out_pil), cv2.COLOR_RGB2BGR)

    # 加上顶部总标题条
    banner_h = 54
    canvas = np.full((out_bgr.shape[0] + banner_h, out_bgr.shape[1], 3), (13, 18, 26), dtype=np.uint8)
    canvas[banner_h:, :] = out_bgr
    pil_c = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw_c = ImageDraw.Draw(pil_c)
    font_ban = ImageFont.truetype(font_path, 24)
    draw_c.text(
        (280, 12),
        "《冲压模具损伤在线预警系统》四分屏工业监控大屏 —— 一图看懂四大窗口与上升趋势线",
        font=font_ban,
        fill=(255, 255, 255),
    )

    tmp_out = os.path.join(SCRATCH_DIR, "_fixed_00_hmi_single.png")
    pil_c.save(tmp_out, format="PNG")

    copy_file_bin(tmp_out, os.path.join(SHOWCASE_DIR, "00_一图看懂监控大屏指南.png"))
    copy_file_bin(tmp_out, os.path.join(ARTIFACT_DIR, "hmi_single_explained.png"))
    print("Successfully regenerated 00_一图看懂监控大屏指南.png with rising curves!")


if __name__ == "__main__":
    regenerate_00_hmi_single_explained()
