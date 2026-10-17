"""RealSense D455 抓拍 + SAM 分割 + 目标物理宽度测量。

流程:
    1. 预览与抓拍: 开启 Color / Depth 数据流, 深度强制对齐到彩色坐标系 (rs.align);
       实时预览彩色画面, 按 's' 抓拍 (锁定彩色帧 + 对齐后的 16 位深度矩阵 + 彩色相机内参), 随后关闭相机流。
    2. SAM 分割: 以画面中心点 (w//2, h//2) 作为正样本 Point Prompt, 对抓拍的静态图生成目标二值 Mask。
       权重固定为 C:\\Users\\lyz\\Desktop\\sam3\\sam_b.pt (SAM 1 ViT-B), 使用 Ultralytics 的基础 SAM Predictor。
    3. 测量计算: Mask 外轮廓 -> cv2.minAreaRect -> 长边像素长度;
       读取中心点深度 Z (米), 按小孔成像公式 物理宽度 = 像素宽度 * Z / fx 换算为毫米。
    4. 结果展示: 叠加半透明 Mask、最小外接矩形与尺寸文字, 自动保存到桌面 "test input.jpg";
       另将 SAM 抠出的目标 (背景涂黑 + 按外接框裁剪) 保存为 "test input_crop.jpg", 两张图同时弹窗显示。

运行 (rad_env 环境, 需先 pip install pyrealsense2):
    python realsense_sam_measure.py
    python realsense_sam_measure.py --device cuda:0
"""
import argparse
import os

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics.models.sam import Predictor as SAMPredictor

# SAM 1 (ViT-B) 权重路径 (硬编码)
SAM_WEIGHT = r'C:\Users\lyz\Desktop\sam3\sam_b.pt'
# SAM 1 默认网络输入边长
SAM_IMGSZ = 1024

PREVIEW_WIN = 'RealSense Preview  [s] capture   [q/Esc] quit'
RESULT_WIN = 'Measurement Result  [any key] close'
CROP_WIN = 'SAM Crop'
PICK_WIN = 'Snapshot  [drag] box the target   [Enter/Space] confirm   [c] cancel'


# ================================================================================================================
# 1. 预览与抓拍阶段
# ================================================================================================================
def capture_rgbd(width=1280, height=720, fps=30, warmup_frames=30):
    """打开 D455 实时预览, 按 's' 抓拍。预览中鼠标左键点击目标可移动提示点, 右键恢复为画面中心。

    返回:
        color_rgb   : (H, W, 3) uint8, RGB 顺序的彩色图
        depth_raw   : (H, W) uint16, 已对齐到彩色图像素坐标的原始深度值 (单位 = depth_scale 米)
        intrinsics  : rs.intrinsics, 彩色相机内参 (fx, fy, ppx, ppy ...)
        depth_scale : float, 深度原始值 -> 米 的换算系数 (D455 默认 0.001, 即原始值单位为毫米)
        prompt_pt   : (x, y), SAM 提示点 (默认画面中心)
    若用户按 q / Esc 或关闭窗口放弃抓拍, 返回 None。
    """
    click = {'pt': None}   # None 表示使用画面中心

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click['pt'] = (x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            click['pt'] = None

    pipeline = rs.pipeline()
    config = rs.config()
    # 彩色流直接要求 rgb8 格式, 抓拍结果即为 RGB; OpenCV 显示时再转 BGR
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    # 深度流为 z16: 每个像素一个 16 位无符号整数
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    profile = pipeline.start(config)
    try:
        # 深度比例因子: depth_raw * depth_scale = 米
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # 深度相机与彩色相机是两个物理镜头, 视角和位置都不同。
        # rs.align(rs.stream.color) 把深度图重投影到彩色相机坐标系, 对齐后 depth[v, u] 与 color[v, u] 是同一个空间点,
        # 这样用彩色图上分割出的像素去查深度才有意义, 并且之后应使用"彩色相机"的内参做换算。
        align = rs.align(rs.stream.color)

        # 刚上电时自动曝光 / 白平衡尚未收敛, 丢弃前若干帧
        for _ in range(warmup_frames):
            pipeline.wait_for_frames()

        cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(PREVIEW_WIN, on_mouse)
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)            # 执行深度 -> 彩色对齐
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()     # 这是"对齐后"的深度帧
            if not color_frame or not depth_frame:
                continue  # 偶发丢帧, 跳过本次循环

            color_rgb = np.asanyarray(color_frame.get_data())
            h, w = color_rgb.shape[:2]
            cx, cy = click['pt'] or (w // 2, h // 2)

            # ---------- 预览画面: 叠加提示点十字与实时深度 (只画在副本上, 不污染抓拍数据) ----------
            preview = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            cv2.drawMarker(preview, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
            z_live = depth_frame.get_distance(cx, cy)   # 单点深度 (米), 0 表示该点无有效深度
            cv2.putText(preview, f'point Z = {z_live:.3f} m' if z_live > 0 else 'point Z = invalid',
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(preview, "Left-click target (right-click: center), press 's' to capture",
                        (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(PREVIEW_WIN, preview)

            # ---------- 键盘监听 ----------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # 抓拍: 必须 .copy()。get_data() 返回的是 SDK 内部帧缓冲的视图,
                # 帧被释放 / 相机流关闭后这块内存会失效, 复制后才是独立的静态数据。
                color_snap = color_rgb.copy()
                depth_snap = np.asanyarray(depth_frame.get_data()).copy()   # uint16 深度矩阵
                # 彩色相机内参 (对齐后深度与彩色共用这一套内参)
                intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
                print(f'[抓拍] 分辨率 {w}x{h}, fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, '
                      f'ppx={intrinsics.ppx:.2f}, ppy={intrinsics.ppy:.2f}, depth_scale={depth_scale}, '
                      f'提示点=({cx}, {cy})')
                return color_snap, depth_snap, intrinsics, depth_scale, (cx, cy)
            if key in (ord('q'), 27):
                print('[预览] 用户取消抓拍')
                return None
            # 用户直接点窗口右上角 X 关闭时同样视为取消
            if cv2.getWindowProperty(PREVIEW_WIN, cv2.WND_PROP_VISIBLE) < 1:
                print('[预览] 预览窗口已关闭')
                return None
    finally:
        # 无论正常抓拍、取消还是异常, 都要停止数据流, 释放 USB 相机资源
        pipeline.stop()
        cv2.destroyAllWindows()
        print('[相机] 数据流已关闭')


# ================================================================================================================
# 2. SAM 分割阶段
# ================================================================================================================
def build_sam_predictor(device):
    """加载 SAM 1 (sam_b.pt) 权重, 构建 Ultralytics 的基础 SAM Predictor (交互式 point / box prompt 分割)。"""
    print('正在加载 SAM 1 (sam_b.pt) 模型进行快速物理链路验证...')
    if not os.path.isfile(SAM_WEIGHT):
        raise FileNotFoundError(f'SAM 1 权重不存在: {SAM_WEIGHT}')
    overrides = dict(conf=0.0, task='segment', mode='predict', imgsz=SAM_IMGSZ, model=SAM_WEIGHT,
                     device=device, save=False, verbose=False)
    predictor = SAMPredictor(overrides=overrides)
    predictor.setup_model()   # 立即加载权重, 避免抓拍后才等待模型加载
    print(f'[SAM] 已加载 SAMPredictor: {SAM_WEIGHT} (device={device})')
    return predictor


def _fill_holes(mask):
    """填充 Mask 内部空洞 (如铅笔盒表面反光、印花导致的漏分割), 保证外轮廓完整。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _component_at(mask, pt):
    """只保留包含提示点的那个连通域, 去掉 SAM 可能顺带分出的远处零碎区域。"""
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    lab = labels[pt[1], pt[0]]
    if lab == 0:  # 提示点恰好落在 Mask 外 (极少见), 退而取面积最大的连通域
        areas = [(labels == i).sum() for i in range(1, n)]
        lab = 1 + int(np.argmax(areas)) if areas else 0
    return (labels == lab).astype(np.uint8) * 255 if lab else mask


def select_box_on_snapshot(color_rgb, max_tries=3):
    """在定格的抓拍图上按住鼠标左键拖框框住目标, Enter / 空格确认。

    细长物体 (如铅笔芯盒) 用单点提示很容易点偏到背后的人 / 衣服上, 框提示能明确告诉 SAM 目标的范围, 稳定得多。
    没拖框就按了确认 (或按 c) 会重新弹出让你再框, 连续 max_tries 次都没框才返回 None 退回单点提示。
    """
    base = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    for i in range(max_tries):
        img = base.copy()
        tip = 'Hold LEFT mouse button and DRAG a box around the target, release, then press Enter'
        if i > 0:
            tip = f'No box drawn! Try again ({i + 1}/{max_tries}): ' + tip
        cv2.putText(img, tip, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        x, y, w, h = cv2.selectROI(PICK_WIN, img, showCrosshair=False, fromCenter=False)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        if w >= 2 and h >= 2:
            box = (int(x), int(y), int(x + w), int(y + h))
            print(f'[框选] SAM 框提示 = {box}')
            return box
        print(f'[框选] 第 {i + 1} 次未框选到目标')
    print('[框选] 多次未框选, 改用单点提示')
    return None


def sam_segment_box(predictor, color_rgb, box, margin=10):
    """以框 (x0, y0, x1, y1) 作为 SAM 提示分割目标, 返回 (mask uint8 0/255, 框中心点, SAM 置信度)。"""
    h, w = color_rgb.shape[:2]
    x0, y0, x1, y1 = box
    img_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    predictor.set_image(img_bgr)
    # 框提示本身已无歧义, 单输出即可
    res = predictor(bboxes=[[float(x0), float(y0), float(x1), float(y1)]], multimask_output=False)[0]
    predictor.reset_image()

    m = cv2.resize(res.masks.data[0].cpu().numpy().astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    # 只保留框 (外扩 margin) 以内的部分, 防止 Mask 沿相似颜色泄漏到框外
    keep = np.zeros_like(m)
    keep[max(0, y0 - margin):y1 + margin, max(0, x0 - margin):x1 + margin] = 255
    m = cv2.bitwise_and(m, keep)
    # 只保留最大连通域并填洞
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        m = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255
    m = _fill_holes(m)
    score = float(res.boxes.conf[0])
    print(f'[SAM] 框提示分割: score={score:.3f}, area={(m > 0).mean():.4f}')
    return m, ((x0 + x1) // 2, (y0 + y1) // 2), score


def sam_segment_center(predictor, color_rgb, pt=None, depth_m=None, max_area_ratio=0.8, min_score=0.7,
                       depth_tol_m=0.06, min_depth_consistency=0.7):
    """以提示点 (默认画面中心) 作为正样本提示, 返回 (mask uint8 0/255, 提示点, SAM 置信度)。

    depth_m: 对齐后的深度图 (米, 0 = 无效)。提供时, 候选 Mask 内的有效深度必须有 min_depth_consistency 以上
             落在提示点深度 ±depth_tol_m 内, 用来排除"提示点在手持物体上, 候选却扩展到后方人体 / 背景"的情况。
    """
    h, w = color_rgb.shape[:2]
    pt = pt or (w // 2, h // 2)

    # Ultralytics 的 SAM 预测器约定输入为 cv2 风格的 BGR 图, 因此把抓拍的 RGB 转换一次
    img_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    predictor.set_image(img_bgr)                          # 图像编码 (最耗时的一步, 每张图只做一次)
    res = predictor(points=[[float(pt[0]), float(pt[1])]],
                    labels=[1],                          # 1 = 前景点 (正样本)
                    multimask_output=True)[0]             # 输出 3 个候选: 子部件 / 部件 / 整体
    predictor.reset_image()

    masks = res.masks.data.cpu().numpy() > 0.5            # (N, H', W') bool
    scores = res.boxes.conf.cpu().numpy()                 # 每个候选的预测 IoU 置信度

    # 形态学开运算核: 大小随图像尺寸自适应, 用于去除边缘毛刺
    k = max(3, int(0.01 * min(h, w)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    cands = []
    for m, s in zip(masks, scores):
        m = cv2.resize(m.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        m = _fill_holes(_component_at(m, pt))
        cands.append((m, float(s), float((m > 0).mean())))

    # 提示点处的参考深度: 7x7 邻域有效深度中位数; 取不到则不启用深度一致性过滤
    z0 = None
    if depth_m is not None:
        x, y = pt
        win = depth_m[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        if (win > 0).any():
            z0 = float(np.median(win[win > 0]))

    def depth_consistency(m):
        if z0 is None:
            return 1.0
        d = depth_m[(m > 0) & (depth_m > 0)]
        return float((np.abs(d - z0) <= depth_tol_m).mean()) if d.size else 0.0

    cons = [depth_consistency(c[0]) for c in cands]

    # 3 个候选由小到大通常是: 局部花纹/标签 -> 部件 -> 整个物体。测量要的是"整个物体",
    # 所以在置信度够高、不至于覆盖大半画面 (那通常是把桌面 / 背景当成了目标)、且深度与提示点一致
    # (不会越过物体边缘扩展到前后其它物体) 的候选中取面积最大者; 若没有候选满足条件, 退而取置信度最高者。
    valid = [c for c, k in zip(cands, cons)
             if c[1] >= min_score and 0 < c[2] <= max_area_ratio and k >= min_depth_consistency]
    mask, score, area = max(valid, key=lambda c: c[2]) if valid else max(cands, key=lambda c: c[1])
    z_info = f'提示点深度 {z0:.3f} m' if z0 is not None else '提示点深度无效, 未启用深度过滤'
    print(f'[SAM] {z_info}; 候选(score, 面积占比, 深度一致率)='
          f'{[(round(c[1], 3), round(c[2], 3), round(k, 2)) for c, k in zip(cands, cons)]} '
          f'-> 选中 score={score:.3f}, area={area:.3f}')
    return mask, pt, score


# ================================================================================================================
# 3. 测量与计算阶段
# ================================================================================================================
def measure_long_side_px(mask):
    """提取 Mask 最大外轮廓并求最小外接旋转矩形, 返回 (rect, 长边像素长度, 短边像素长度)。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError('Mask 中没有找到任何轮廓, SAM 分割失败')
    cnt = max(contours, key=cv2.contourArea)
    # rect = ((中心x, 中心y), (边长1, 边长2), 旋转角度); 边长1/2 的顺序与角度定义有关, 所以用 max/min 取长短边
    rect = cv2.minAreaRect(cnt)
    (rw, rh) = rect[1]
    return rect, float(max(rw, rh)), float(min(rw, rh))


def read_center_depth_m(depth_raw, depth_scale, pt, mask=None, radius=3):
    """读取中心点 Z 距离 (米)。

    单个像素的深度很容易为 0 (无效: 遮挡 / 反光 / 超量程) 或带随机噪声,
    因此取中心点 (2*radius+1)^2 邻域内所有有效深度的中位数, 比单像素读数稳健得多。
    若整个邻域都无效, 退而使用 Mask 区域内所有有效深度的中位数。
    """
    x, y = pt
    h, w = depth_raw.shape
    win = depth_raw[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
    valid = win[win > 0]
    src = f'中心 {win.shape[1]}x{win.shape[0]} 邻域中位数'
    if valid.size == 0 and mask is not None:
        valid = depth_raw[(mask > 0) & (depth_raw > 0)]
        src = 'Mask 区域中位数 (中心点深度无效, 已回退)'
    if valid.size == 0:
        raise RuntimeError('中心点及目标区域均无有效深度, 请调整距离 (D455 建议 0.4 m ~ 6 m) 或避开强反光表面')
    z_m = float(np.median(valid)) * depth_scale   # 原始值 * 比例因子 = 米
    print(f'[深度] Z = {z_m:.4f} m  ({src}, 有效像素 {valid.size})')
    return z_m


def pixel_to_mm(pixel_len, z_m, fx):
    """小孔成像: 物理尺寸 = 像素尺寸 * 深度 / 焦距(像素)。z 以米计, 结果 *1000 换算为毫米。"""
    return pixel_len * z_m / fx * 1000.0


# ================================================================================================================
# 4. 静态结果展示与保存
# ================================================================================================================
def draw_result(color_rgb, mask, rect, pt, width_mm, long_px, z_m, alpha=0.45):
    """在抓拍图上叠加半透明 Mask、最小外接矩形、提示点和测量文字, 返回 BGR 结果图。"""
    vis = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

    # ---------- 半透明彩色 Mask: 仅在目标像素上做 原图*(1-alpha) + 颜色*alpha 的混合 ----------
    color_layer = np.zeros_like(vis)
    color_layer[:] = (255, 144, 30)                      # BGR: 亮蓝色
    blended = cv2.addWeighted(vis, 1 - alpha, color_layer, alpha, 0)
    m = mask > 0
    vis[m] = blended[m]

    # ---------- 最小外接矩形轮廓 ----------
    box = cv2.boxPoints(rect).astype(np.int32)           # 旋转矩形的 4 个角点
    cv2.drawContours(vis, [box], 0, (0, 255, 0), 3)
    # 用黄色加粗标出被测量的"长边"
    edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
    p1, p2 = max(edges, key=lambda e: np.linalg.norm(e[0] - e[1]))
    cv2.line(vis, tuple(map(int, p1)), tuple(map(int, p2)), (0, 255, 255), 5, cv2.LINE_AA)

    # ---------- 提示点 ----------
    cv2.drawMarker(vis, pt, (0, 0, 255), cv2.MARKER_CROSS, 30, 3)

    # ---------- 文字标注 (OpenCV Hershey 字体不支持中文, 图上统一用英文) ----------
    h, w = vis.shape[:2]
    scale = max(0.8, w / 1280 * 1.3)
    lines = [(f'Width: {width_mm:.1f} mm', (0, 255, 255), scale * 1.4),
             (f'Pixel length: {long_px:.1f} px', (255, 255, 255), scale),
             (f'Depth Z: {z_m * 1000:.1f} mm', (255, 255, 255), scale)]
    x0, y = 20, 20
    for text, color, s in lines:
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2)
        y += th + base + 10
        # 半透明黑底, 保证文字在任何背景上都清晰可读
        y_top, y_bot = y - th - 8, y + base + 4
        roi = vis[y_top:y_bot, x0 - 8:x0 + tw + 8]
        vis[y_top:y_bot, x0 - 8:x0 + tw + 8] = (roi * 0.35).astype(np.uint8)
        cv2.putText(vis, text, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, s, color, 2, cv2.LINE_AA)
    return vis


def crop_by_mask(color_rgb, mask, margin=10):
    """SAM 抠图: Mask 外的像素置为纯黑, 再按 Mask 的外接框 (四周留 margin 像素) 裁剪, 返回 BGR 图。"""
    cut = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    cut[mask == 0] = (0, 0, 0)
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + 1 + margin)
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + 1 + margin)
    return cut[y0:y1, x0:x1]


def save_to_desktop(img_bgr, filename='test input.jpg'):
    """保存结果图到当前 Windows 用户桌面: %USERPROFILE%\\Desktop\\<filename>。"""
    desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')
    os.makedirs(desktop, exist_ok=True)
    path = os.path.join(desktop, filename)
    # cv2.imwrite 在 Windows 上无法写入含中文等非 ASCII 字符的路径, 改用 imencode 编码后由 numpy 写盘
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError('JPEG 编码失败')
    buf.tofile(path)
    print(f'文件已成功保存至: {path}')
    return path


# ================================================================================================================
# 主流程
# ================================================================================================================
def main():
    ap = argparse.ArgumentParser(description='RealSense D455 抓拍 + SAM 分割 + 物理宽度测量')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                    help='推理设备, 例如 cpu / cuda:0')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--crop_only', action='store_true',
                    help='只做 SAM 抠图裁剪 (保存 test input_crop.jpg), 跳过宽度测量')
    args = ap.parse_args()

    # 先加载模型再开相机: 这样按下 's' 之后可以立即推理, 不用再等权重加载
    predictor = build_sam_predictor(args.device)

    # ---------- 1. 预览与抓拍 ----------
    snap = capture_rgbd(args.width, args.height, args.fps)
    if snap is None:
        return
    color_rgb, depth_raw, intr, depth_scale, prompt_pt = snap
    debug_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'realsense_debug')
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imencode('.png', cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))[1].tofile(os.path.join(debug_dir, 'last_color.png'))
    np.save(os.path.join(debug_dir, 'last_depth_m.npy'), depth_raw.astype(np.float32) * depth_scale)

    # ---------- 2. SAM 分割 ----------
    box = select_box_on_snapshot(color_rgb) if args.crop_only else None
    if box is not None:
        mask, pt, score = sam_segment_box(predictor, color_rgb, box)
    else:
        mask, pt, score = sam_segment_center(predictor, color_rgb, prompt_pt,
                                             depth_m=depth_raw.astype(np.float32) * depth_scale)

    if args.crop_only:
        crop = crop_by_mask(color_rgb, mask)
        save_to_desktop(crop, 'test input_crop.jpg')
        cv2.imshow(CROP_WIN, crop)
        cv2.waitKey(0)        # 等待任意键关闭
        cv2.destroyAllWindows()
        return

    # ---------- 3. 测量与计算 ----------
    rect, long_px, short_px = measure_long_side_px(mask)
    z_m = read_center_depth_m(depth_raw, depth_scale, pt, mask)
    # 按需求使用 fx 换算长边。D455 彩色相机 fx 与 fy 几乎相等 (像素近似正方形), 因此长边倾斜时误差可忽略
    width_mm = pixel_to_mm(long_px, z_m, intr.fx)
    short_mm = pixel_to_mm(short_px, z_m, intr.fx)
    print(f'[测量] 长边 {long_px:.1f} px, Z = {z_m:.4f} m, fx = {intr.fx:.2f} px '
          f'-> 物理宽度 = {long_px:.1f} * {z_m:.4f} / {intr.fx:.2f} = {width_mm:.1f} mm '
          f'(短边约 {short_mm:.1f} mm)')

    # ---------- 4. 绘制、保存、显示 ----------
    vis = draw_result(color_rgb, mask, rect, pt, width_mm, long_px, z_m)
    save_to_desktop(vis, 'test input.jpg')
    crop = crop_by_mask(color_rgb, mask)
    save_to_desktop(crop, 'test input_crop.jpg')

    cv2.namedWindow(RESULT_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(RESULT_WIN, vis)
    cv2.imshow(CROP_WIN, crop)
    cv2.waitKey(0)            # 等待任意键关闭
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
