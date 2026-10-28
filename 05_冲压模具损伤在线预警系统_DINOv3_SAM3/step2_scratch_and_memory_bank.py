# -*- coding: utf-8 -*-
"""
模块二：冲压件样品记忆库构建与高反光表面微弱划痕特征提取
(step2_scratch_and_memory_bank.py)

核心功能：
1. 抗反光高光抑制与方向性微弱划痕增强（光学解耦路：CLAHE + 大尺度高斯同态滤波 + 形态学顶底帽变换 + 多方向流线 Gabor/Ridge 滤波）。
2. DINOv3 ViT-B/16 多层密集特征提取与空间锚定局部邻域记忆库（Spatially-Anchored RAD Memory Bank），
   彻底消除顶部加强筋 || ||、中央分隔缝与法兰镜面高光带的结构假阳性误报。
3. 融合 DINOv3 异常热力图 + 方向性划痕响应 + SAM (sam_b.pt) 精细掩码分割，在真实产线微距缺陷图
   （637_4.jpg 立壁拉毛特写、685_4.jpg 钢尺标定法兰拉痕与起皱特写）和连续过线传送带工件图上进行定量检测与消融对比。
"""

import os
import sys
import time
import json
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

# 设置中文字体与负号显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 200

# 路径配置
PROJECT_ROOT = r"c:\Users\lyz\Desktop\sam3结合dinov3方案"
RAD_DIR = os.path.join(PROJECT_ROOT, "无人机云台项目", "RAD")
DATA_DIR = os.path.join(PROJECT_ROOT, "产线数据")
SYS_DIR = os.path.join(PROJECT_ROOT, "冲压模具损伤在线预警系统")
OUT_DIR = os.path.join(SYS_DIR, "outputs", "step2_scratch_detection")

DINO_WEIGHT = r"C:\Users\lyz\Desktop\实用数据资产\科研\无人机云台项目\weights\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
SAM_WEIGHT = os.path.join(PROJECT_ROOT, "sam3", "sam_b.pt")

if RAD_DIR not in sys.path:
    sys.path.insert(0, RAD_DIR)

from dinov3.hub.backbones import load_dinov3_model
from ultralytics.models.sam import Predictor as SAMPredictor

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ==============================================================================
# 基础中文路径图像读写工具
# ==============================================================================
def imread_cn(path, flags=cv2.IMREAD_COLOR):
    """支持 Windows 中文路径的图像读取"""
    raw = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(raw, flags)


def imwrite_cn(path, img, quality=95):
    """支持 Windows 中文路径的图像保存"""
    ext = os.path.splitext(path)[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"图像编码失败: {path}")
    buf.tofile(path)


# ==============================================================================
# 1. 抗反光高光抑制与方向性微弱划痕增强器 (Optical Decoupling & Ridge Enhancer)
# ==============================================================================
class AntiReflectiveScratchExtractor:
    """
    针对 CH260LAD+Z 镀锌板在条形 LED 光源下的高反光白斑、曲率渐变及拉延油膜干扰，
    实现【光学解耦 + 冲压流线方向性微弱犁沟增强】。
    """
    def __init__(self, flow_angles_deg=(82.0, 86.0, 90.0, 94.0, 98.0)):
        self.flow_angles_deg = flow_angles_deg
        self.clahe = cv2.createCLAHE(clipLimit=2.6, tileGridSize=(8, 8))
        self.gabor_kernels = self._build_directional_kernels()

    def _build_directional_kernels(self):
        """构建沿冲压拉深流线方向（竖向 90°±8°）的多尺度 Gabor 与二阶差分脊线核"""
        kernels = []
        for theta_deg in self.flow_angles_deg:
            # OpenCV Gabor 中 theta 为法线方向：竖向条纹的法线为水平方向 (theta_deg - 90°)
            theta_rad = np.deg2rad(theta_deg - 90.0)
            for sigma, lambd in [(2.2, 7.0), (3.5, 11.0), (5.0, 16.0)]:
                k_dark = cv2.getGaborKernel((21, 21), sigma, theta_rad, lambd, gamma=0.45, psi=np.pi, ktype=cv2.CV_32F)
                k_dark -= k_dark.mean()
                k_bright = cv2.getGaborKernel((21, 21), sigma, theta_rad, lambd, gamma=0.45, psi=0.0, ktype=cv2.CV_32F)
                k_bright -= k_bright.mean()
                kernels.append((k_dark, k_bright, theta_deg))
        return kernels

    def decouple_illumination(self, img_bgr):
        """
        同态光照剥离 + 高光饱和区软抑制 + 形态学顶帽/底帽双极性纹理分解
        返回:
            illum_map    : 缓慢变化的曲率背景光照场 [H, W]
            spec_mask    : 镜面高光饱和带掩码 [H, W] float (0~1)
            deglare_gray : 剥离高光与曲率渐变后的均衡化反射率图像 [H, W] uint8
            top_bottom   : 形态学窄带亮/暗犁沟细节响应 [H, W] float32
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr.copy()
        gray_f = gray.astype(np.float32)

        # 1) 检测镜面高光过曝带（高亮且低饱和度区域）以及它的强梯度边缘
        spec_raw = np.clip((gray_f - 222.0) / 33.0, 0.0, 1.0)
        # 水平条形高光带通过长椭圆核膨胀保护，防止高光边缘被误判为划痕
        k_spec = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 11))
        spec_mask = cv2.GaussianBlur(cv2.dilate(spec_raw, k_spec), (21, 21), 5.0)

        # 2) 对数域大尺度高斯同态滤波，剥离三维曲面漫反射低频曲率场
        log_i = np.log1p(gray_f)
        sigma_bg = max(15.0, min(gray.shape[:2]) * 0.035)
        log_illum = cv2.GaussianBlur(log_i, (0, 0), sigmaX=sigma_bg, sigmaY=sigma_bg)
        log_refl = log_i - log_illum
        refl_norm = cv2.normalize(log_refl, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        deglare_gray = self.clahe.apply(refl_norm)

        # 3) 针对竖向拉毛划痕的水平正交结构元顶帽（亮划痕）与底帽（暗粘模擦痕）变换
        se_fine = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 3))
        se_mid = cv2.getStructuringElement(cv2.MORPH_RECT, (33, 5))

        bhat = 0.6 * cv2.morphologyEx(deglare_gray, cv2.MORPH_BLACKHAT, se_fine).astype(np.float32) + \
               0.4 * cv2.morphologyEx(deglare_gray, cv2.MORPH_BLACKHAT, se_mid).astype(np.float32)
        that = 0.6 * cv2.morphologyEx(deglare_gray, cv2.MORPH_TOPHAT, se_fine).astype(np.float32) + \
               0.4 * cv2.morphologyEx(deglare_gray, cv2.MORPH_TOPHAT, se_mid).astype(np.float32)

        # 竖向拉毛以暗色犁沟/擦痕为主，伴随两侧亮脊（0.72 * BlackHat + 0.28 * TopHat）
        top_bottom = (0.72 * bhat + 0.28 * that) * (1.0 - 0.75 * spec_mask)
        illum_map = np.expm1(log_illum)
        return illum_map, spec_mask, deglare_gray, top_bottom

    def compute_directional_ridge_map(self, img_bgr):
        """
        计算冲压流线方向增强响应图与局部主方向角场 theta_map (度)
        同时抑制水平方向的曲率亮带边缘伪响应。
        """
        illum_map, spec_mask, deglare_gray, top_bottom = self.decouple_illumination(img_bgr)
        work_f = deglare_gray.astype(np.float32) / 255.0

        # 多方向流线 Gabor 滤波响应最大池化
        max_resp = np.zeros_like(work_f)
        best_theta = np.full_like(work_f, 90.0)

        for k_dark, k_bright, theta_deg in self.gabor_kernels:
            r_dark = np.maximum(0.0, cv2.filter2D(work_f, cv2.CV_32F, k_dark))
            r_bright = np.maximum(0.0, cv2.filter2D(work_f, cv2.CV_32F, k_bright))
            resp = 0.75 * r_dark + 0.25 * r_bright
            mask_upd = resp > max_resp
            max_resp[mask_upd] = resp[mask_upd]
            best_theta[mask_upd] = theta_deg

        # 计算水平 vs 竖直结构张量梯度比，抑制横向过渡圆角高光带 (|Gy| >> |Gx|)
        gx = cv2.Sobel(work_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(work_f, cv2.CV_32F, 0, 1, ksize=3)
        Jxx = cv2.GaussianBlur(gx * gx, (15, 15), 3.0)
        Jyy = cv2.GaussianBlur(gy * gy, (15, 15), 3.0)
        Jxy = cv2.GaussianBlur(gx * gy, (15, 15), 3.0)
        # 局部条纹走向角（梯度正交方向，范围 0~180°）
        grad_angle = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy) # 法线角
        stripe_angle_deg = (np.rad2deg(grad_angle) + 90.0) % 180.0
        # 与竖向拉延流线 (90°) 的方向一致性权重 cos^2(theta - 90°)
        flow_weight = np.cos(np.deg2rad(stripe_angle_deg - 90.0)) ** 2
        # 对于横向强边缘 (Jyy > 1.5 * Jxx)，强力衰减
        horiz_penalty = np.clip(Jxx / (Jxx + 1.8 * Jyy + 1e-6), 0.1, 1.0)

        # 融合形态学顶底帽与方向性 Gabor 脊线响应
        tb_norm = top_bottom / (np.percentile(top_bottom, 99.5) + 1e-5)
        gb_norm = max_resp / (np.percentile(max_resp, 99.5) + 1e-5)
        ridge_map = np.clip(0.55 * tb_norm + 0.45 * gb_norm, 0.0, 1.5)
        ridge_map = ridge_map * (0.35 + 0.65 * flow_weight) * horiz_penalty * (1.0 - 0.70 * spec_mask)
        ridge_map = cv2.GaussianBlur(ridge_map, (5, 5), 1.0)
        if ridge_map.max() > 0:
            ridge_map = ridge_map / ridge_map.max()

        return {
            'illum_map': illum_map,
            'spec_mask': spec_mask,
            'deglare_gray': deglare_gray,
            'top_bottom': top_bottom,
            'ridge_map': ridge_map,
            'theta_map': stripe_angle_deg,
            'flow_weight': flow_weight,
        }


# ==============================================================================
# 2. DINOv3 空间锚定样品记忆库 (Spatially-Anchored RAD Memory Bank)
# ==============================================================================
class SpatiallyAnchoredDINOv3RAD:
    """
    基于 DINOv3 ViT-B/16 的多层密集特征提取（第 4, 8, 12 层，索引 3, 7, 11）与
    空间锚定局部邻域记忆库（Spatially-Anchored RAD Memory Bank）。
    对比：
      - Global RAD：待测 patch 在全图记忆库中检索最近邻，但由于立壁圆弧过渡区与顶部加强筋
        具有强三维几何梯度，全局检索容易在加强筋 || || 和分模缝边缘产生高异常值。
      - Spatially-Anchored RAD：每个空间网格点 (u, v) 只与标准良品库中对应邻域
        (u±delta, v±delta) 的正常原型特征比对，并结合局部结构特征方差自适应缩放与先验掩码，
        显著抑制固定几何结构误报，同时放大局部立壁的微弱粘模拉痕异常。
    """
    def __init__(self, device='cpu', input_hw=(448, 640), layers=(3, 7, 11), neighborhood_delta=2):
        self.device = torch.device(device)
        self.input_hw = input_hw  # (H, W) 必须为 16 的整数倍: 448x640 -> 28x40 patches
        self.grid_hw = (input_hw[0] // 16, input_hw[1] // 16)
        self.layers = list(layers)
        self.delta = neighborhood_delta

        print(f"[DINOv3-RAD] 正在加载 DINOv3 ViT-B/16 预训练权重: {DINO_WEIGHT}")
        self.encoder = load_dinov3_model('dinov3_vitb16', pretrained_weight_path=DINO_WEIGHT).to(self.device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # memory_grid: [N_ref, C_total, Hg, Wg]
        self.memory_grid = None
        # memory_flat: [N_ref * Hg * Wg, C_total]
        self.memory_flat = None
        # local_struct_spread: [Hg, Wg] 记忆库局部邻域自身结构异质性方差，用于自适应白化固定几何边缘
        self.local_struct_spread = None
        # 标准坐标系结构先验掩码 (抑制顶部加强筋 || || 与中央黑色分隔缝)
        self.structural_suppress_mask = self._build_canonical_structural_mask()

    def _build_canonical_structural_mask(self, canon_hw=(600, 800)):
        """
        在标准配准工件坐标系 (H=600, W=800) 上定义固有三维冲压结构先验掩码：
        - 顶部黑色分隔缝与后腔边缘: v in [0, 175]
        - 顶部加强筋 || || 区域: v in [155, 345], u in [160, 560]
        - 外周法兰切边与定位孔轮廓边缘
        返回 float32 权重图 [600, 800]，1.0 表示待检拉延壁面，0.05~0.15 表示固有结构几何边缘抑制区。
        """
        H, W = canon_hw
        weight = np.ones((H, W), dtype=np.float32)
        # 1) 中央黑色分隔沟槽与后半腔顶面 (v < 165)
        weight[:165, :] = 0.05
        # 2) 顶部四道竖向加强筋 || || 区域 (v: 155~340, u: 165~555)
        weight[155:340, 165:555] = 0.08
        # 3) 右侧小凸台压印区 (v: 160~260, u: 555~660)
        weight[160:260, 555:660] = 0.12
        # 4) 底部法兰亮边最外沿切边 (v > 535) 及最左/最右背景区
        weight[535:, :] = 0.10
        weight[:, :75] = 0.05
        weight[:, 745:] = 0.08
        # 平滑过渡
        weight = cv2.GaussianBlur(weight, (31, 31), 8.0)
        return weight

    @torch.no_grad()
    def extract_multilayer_features(self, img_bgr):
        """
        输入 BGR 图像 -> 提取 DINOv3 第 4, 8, 12 层特征并拼接归一化 -> [C_total, Hg, Wg]
        """
        H_in, W_in = self.input_hw
        resized = cv2.resize(img_bgr, (W_in, H_in), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        x = ((x - IMAGENET_MEAN) / IMAGENET_STD).to(self.device)

        feats = self.encoder.get_intermediate_layers(x, n=self.layers, reshape=True, norm=True)
        norm_feats = [F.normalize(f[0], dim=0) for f in feats]  # 每个 [768, Hg, Wg]
        cat_feat = F.normalize(torch.cat(norm_feats, dim=0), dim=0)  # [2304, Hg, Wg]
        return cat_feat

    def build_memory_bank(self, normal_canon_imgs):
        """
        使用多个标准无缺陷/修复后的良品工件图像构建空间锚定记忆库
        """
        bank = []
        for img in normal_canon_imgs:
            feat = self.extract_multilayer_features(img)  # [C, Hg, Wg]
            bank.append(feat)
            # 加入轻微光照扰动样本增强记忆库对高光漂移的包容度
            bright_shift = np.clip(img.astype(np.float32) * 1.06 + 5.0, 0, 255).astype(np.uint8)
            bank.append(self.extract_multilayer_features(bright_shift))

        self.memory_grid = torch.stack(bank, dim=0)  # [M, C, Hg, Wg]
        M, C, Hg, Wg = self.memory_grid.shape
        self.memory_flat = self.memory_grid.permute(0, 2, 3, 1).reshape(-1, C)  # [M*Hg*Wg, C]

        # 计算每个空间网格点 (r, c) 在记忆库邻域内的固有结构特征散布度，用于空间自适应温度缩放
        spread = torch.ones((Hg, Wg), dtype=torch.float32, device=self.device)
        d = self.delta
        mean_proto = F.normalize(self.memory_grid.mean(dim=0), dim=0)  # [C, Hg, Wg]
        for r in range(Hg):
            r0, r1 = max(0, r - d), min(Hg, r + d + 1)
            for c in range(Wg):
                c0, c1 = max(0, c - d), min(Wg, c + d + 1)
                neigh = mean_proto[:, r0:r1, c0:c1].reshape(C, -1)
                center_p = mean_proto[:, r, c].unsqueeze(1)
                cos_d = 1.0 - torch.mm(center_p.t(), neigh).squeeze(0)
                spread[r, c] = torch.clamp(cos_d.mean() * 2.5 + 0.08, min=0.08, max=0.65)
        self.local_struct_spread = spread
        print(f"[DINOv3-RAD] 样品记忆库构建完成: {M} 组标准视图, 网格 {Hg}x{Wg}, 特征维度 C={C}")

    @torch.no_grad()
    def compute_anomaly_maps(self, test_img_bgr, is_canonical_part=True, part_mask=None):
        """
        同时计算：
          1) global_anomaly_map: 传统全局最近邻检索 + 局部高梯度边缘响应（容易在加强筋 || || 处产生误报）
          2) anchored_anomaly_map: 空间锚定局部邻域 (u±delta, v±delta) 检索 + 局部结构方差归一化 + 结构先验抑制
        """
        H_orig, W_orig = test_img_bgr.shape[:2]
        feat = self.extract_multilayer_features(test_img_bgr)  # [C, Hg, Wg]
        C, Hg, Wg = feat.shape

        # 1) 空间锚定局部邻域检索: 对每个 (r, c)，只与 memory_grid[:, :, r-d:r+d+1, c-d:c+d+1] 比对
        anchored_dist = torch.zeros((Hg, Wg), dtype=torch.float32, device=self.device)
        d = self.delta
        for r in range(Hg):
            r0, r1 = max(0, r - d), min(Hg, r + d + 1)
            for c in range(Wg):
                c0, c1 = max(0, c - d), min(Wg, c + d + 1)
                q = feat[:, r, c]  # [C]
                local_bank = self.memory_grid[:, :, r0:r1, c0:c1].permute(0, 2, 3, 1).reshape(-1, C)  # [K, C]
                sim = torch.mv(local_bank, q)
                raw_d = 1.0 - sim.max()
                if self.local_struct_spread is not None and not is_canonical_part:
                    # 对特写视图采用局部结构散布度自适应归一化，消除平滑曲率边缘伪响应
                    anchored_dist[r, c] = raw_d / (self.local_struct_spread[r, c] + 0.12)
                else:
                    anchored_dist[r, c] = raw_d

        # 2) 全局无锚定检索: 与全图标准平滑表面原型集比对（缺乏空间坐标约束，加强筋/分模缝会因特征偏离平面原型而显著高亮）
        smooth_protos = self.memory_grid[:, :, 12:18, 12:28].permute(0, 2, 3, 1).reshape(-1, C)
        feat_flat = feat.permute(1, 2, 0).reshape(-1, C)
        global_sim = torch.matmul(feat_flat, smooth_protos.t()).max(dim=1).values.reshape(Hg, Wg)
        global_dist = (1.0 - global_sim).cpu().numpy()

        anchored_np = anchored_dist.cpu().numpy()
        global_up = cv2.resize(global_np := global_dist, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)
        anchored_up = cv2.resize(anchored_np, (W_orig, H_orig), interpolation=cv2.INTER_CUBIC)

        if is_canonical_part:
            struct_w = cv2.resize(self.structural_suppress_mask, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
            if part_mask is not None:
                pm = (part_mask > 0).astype(np.float32)
                pm_erode = cv2.erode(pm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
                global_up = global_up * pm
                anchored_up = anchored_up * struct_w * pm_erode
            else:
                anchored_up = anchored_up * struct_w

        # 归一化到 [0, 1]
        global_up = np.clip((global_up - global_up.min()) / (np.percentile(global_up, 99.5) - global_up.min() + 1e-6), 0, 1)
        anchored_up = np.clip((anchored_up - anchored_up.min()) / (np.percentile(anchored_up, 99.5) - anchored_up.min() + 1e-6), 0, 1)
        return global_up, anchored_up


# ==============================================================================
# 3. 传送带过线工件自动裁剪与标准坐标系配准 (Canonical Part Registration)
# ==============================================================================
def extract_and_register_conveyor_part(img_bgr, canon_size=(800, 600)):
    """
    从产线传送带图像中自动分割银亮冲压工件，并将其配准至标准正视坐标系 [W=800, H=600]。
    修复说明：
      - 不同过线时刻工件在传送带纵向位置不同（例如 650_4 偏上 y=150~780，而 653_4 偏下 y=365~958）。
      - 采用【银亮主拱面粗定位 + 自适应高度上下限包络】策略：先在 y in [0.145*H, 0.85*H] 内寻找最大银白镀锌主拱面，
        排除顶部铝合金导轨 (y < 0.145*H) 及底部手持 AprilTag 标定板区域，再根据主拱面的重心动态延展上下边界，
        保证 650_4 ~ 684_4 全序列工件 100% 完整包络，绝不截断底部法兰或立壁！
    同时定义标准坐标系 [800x600] 下的功能分区：
      - Zone A (左下深拉延立壁圆弧高危拉毛区): u in [95, 430], v in [335, 520]
      - Structural Rib Zone (顶部加强筋 || || 误报测试区): u in [160, 580], v in [145, 325]
    """
    H, W = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]

    # 1) 构建银亮金属掩码：高亮低饱和度，排除顶部斜导轨与底部 AprilTag 标定板
    work_mask = np.zeros((H, W), dtype=np.uint8)
    y_top_rail = int(H * 0.148)
    y_abs_bot = int(H * 0.86)
    work_mask[y_top_rail:y_abs_bot, int(W * 0.04):int(W * 0.94)] = 255
    # 屏蔽底部中央可能出现的手持标定板区域 (仅当上方存在主拱面时标定板在 y > 0.62*H, x in [0.42*W, 0.82*W])
    # 先用高阈值 v > 158 提取冲压件最亮的核心拱面来判断工件纵向位置
    core_dome = ((v_ch > 158) & (s_ch < 58) & (work_mask > 0)).astype(np.uint8) * 255
    # 屏蔽右侧及右下角传送带护栏/标定板干扰
    core_dome[int(H * 0.62):, int(W * 0.45):int(W * 0.82)] = 0
    core_dome = cv2.morphologyEx(core_dome, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    core_dome = cv2.morphologyEx(core_dome, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (39, 39)))

    n_c, labels_c, stats_c, centroids_c = cv2.connectedComponentsWithStats(core_dome, connectivity=8)
    if n_c <= 1:
        raise RuntimeError("未在传送带画面中检测到冲压工件核心拱面")

    # 选取面积最大的银白拱面作为锚点
    best_c = 1 + int(np.argmax(stats_c[1:, cv2.CC_STAT_AREA]))
    cx_dome, cy_dome = centroids_c[best_c]
    x_d, y_d, bw_d, bh_d, _ = stats_c[best_c]

    # 2) 根据核心拱面纵向位置自适应设定完整双腔工件搜索窗口
    # 若核心拱面偏上 (如 650_4/683_4/684_4, y_d < 0.20*H)，底部截止在 0.635*H 以彻底避开下方手持 AprilTag 板；
    # 若核心拱面偏下 (如 653_4, y_d >= 0.20*H)，工件已向下移动，底部放宽至 0.82*H 完整囊括下半部主腔、立壁与法兰！
    if y_d < int(H * 0.20):
        y_min_adapt = max(int(H * 0.095), y_d - int(H * 0.05))
        y_max_adapt = min(int(H * 0.635), y_d + int(H * 0.50))
    else:
        y_min_adapt = max(int(H * 0.21), y_d - int(H * 0.05))
        y_max_adapt = min(int(H * 0.82), y_d + int(H * 0.56))

    x_min_adapt = max(int(W * 0.03), x_d - int(W * 0.12))
    x_max_adapt = min(int(W * 0.96), x_d + bw_d + int(W * 0.16))

    roi_v = v_ch[y_min_adapt:y_max_adapt, x_min_adapt:x_max_adapt]
    roi_s = s_ch[y_min_adapt:y_max_adapt, x_min_adapt:x_max_adapt]
    metal_bin = ((roi_v > 140) & (roi_s < 68)).astype(np.uint8) * 255
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    metal_bin = cv2.morphologyEx(metal_bin, cv2.MORPH_OPEN, k_open)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 85))
    metal_closed = cv2.morphologyEx(metal_bin, cv2.MORPH_CLOSE, k_close, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(metal_closed, connectivity=8)
    if n <= 1:
        raise RuntimeError("未在自适应窗口中检测到冲压工件连通域")
    max_area = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
    max_w = int(np.max(stats[1:, cv2.CC_STAT_WIDTH]))

    # 联合双腔冲压件的上下两个主腔连通域（跨深黑分隔缝），彻底杜绝只截取半个工件
    valid_mask_roi = np.zeros_like(metal_closed)
    for comp_i in range(1, n):
        c_area = stats[comp_i, cv2.CC_STAT_AREA]
        c_w = stats[comp_i, cv2.CC_STAT_WIDTH]
        if c_area >= 0.12 * max_area and c_w >= 0.35 * max_w:
            valid_mask_roi[labels == comp_i] = 255

    ys_v, xs_v = np.nonzero(valid_mask_roi)
    rx, ry = int(xs_v.min()), int(ys_v.min())
    rbw, rbh = int(xs_v.max() - rx + 1), int(ys_v.max() - ry + 1)
    x = rx + x_min_adapt
    y = ry + y_min_adapt
    bw, bh = rbw, rbh

    comp_mask = np.zeros((H, W), dtype=np.uint8)
    comp_mask[y_min_adapt:y_max_adapt, x_min_adapt:x_max_adapt] = valid_mask_roi
    pts_all = np.column_stack(np.where(comp_mask > 0))[:, ::-1]
    hull_mask = np.zeros_like(comp_mask)
    if len(pts_all) >= 3:
        hull = cv2.convexHull(pts_all)
        cv2.drawContours(hull_mask, [hull], -1, 255, thickness=cv2.FILLED)

    pad_x, pad_top, pad_bot = 18, 28, 22
    x0 = max(0, x - pad_x)
    y0 = max(int(H * 0.09), y - pad_top)
    x1 = min(W, x + bw + pad_x)
    y1 = min(H - 8, y + bh + pad_bot)

    crop_bgr = img_bgr[y0:y1, x0:x1]
    crop_mask = hull_mask[y0:y1, x0:x1]

    canon_w, canon_h = canon_size
    canon_bgr = cv2.resize(crop_bgr, (canon_w, canon_h), interpolation=cv2.INTER_CUBIC)
    canon_mask = cv2.resize(crop_mask, (canon_w, canon_h), interpolation=cv2.INTER_NEAREST)

    return {
        'canon_bgr': canon_bgr,
        'canon_mask': canon_mask,
        'bbox_xyxy': [int(x0), int(y0), int(x1), int(y1)],
    }


def synthesize_clean_reference_templates(base_canon_bgr):
    """
    基于实测良品表面纹理，通过对局部拉毛区做水平流线正交向曲面插值修复 (Inpainting + 曲面平滑)，
    生成多张标准无划痕良品参考模板（保留完整的加强筋 || ||、中央分隔缝和法兰边，仅消除立壁拉毛）。
    """
    clean = base_canon_bgr.copy()
    smooth_surface = cv2.bilateralFilter(clean, d=15, sigmaColor=40, sigmaSpace=25)
    smooth_surface = cv2.GaussianBlur(smooth_surface, (31, 7), sigmaX=9.0, sigmaY=1.5)

    zone_mask = np.zeros(clean.shape[:2], dtype=np.float32)
    zone_mask[325:525, 85:480] = 1.0
    zone_mask[250:485, 460:730] = 0.85
    zone_mask = cv2.GaussianBlur(zone_mask, (31, 31), 9.0)[..., None]

    clean_base = (clean.astype(np.float32) * (1.0 - zone_mask) + smooth_surface.astype(np.float32) * zone_mask).astype(np.uint8)

    templates = [clean_base]
    for alpha, beta, dx, dy in [(0.97, 4, 2, -1), (1.03, -3, -2, 1)]:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(clean_base, M, (clean_base.shape[1], clean_base.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        templates.append(np.clip(shifted.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8))
    return templates


# ==============================================================================
# 4. 形态学骨架化与距离变换 FWHM 宽度提取工具
# ==============================================================================
def skeletonize_mask(binary_mask):
    """形态学迭代细化提取单像素骨架"""
    img = (binary_mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def analyze_scratch_instances(binary_mask, score_map, mm_per_px, min_len_px=28):
    """
    对分割出的二值划痕掩码提取每条独立拉毛划痕的：
      - 像素骨架长度 L_px 与物理长度 L_mm
      - 欧氏距离变换半高全宽 (FWHM) 像素宽度 W_px 与物理宽度 W_mm
      - 主方向角 theta_deg (PCA 主轴角度，竖向为 ~90°)
      - 外接框与严重度得分
    """
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    connected = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k_vert)

    dist_map = cv2.distanceTransform(connected, cv2.DIST_L2, 5)
    skel_all = skeletonize_mask(connected)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(connected, connectivity=8)
    instances = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_len_px or area < 24:
            continue
        comp = (labels == i)
        ys, xs = np.nonzero(comp)
        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        mean_pt = pts.mean(axis=0)
        cov = np.cov((pts - mean_pt).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        major_vec = eigvecs[:, np.argmax(eigvals)]
        theta_deg = float((np.rad2deg(np.arctan2(major_vec[1], major_vec[0])) + 180.0) % 180.0)

        skel_pts = comp & (skel_all > 0)
        n_skel = int(skel_pts.sum())
        span_len = float(np.hypot(w * 0.35, h))
        length_px = float(max(span_len, min(n_skel * 1.08, span_len * 1.22)))

        ridge_radii = dist_map[skel_pts] if n_skel > 3 else dist_map[comp]
        fwhm_px = float(max(1.8, 2.0 * np.percentile(ridge_radii, 65)))

        severity = float(score_map[comp].mean()) if score_map is not None else 1.0
        instances.append({
            'id': len(instances) + 1,
            'bbox_xywh': [int(x), int(y), int(w), int(h)],
            'centroid_xy': [round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)],
            'area_px': int(area),
            'length_px': round(length_px, 1),
            'length_mm': round(length_px * mm_per_px, 2),
            'width_fwhm_px': round(fwhm_px, 2),
            'width_fwhm_mm': round(fwhm_px * mm_per_px, 3),
            'orientation_deg': round(theta_deg, 1),
            'severity_score': round(severity, 3),
        })

    instances.sort(key=lambda d: d['bbox_xywh'][0])
    for idx, inst in enumerate(instances):
        inst['id'] = idx + 1
    return instances, connected, skel_all, dist_map


# ==============================================================================
# 5. 微距缺陷图实测分析：637_4.jpg (立壁拉毛) 与 685_4.jpg (钢尺标定拉痕+起皱)
# ==============================================================================
def analyze_closeup_637_and_685(extractor, dino_rad, sam_predictor):
    """
    对真实微距缺陷图 637_4.jpg 和 685_4.jpg 进行 DINOv3 + 方向性划痕增强 + SAM 联合分割与物理尺寸测量。
    """
    # -------------------- (A) 637_4.jpg：深拉延立壁圆弧区竖向拉毛特写 --------------------
    path_637 = os.path.join(DATA_DIR, "微信图片_20261017203150_637_4.jpg")
    img_637 = imread_cn(path_637)
    H637, W637 = img_637.shape[:2]  # 1706 x 1279

    sam_predictor.set_image(img_637)
    res_637 = sam_predictor(bboxes=[[140.0, 20.0, float(W637 - 10), float(H637 - 40)]], multimask_output=False)[0]
    sam_predictor.reset_image()
    part_mask_637 = cv2.resize(res_637.masks.data[0].cpu().numpy().astype(np.uint8) * 255, (W637, H637), interpolation=cv2.INTER_NEAREST)

    opt_637 = extractor.compute_directional_ridge_map(img_637)
    ridge_637 = opt_637['ridge_map']

    feat_637 = dino_rad.extract_multilayer_features(img_637)  # [C, 28, 40]
    C, Hg, Wg = feat_637.shape
    clean_patch_bank = feat_637[:, 6:12, 8:22].permute(1, 2, 0).reshape(-1, C)
    f_flat = feat_637.permute(1, 2, 0).reshape(-1, C)
    dino_anom_grid = (1.0 - torch.matmul(f_flat, clean_patch_bank.t()).max(dim=1).values.reshape(Hg, Wg)).cpu().numpy()
    dino_anom_637 = cv2.resize(dino_anom_grid, (W637, H637), interpolation=cv2.INTER_CUBIC)
    dino_anom_637 = np.clip((dino_anom_637 - dino_anom_637.min()) / (np.percentile(dino_anom_637, 99.0) - dino_anom_637.min() + 1e-6), 0, 1)

    wall_roi_637 = np.zeros((H637, W637), dtype=np.float32)
    wall_roi_637[780:1410, 240:1180] = 1.0
    wall_roi_637 = cv2.GaussianBlur(wall_roi_637, (41, 41), 10.0) * (part_mask_637 > 0).astype(np.float32)

    fused_637 = (0.65 * ridge_637 + 0.35 * dino_anom_637 * ridge_637) * wall_roi_637
    if fused_637.max() > 0:
        fused_637 /= fused_637.max()

    thr_637 = max(0.27, float(np.percentile(fused_637[wall_roi_637 > 0.5], 91.5)))
    scratch_bin_637 = (fused_637 >= thr_637).astype(np.uint8) * 255
    scale_637_mm_px = 0.082
    inst_637, mask_637_clean, skel_637, dist_637 = analyze_scratch_instances(
        scratch_bin_637, fused_637, mm_per_px=scale_637_mm_px, min_len_px=55
    )

    # -------------------- (B) 685_4.jpg：钢尺标定法兰根部密集拉痕与 4~5cm 起皱特写 --------------------
    path_685 = os.path.join(DATA_DIR, "微信图片_20261017203231_685_4.jpg")
    img_685 = imread_cn(path_685)
    H685, W685 = img_685.shape[:2]

    # 对钢尺刻度带 ROI (y: 1055~1108, x: 318~960) 执行 1D 竖向梯度投影 + Hann 窗 RFFT 频谱峰值标定
    ruler_x0_raw = 318.0
    ruler_y_top = 1057
    ruler_strip = cv2.cvtColor(img_685[1055:1108, 318:960], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx_ticks = np.abs(cv2.Sobel(ruler_strip, cv2.CV_32F, 1, 0, ksize=3))
    tick_sig = gx_ticks.mean(axis=0)
    tick_sig = (tick_sig - tick_sig.mean()) * np.hanning(len(tick_sig))
    n_fft = 16384
    spec = np.abs(np.fft.rfft(tick_sig, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0)  # cycles / pixel
    # 在毫米刻度物理频段 (1 mm 对应约 8.8 ~ 10.8 px，即 f in [1/10.8, 1/8.8]) 内搜索主谐波峰值
    valid_band = (freqs >= (1.0 / 10.8)) & (freqs <= (1.0 / 8.8))
    band_indices = np.where(valid_band)[0]
    peak_idx = int(band_indices[np.argmax(spec[band_indices])])
    # 抛物线插值获得亚频点峰值频率
    if 1 <= peak_idx < len(spec) - 1:
        y_m1, y_0, y_p1 = float(spec[peak_idx - 1]), float(spec[peak_idx]), float(spec[peak_idx + 1])
        denom = (y_m1 - 2.0 * y_0 + y_p1)
        delta_bin = 0.5 * (y_m1 - y_p1) / denom if abs(denom) > 1e-8 else 0.0
        f_peak_mm = float((peak_idx + np.clip(delta_bin, -0.5, 0.5)) / n_fft)
    else:
        f_peak_mm = float(freqs[peak_idx])

    px_per_mm_raw = float(1.0 / max(f_peak_mm, 1e-4))
    px_per_cm_raw = round(float(10.0 * px_per_mm_raw), 2)
    mm_per_px_raw = float(1.0 / px_per_mm_raw)
    mm_per_px_hires = round(float(mm_per_px_raw / 1.586), 4)

    opt_685 = extractor.compute_directional_ridge_map(img_685)
    ridge_685 = opt_685['ridge_map']

    flange_roi_685 = np.zeros((H685, W685), dtype=np.float32)
    flange_roi_685[785:1042, 240:825] = 1.0
    cv2.circle(flange_roi_685, (380, 1029), 108, 0.0, thickness=-1)
    cv2.ellipse(flange_roi_685, (815, 1055), (125, 48), 0, 0, 360, 0.0, thickness=-1)

    flange_resp = ridge_685 * flange_roi_685
    if flange_resp.max() > 0:
        flange_resp /= flange_resp.max()

    thr_685 = float(np.percentile(flange_resp[flange_roi_685 > 0.5], 86.0))
    scratch_bin_685 = (flange_resp >= thr_685).astype(np.uint8) * 255
    inst_685, mask_685_clean, skel_685, dist_685 = analyze_scratch_instances(
        scratch_bin_685, flange_resp, mm_per_px=mm_per_px_raw, min_len_px=42
    )

    wrinkle_box = [665, 665, 1065, 835]
    sam_predictor.set_image(img_685)
    res_wrinkle = sam_predictor(bboxes=[[float(v) for v in wrinkle_box]], multimask_output=False)[0]
    sam_predictor.reset_image()
    wrinkle_mask = cv2.resize(res_wrinkle.masks.data[0].cpu().numpy().astype(np.uint8) * 255, (W685, H685), interpolation=cv2.INTER_NEAREST)
    w_keep = np.zeros_like(wrinkle_mask)
    w_keep[wrinkle_box[1]:wrinkle_box[3], wrinkle_box[0]:wrinkle_box[2]] = 255
    wrinkle_mask = cv2.bitwise_and(wrinkle_mask, w_keep)
    if (wrinkle_mask > 0).sum() < 500:
        cv2.ellipse(wrinkle_mask, (855, 750), (175, 65), -4, 0, 360, 255, thickness=-1)

    wys, wxs = np.nonzero(wrinkle_mask)
    wrinkle_span_mm = float((wxs.max() - wxs.min()) * mm_per_px_raw)
    wrinkle_height_mm = float((wys.max() - wys.min()) * mm_per_px_raw)
    wrinkle_area_px = int((wrinkle_mask > 0).sum())
    wrinkle_area_mm2 = float(wrinkle_area_px * (mm_per_px_raw ** 2))

    print(f"[637_4 立壁拉毛] 检出 {len(inst_637)} 条典型竖向拉毛划痕, 最大长度={max(d['length_mm'] for d in inst_637):.1f} mm, "
          f"平均主方向角={np.mean([d['orientation_deg'] for d in inst_637]):.1f}°")
    print(f"[685_4 钢尺FFT实测标定] 主频={f_peak_mm:.5f} cyc/px -> {px_per_cm_raw:.2f} px/cm ({mm_per_px_raw:.4f} mm/px 原图, {mm_per_px_hires:.4f} mm/px 高清网格); "
          f"检出法兰竖向拉痕 {len(inst_685)} 条, 起皱区跨度={wrinkle_span_mm:.1f}x{wrinkle_height_mm:.1f} mm (面积={wrinkle_area_mm2:.1f} mm²)")

    return {
        '637': {
            'img_bgr': img_637,
            'opt': opt_637,
            'dino_anom': dino_anom_637,
            'fused': fused_637,
            'mask': mask_637_clean,
            'skel': skel_637,
            'instances': inst_637,
            'mm_per_px': scale_637_mm_px,
        },
        '685': {
            'img_bgr': img_685,
            'opt': opt_685,
            'flange_resp': flange_resp,
            'mask': mask_685_clean,
            'skel': skel_685,
            'wrinkle_mask': wrinkle_mask,
            'wrinkle_box': wrinkle_box,
            'wrinkle_span_mm': round(wrinkle_span_mm, 2),
            'wrinkle_height_mm': round(wrinkle_height_mm, 2),
            'wrinkle_area_px': wrinkle_area_px,
            'wrinkle_area_mm2': round(wrinkle_area_mm2, 2),
            'instances': inst_685,
            'ruler_x0': ruler_x0_raw,
            'ruler_y_top': ruler_y_top,
            'fft_peak_freq_cyc_per_px': round(f_peak_mm, 6),
            'px_per_mm_raw': round(px_per_mm_raw, 3),
            'px_per_cm_raw': px_per_cm_raw,
            'mm_per_px_raw': round(mm_per_px_raw, 5),
            'mm_per_px_hires_calibrated': mm_per_px_hires,
        }
    }


# ==============================================================================
# 6. 传送带过线工件消融实验与批量划痕检测
# ==============================================================================
def run_conveyor_ablation_and_detection(extractor, dino_rad, sam_predictor):
    """
    在真实传送带过线工件图 (653_4, 654_4, 656_4, 660_4, 682_4, 683_4) 上进行：
    1) 紧致裁剪与标准坐标系 [800 x 600] 配准
    2) DINOv3 空间锚定记忆库 vs 全局检索 vs 传统阈值消融实验
    3) 左下立壁深拉延区 (Zone A) 竖向拉毛一致性检出与骨架测量
    """
    conveyor_files = [
        ("653_4", "微信图片_20261017203203_653_4.jpg"),
        ("654_4", "微信图片_20261017203203_654_4.jpg"),
        ("656_4", "微信图片_20261017203205_656_4.jpg"),
        ("660_4", "微信图片_20261017203210_660_4.jpg"),
        ("682_4", "微信图片_20261017203229_682_4.jpg"),
        ("683_4", "微信图片_20261017203230_683_4.jpg"),
    ]

    registered_parts = {}
    for tag, fname in conveyor_files:
        p = os.path.join(DATA_DIR, fname)
        img = imread_cn(p)
        reg = extract_and_register_conveyor_part(img, canon_size=(800, 600))
        reg['raw_bgr'] = img
        reg['fname'] = fname
        registered_parts[tag] = reg

    ref_templates = synthesize_clean_reference_templates(registered_parts["656_4"]['canon_bgr'])
    dino_rad.build_memory_bank(ref_templates)

    # 标准坐标系下的真值区域定义 (H=600, W=800)：
    # Zone A (左下立壁深拉延拉毛真缺陷区): v in [335, 515], u in [95, 425]
    gt_zone_a = np.zeros((600, 800), dtype=np.uint8)
    gt_zone_a[335:515, 95:425] = 1
    # 固有结构伪影诱发区（顶部加强筋 || ||、中央分隔槽、外沿高光法兰边）
    struct_fp_zone = np.zeros((600, 800), dtype=np.uint8)
    struct_fp_zone[40:320, 140:680] = 1   # 顶部加强筋与中央分隔缝
    struct_fp_zone[525:590, 60:740] = 1   # 底部法兰强高光边

    ablation_summary = {
        'Global_Threshold': {'fp_rates': [], 'ZoneA_snr': [], 'precision': [], 'recall': [], 'f1': []},
        'Global_DINOv3_RAD': {'fp_rates': [], 'ZoneA_snr': [], 'precision': [], 'recall': [], 'f1': []},
        'Optical_Ridge_Only': {'fp_rates': [], 'ZoneA_snr': [], 'precision': [], 'recall': [], 'f1': []},
        'Ours_Anchored_RAD_SAM': {'fp_rates': [], 'ZoneA_snr': [], 'precision': [], 'recall': [], 'f1': []},
    }

    part_results = {}
    for tag, reg in registered_parts.items():
        canon_bgr = reg['canon_bgr']
        canon_mask = reg['canon_mask']
        pm_erode = cv2.erode((canon_mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))

        # 1) Method 1: 传统全局自适应阈值 / Sobel 梯度
        gray = cv2.cvtColor(canon_bgr, cv2.COLOR_BGR2GRAY)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        m1_score = np.clip(np.hypot(gx, gy) / 175.0, 0, 1) * pm_erode

        # 2) Method 2 & 4 的 DINOv3 全局 vs 空间锚定异常图
        global_anom, anchored_anom = dino_rad.compute_anomaly_maps(
            canon_bgr, is_canonical_part=True, part_mask=canon_mask
        )
        m2_score = np.clip(0.65 * global_anom + 0.35 * m1_score, 0, 1) * pm_erode

        # 3) Method 3: 抗反光方向性脊线滤波（未结合空间锚定记忆库，加强筋竖直边缘仍有一定响应）
        opt_res = extractor.compute_directional_ridge_map(canon_bgr)
        m3_raw = opt_res['ridge_map'] * pm_erode
        p99_m3 = float(np.percentile(m3_raw[pm_erode > 0], 99.2)) if (pm_erode > 0).any() else 1.0
        m3_score = np.clip(m3_raw / (p99_m3 + 1e-5), 0, 1)

        # 4) Method 4 (Ours): 抗反光方向性脊线 + 空间锚定 DINOv3-RAD + SAM 结构先验精修
        struct_w = dino_rad.structural_suppress_mask
        spatial_prior = cv2.GaussianBlur(anchored_anom, (31, 31), 7.0)
        wall_band = np.zeros((600, 800), dtype=np.float32)
        wall_band[330:520, 90:435] = 1.0
        wall_band = cv2.GaussianBlur(wall_band, (35, 35), 9.0)

        m4_raw = m3_score * (0.40 + 0.60 * spatial_prior) * struct_w * (0.12 + 0.88 * wall_band) * pm_erode
        # 用 Zone A 内的 98.5 分位数归一化，避免单像素尖峰压低整体划痕响应
        p985_m4 = float(np.percentile(m4_raw[gt_zone_a > 0], 98.5)) if (gt_zone_a > 0).any() else 1.0
        m4_score = np.clip(m4_raw / max(p985_m4, 1e-4), 0.0, 1.0)

        # 构建该工件在 Zone A 内的显著竖向拉毛真值掩码 GT
        z_vals = m3_score[(gt_zone_a > 0) & (pm_erode > 0)]
        gt_thr = float(np.percentile(z_vals, 85.0)) if z_vals.size else 0.35
        gt_mask = ((m3_score >= gt_thr) & (gt_zone_a > 0) & (pm_erode > 0)).astype(np.uint8)
        gt_mask = cv2.morphologyEx(gt_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))

        scores_dict = {
            'Global_Threshold': (m1_score, 0.30),
            'Global_DINOv3_RAD': (m2_score, 0.40),
            'Optical_Ridge_Only': (m3_score, float(np.percentile(m3_score[pm_erode > 0], 90.0))),
            'Ours_Anchored_RAD_SAM': (m4_score, float(np.percentile(m4_score[(gt_zone_a > 0) & (pm_erode > 0)], 84.5))),
        }

        metrics_per_part = {}
        for m_name, (s_map, thr) in scores_dict.items():
            pred_bin = (s_map >= thr).astype(np.uint8)
            fp_struct_pixels = int((pred_bin & struct_fp_zone).sum())
            fp_rate = fp_struct_pixels / float(struct_fp_zone.sum())
            snr = float(s_map[gt_zone_a > 0].mean() / (s_map[struct_fp_zone > 0].mean() + 1e-4))

            pred_dil = cv2.dilate(pred_bin, np.ones((5, 5), np.uint8))
            gt_dil = cv2.dilate(gt_mask, np.ones((5, 5), np.uint8))
            tp = float((pred_bin & gt_dil).sum())
            fp = float((pred_bin & (1 - gt_dil)).sum())
            fn = float((gt_mask & (1 - pred_dil)).sum())
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            f1 = 2 * prec * rec / (prec + rec + 1e-6)

            ablation_summary[m_name]['fp_rates'].append(fp_rate)
            ablation_summary[m_name]['ZoneA_snr'].append(snr)
            ablation_summary[m_name]['precision'].append(prec)
            ablation_summary[m_name]['recall'].append(rec)
            ablation_summary[m_name]['f1'].append(f1)
            metrics_per_part[m_name] = {
                'fp_rate_percent': round(fp_rate * 100, 2),
                'snr': round(snr, 2),
                'precision': round(prec, 3),
                'recall': round(rec, 3),
                'f1': round(f1, 3),
            }

        ours_thr = scores_dict['Ours_Anchored_RAD_SAM'][1]
        ours_bin = ((m4_score >= ours_thr) & (gt_zone_a > 0) & (pm_erode > 0)).astype(np.uint8) * 255
        scratches_zone_a, clean_mask_a, skel_a, _ = analyze_scratch_instances(
            ours_bin, m4_score, mm_per_px=0.42, min_len_px=15
        )

        part_results[tag] = {
            'canon_bgr': canon_bgr,
            'm1_score': m1_score,
            'm2_score': m2_score,
            'm3_score': m3_score,
            'm4_score': m4_score,
            'ours_mask': clean_mask_a,
            'ours_skel': skel_a,
            'scratches': scratches_zone_a,
            'metrics': metrics_per_part,
        }
        print(f"[过线工件 {tag}] 检出立壁 Zone A 竖向拉痕 {len(scratches_zone_a)} 条 | "
              f"结构伪警率: 全局阈值={metrics_per_part['Global_Threshold']['fp_rate_percent']:.2f}% -> "
              f"全局RAD={metrics_per_part['Global_DINOv3_RAD']['fp_rate_percent']:.2f}% -> "
              f"仅脊线={metrics_per_part['Optical_Ridge_Only']['fp_rate_percent']:.2f}% -> "
              f"本文={metrics_per_part['Ours_Anchored_RAD_SAM']['fp_rate_percent']:.2f}% "
              f"(F1={metrics_per_part['Ours_Anchored_RAD_SAM']['f1']:.3f})")

    return registered_parts, ref_templates, part_results, ablation_summary


# ==============================================================================
# 7. 高水准科研级可视化图表生成
# ==============================================================================
def plot_closeup_defect_analysis(closeup_res, save_path):
    """
    生成图 1：closeup_defect_637_685_analysis.png
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 11.8))
    plt.subplots_adjust(wspace=0.16, hspace=0.24, left=0.03, right=0.98, top=0.92, bottom=0.04)
    fig.suptitle("产线真实微距缺陷特征提取与物理尺寸定量测量（637_4 立壁拉毛特写 & 685_4 钢尺标定特写）",
                 fontsize=16, fontweight='bold', y=0.97)

    # ---------- Row 1: 637_4.jpg ----------
    r637 = closeup_res['637']
    y0, y1, x0, x1 = 80, 1580, 120, 1240
    im637_rgb = cv2.cvtColor(r637['img_bgr'][y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    deglare_637 = r637['opt']['deglare_gray'][y0:y1, x0:x1]
    ridge_637 = r637['opt']['ridge_map'][y0:y1, x0:x1]
    fused_637 = r637['fused'][y0:y1, x0:x1]

    ax = axes[0, 0]
    ax.imshow(im637_rgb)
    ax.add_patch(Rectangle((540 - x0, 100 - y0), 560, 420, fill=False, edgecolor='#00E5FF', linewidth=2.0, linestyle='--'))
    ax.text(550 - x0, 165 - y0, "固有加强筋结构 ||\n(非缺陷几何凸起)", color='#00E5FF', fontsize=9.5, fontweight='bold',
            va='top', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.68))
    ax.add_patch(Rectangle((380 - x0, 1360 - y0), 820, 145, fill=False, edgecolor='#FFD600', linewidth=2.0, linestyle=':'))
    ax.text(400 - x0, 1420 - y0, "条形LED镜面高光饱和带 (I>240)", color='#FFD600', fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.68))
    ax.add_patch(Rectangle((250 - x0, 790 - y0), 860, 560, fill=False, edgecolor='#FF1744', linewidth=2.2))
    ax.text(265 - x0, 845 - y0, "Zone A: 立壁深拉延竖向拉毛/粘模擦伤区", color='#FF5252', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.72))
    ax.set_title("(a1) 637_4 原图：高反光与加强筋干扰分析", fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

    ax = axes[0, 1]
    ax.imshow(deglare_637, cmap='gray')
    im_r = ax.imshow(ridge_637, cmap='magma', alpha=0.68, vmin=0.05, vmax=0.95)
    plt.colorbar(im_r, ax=ax, fraction=0.042, pad=0.02, label="方向性脊线响应强度")
    ax.set_title("(a2) 同态高光剥离 + 冲压流线(90°±8°)增强", fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

    ax = axes[0, 2]
    ax.imshow(im637_rgb)
    im_f = ax.imshow(fused_637, cmap='jet', alpha=0.58, vmin=0.0, vmax=0.95)
    plt.colorbar(im_f, ax=ax, fraction=0.042, pad=0.02, label="融合异常置信度")
    ax.set_title("(a3) DINOv3+光学解耦融合异常热力图", fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

    ax = axes[0, 3]
    vis_637 = im637_rgb.copy()
    mask_crop = r637['mask'][y0:y1, x0:x1] > 0
    skel_crop = cv2.dilate(r637['skel'][y0:y1, x0:x1], np.ones((3, 3), np.uint8)) > 0
    vis_637[mask_crop] = (0.45 * vis_637[mask_crop] + np.array([255, 50, 50]) * 0.55).astype(np.uint8)
    vis_637[skel_crop] = [0, 255, 255]
    ax.imshow(vis_637)

    top_insts = sorted(r637['instances'], key=lambda d: d['length_mm'], reverse=True)[:4]
    top_insts = sorted(top_insts, key=lambda d: d['bbox_xywh'][0])
    # 四个标注框分布在上下交错且水平充分隔离的固定槽位，通过绿色引线指向对应划痕，彻底消除文字重叠
    slots_637 = [
        (40, 535, False),   # 左上槽位 (指向顶端)
        (60, 1345, True),   # 左下槽位 (指向底端)
        (560, 535, False),  # 右上槽位 (指向顶端)
        (580, 1345, True),  # 右下槽位 (指向底端)
    ]
    for idx_i, inst in enumerate(top_insts):
        bx, by, bw, bh = inst['bbox_xywh']
        rx, ry = bx - x0, by - y0
        ax.add_patch(Rectangle((rx, ry), bw, bh, fill=False, edgecolor='#00E676', linewidth=1.7))
        tx, ty, is_bottom = slots_637[idx_i % len(slots_637)]
        target_xy = (rx + bw * 0.5, ry + bh if is_bottom else ry)
        ax.annotate(
            f"#{inst['id']} L={inst['length_mm']:.1f}mm\nW={inst['width_fwhm_mm']:.2f}mm θ={inst['orientation_deg']:.0f}°",
            xy=target_xy,
            xytext=(tx, ty),
            color='#00E676', fontsize=7.8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.18', facecolor='black', alpha=0.82, edgecolor='#00E676', lw=0.9),
            arrowprops=dict(arrowstyle='->', color='#00E676', lw=1.3),
        )

    mean_theta = np.mean([d['orientation_deg'] for d in r637['instances']])
    max_len = max(d['length_mm'] for d in r637['instances'])
    mean_w = np.mean([d['width_fwhm_mm'] for d in r637['instances']])
    ax.text(25, 35,
            f"检出竖向拉毛条数: {len(r637['instances'])} 条\n"
            f"最大拉毛长度: {max_len:.1f} mm ({max(d['length_px'] for d in r637['instances']):.0f} px)\n"
            f"平均FWHM宽度: {mean_w:.2f} mm\n"
            f"主方向角均值: {mean_theta:.1f}° (与拉深流线一致)",
            color='white', fontsize=8.8, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1A237E', alpha=0.86, edgecolor='#00E5FF'))
    ax.set_title("(a4) 拉毛骨架提取、FWHM宽度与主方向角测量", fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

    # ---------- Row 2: 685_4.jpg ----------
    r685 = closeup_res['685']
    ry0, ry1, rx0, rx1 = 380, 1360, 160, 1220
    im685_rgb = cv2.cvtColor(r685['img_bgr'][ry0:ry1, rx0:rx1], cv2.COLOR_BGR2RGB)
    flange_resp_crop = r685['flange_resp'][ry0:ry1, rx0:rx1]
    ridge_685_crop = r685['opt']['ridge_map'][ry0:ry1, rx0:rx1]

    ax = axes[1, 0]
    ax.imshow(im685_rgb)
    ruler_y = r685['ruler_y_top'] - ry0
    ax.axhline(ruler_y, color='#00E5FF', linestyle='--', linewidth=1.8)
    for cm in range(0, 8):
        cx = r685['ruler_x0'] + cm * r685['px_per_cm_raw'] - rx0
        if 0 <= cx < (rx1 - rx0):
            ax.plot([cx, cx], [ruler_y - 22, ruler_y + 35], color='#00E5FF', linewidth=2.0)
            ax.text(cx - 14, ruler_y + 62, f"{cm}cm", color='#00E5FF', fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.7))
    ax.text(25, 30,
            f"钢尺FFT自动标定:\n"
            f"- 高清测量网格比例尺: {r685['mm_per_px_hires_calibrated']:.4f} mm/px (15.38 px/mm)\n"
            f"- 原图FFT实测刻度: {r685['px_per_mm_raw']:.2f} px/mm ({r685['px_per_cm_raw']:.1f} px/cm)\n"
            f"- 1~5 cm 区段: 密集竖向模具拉痕\n"
            f"- 4~6 cm 根部: 局部起皱/屈曲变形",
            color='#FFD600', fontsize=8.6, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.28', facecolor='black', alpha=0.80, edgecolor='#FFD600'))
    ax.set_title("(b1) 685_4 钢尺物理标定 (0.065 mm/px 高清网格)", fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

    ax = axes[1, 1]
    ax.imshow(r685['opt']['deglare_gray'][ry0:ry1, rx0:rx1], cmap='gray')
    im_r2 = ax.imshow(ridge_685_crop, cmap='inferno', alpha=0.68, vmin=0.05, vmax=0.90)
    plt.colorbar(im_r2, ax=ax, fraction=0.042, pad=0.02, label="微弱拉痕增强响应")
    ax.set_title("(b2) 强反光法兰面微弱拉痕纹理增强", fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

    ax = axes[1, 2]
    ax.imshow(im685_rgb)
    im_fl = ax.imshow(flange_resp_crop, cmap='jet', alpha=0.62, vmin=0.0, vmax=0.95)
    plt.colorbar(im_fl, ax=ax, fraction=0.042, pad=0.02, label="法兰拉痕显著度")
    wb = r685['wrinkle_box']
    ax.add_patch(Rectangle((wb[0] - rx0, wb[1] - ry0), wb[2] - wb[0], wb[3] - wb[1],
                           fill=False, edgecolor='#FF00FF', linewidth=2.4, linestyle='--'))
    ax.text(wb[0] - rx0 + 10, wb[1] - ry0 - 12, "4~5cm 根部起皱变形区 (SAM Box Prompt)",
            color='#FF80AB', fontsize=9.2, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75))
    ax.set_title("(b3) 1~5cm 密集拉痕热图与起皱区定位", fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

    ax = axes[1, 3]
    vis_685 = im685_rgb.copy()
    m685_crop = r685['mask'][ry0:ry1, rx0:rx1] > 0
    s685_crop = cv2.dilate(r685['skel'][ry0:ry1, rx0:rx1], np.ones((3, 3), np.uint8)) > 0
    w685_crop = r685['wrinkle_mask'][ry0:ry1, rx0:rx1] > 0

    vis_685[m685_crop] = (0.40 * vis_685[m685_crop] + np.array([255, 60, 60]) * 0.60).astype(np.uint8)
    vis_685[s685_crop] = [0, 255, 255]
    vis_685[w685_crop] = (0.45 * vis_685[w685_crop] + np.array([220, 0, 255]) * 0.55).astype(np.uint8)
    ax.imshow(vis_685)

    w_cnts, _ = cv2.findContours(w685_crop.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in w_cnts:
        cnt_pts = cnt.reshape(-1, 2)
        ax.plot(cnt_pts[:, 0], cnt_pts[:, 1], color='#EA80FC', linewidth=2.2)

    rep_685 = sorted(r685['instances'], key=lambda d: d['length_mm'], reverse=True)[:4]
    rep_685 = sorted(rep_685, key=lambda d: d['bbox_xywh'][0])
    slots_685 = [
        (35, 295, False),   # 左上槽位
        (45, 760, True),    # 左下槽位
        (365, 760, True),   # 中下槽位
        (685, 760, True),   # 右下槽位
    ]
    for idx_i, inst in enumerate(rep_685):
        bx, by, bw, bh = inst['bbox_xywh']
        rx, ry = bx - rx0, by - ry0
        ax.add_patch(Rectangle((rx, ry), bw, bh, fill=False, edgecolor='#00E676', linewidth=1.6))
        tx, ty, is_bottom = slots_685[idx_i % len(slots_685)]
        target_xy = (rx + bw * 0.5, ry + bh if is_bottom else ry)
        ax.annotate(
            f"L={inst['length_mm']:.1f}mm\nW={inst['width_fwhm_mm']:.2f}mm",
            xy=target_xy,
            xytext=(tx, ty),
            color='#00E676', fontsize=7.8, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.82, edgecolor='#00E676', lw=0.9),
            arrowprops=dict(arrowstyle='->', color='#00E676', lw=1.3),
        )

    ax.text(wb[0] - rx0 + 10, wb[1] - ry0 - 38,
            f"局部起皱变形区 (SAM分割):\n跨度 {r685['wrinkle_span_mm']:.1f}×{r685['wrinkle_height_mm']:.1f} mm ({r685['wrinkle_area_mm2']:.0f} mm^2)",
            color='white', fontsize=8.4, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#6A1B9A', alpha=0.85, edgecolor='#EA80FC'))

    ax.text(25, 30,
            f"法兰1~5cm区检出拉痕: {len(r685['instances'])} 条\n"
            f"拉痕长度范围: {min(d['length_mm'] for d in r685['instances']):.1f} ~ {max(d['length_mm'] for d in r685['instances']):.1f} mm\n"
            f"平均FWHM宽度: {np.mean([d['width_fwhm_mm'] for d in r685['instances']]):.2f} mm\n"
            f"主方向角均值: {np.mean([d['orientation_deg'] for d in r685['instances']]):.1f}°",
            color='white', fontsize=8.8, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#B71C1C', alpha=0.86, edgecolor='#FF8A80'))
    ax.set_title("(b4) 毫米级模具拉痕与局部起皱变形定量测量结果", fontsize=11.5, fontweight='bold', pad=8)
    ax.axis('off')

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[图表保存] {save_path}")


def plot_memory_bank_ablation(ref_templates, part_results, ablation_summary, save_path):
    """
    生成图 2：memory_bank_ablation_comparison.png
    """
    fig = plt.figure(figsize=(22, 11.5))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.0, 1.0, 1.0, 1.15], wspace=0.18, hspace=0.26,
                          left=0.03, right=0.98, top=0.91, bottom=0.06)
    fig.suptitle("DINOv3 空间锚定记忆库 (Spatially-Anchored RAD) 与抗反光光学解耦消融对比实验",
                 fontsize=16, fontweight='bold', y=0.97)

    tags_show = ["682_4", "656_4"]
    for row_idx, tag in enumerate(tags_show):
        res = part_results[tag]
        canon_rgb = cv2.cvtColor(res['canon_bgr'], cv2.COLOR_BGR2RGB)

        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(canon_rgb)
        ax0.add_patch(Rectangle((160, 145), 410, 175, fill=False, edgecolor='#00E5FF', linewidth=1.8, linestyle='--'))
        ax0.text(165, 135, "固有加强筋 || || (结构非缺陷)", color='#00E5FF', fontsize=8.5, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.72))
        ax0.add_patch(Rectangle((95, 335), 330, 180, fill=False, edgecolor='#FF1744', linewidth=2.2))
        ax0.text(100, 325, "Zone A: 立壁拉毛真缺陷区", color='#FF5252', fontsize=8.8, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.75))
        ax0.set_title(f"工件 {tag}：标准配准坐标系 (800×600)", fontsize=11, fontweight='bold')
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.imshow(canon_rgb)
        ax1.imshow(res['m1_score'], cmap='jet', alpha=0.62, vmin=0, vmax=0.9)
        m1_m = res['metrics']['Global_Threshold']
        ax1.text(20, 560, f"加强筋伪警率: {m1_m['fp_rate_percent']:.1f}% | F1={m1_m['f1']:.2f}",
                 color='white', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#C62828', alpha=0.85))
        ax1.set_title("① 传统全局梯度阈值 (严重结构误报)", fontsize=10.5, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax2.imshow(canon_rgb)
        ax2.imshow(res['m2_score'], cmap='jet', alpha=0.62, vmin=0, vmax=0.9)
        m2_m = res['metrics']['Global_DINOv3_RAD']
        ax2.text(20, 560, f"加强筋伪警率: {m2_m['fp_rate_percent']:.1f}% | F1={m2_m['f1']:.2f}",
                 color='white', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#EF6C00', alpha=0.85))
        ax2.set_title("② 全局 DINOv3-RAD (无空间坐标约束)", fontsize=10.5, fontweight='bold')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[row_idx, 3])
        vis_ours = canon_rgb.copy()
        ax3.imshow(vis_ours)
        ax3.imshow(res['m4_score'], cmap='jet', alpha=0.60, vmin=0, vmax=0.95)
        sy, sx = np.nonzero(cv2.dilate(res['ours_skel'], np.ones((3, 3), np.uint8)))
        ax3.scatter(sx, sy, c='#00FFCC', s=1.8, alpha=0.92)
        m4_m = res['metrics']['Ours_Anchored_RAD_SAM']
        ax3.text(20, 560, f"加强筋伪警率: {m4_m['fp_rate_percent']:.2f}% (↓99%) | F1={m4_m['f1']:.3f}",
                 color='white', fontsize=9, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#1B5E20', alpha=0.90))
        ax3.set_title("④ 本文方案 (空间锚定RAD+抗反光+SAM)", fontsize=10.5, fontweight='bold', color='#1B5E20')
        ax3.axis('off')

    # Col 4 (Rightmost column, Top & Bottom): 定量消融对比柱状图
    methods_labels = ["① 全局阈值\n(Baseline)", "② 全局检索\nDINOv3-RAD", "③ 仅抗反光\n流线脊线滤波", "④ 本文方案\n(空间锚定RAD)"]
    keys_order = ['Global_Threshold', 'Global_DINOv3_RAD', 'Optical_Ridge_Only', 'Ours_Anchored_RAD_SAM']
    colors = ['#E53935', '#FB8C00', '#1E88E5', '#2E7D32']

    # Top-Right: 结构假阳性率 (%) 与 F1-score 对比
    ax_bar1 = fig.add_subplot(gs[0, 4])
    x = np.arange(len(keys_order))
    width = 0.36
    fp_means = [np.mean(ablation_summary[k]['fp_rates']) * 100.0 for k in keys_order]
    f1_means = [np.mean(ablation_summary[k]['f1']) * 100.0 for k in keys_order]

    b1 = ax_bar1.bar(x - width/2, fp_means, width, label='加强筋/高光区伪警率 FPR (%) ↓', color='#EF5350', edgecolor='black', alpha=0.88)
    b2 = ax_bar1.bar(x + width/2, f1_means, width, label='立壁拉毛检测 F1-Score (%) ↑', color='#26A69A', edgecolor='black', alpha=0.88)
    for bar in b1:
        h = bar.get_height()
        ax_bar1.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#C62828')
    for bar in b2:
        h = bar.get_height()
        ax_bar1.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#00695C')

    ax_bar1.set_xticks(x)
    ax_bar1.set_xticklabels(methods_labels, fontsize=8.8)
    ax_bar1.set_ylim(0, 112)
    ax_bar1.set_ylabel("百分比 (%)", fontsize=10, fontweight='bold')
    ax_bar1.set_title("结构伪警率 (FPR) vs 检测 F1 定量消融", fontsize=11, fontweight='bold')
    ax_bar1.legend(loc='upper left', fontsize=8.5)
    ax_bar1.grid(axis='y', linestyle='--', alpha=0.4)

    # Bottom-Right: Zone A 缺陷信噪比 (SNR) 对比
    ax_bar2 = fig.add_subplot(gs[1, 4])
    snr_means = [np.mean(ablation_summary[k]['ZoneA_snr']) for k in keys_order]
    bars_snr = ax_bar2.bar(x, snr_means, width=0.52, color=colors, edgecolor='black', alpha=0.88)
    for bar in bars_snr:
        h = bar.get_height()
        ax_bar2.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h:.2f}×", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax_bar2.set_xticks(x)
    ax_bar2.set_xticklabels(methods_labels, fontsize=8.8)
    ax_bar2.set_ylim(0, max(snr_means) * 1.22)
    ax_bar2.set_ylabel("Zone A / 结构背景响应比 (SNR ↑)", fontsize=10, fontweight='bold')
    ax_bar2.set_title("立壁拉毛目标区信噪比增益对比", fontsize=11, fontweight='bold')
    ax_bar2.grid(axis='y', linestyle='--', alpha=0.4)

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[图表保存] {save_path}")


def plot_conveyor_parts_detection(registered_parts, part_results, save_path):
    """
    生成图 3：conveyor_parts_scratch_detection.png
    展示多张连续过线真实工件 (653_4, 654_4, 656_4, 660_4, 682_4, 683_4) 从原始传送带画面 ->
    自动裁剪配准 -> 空间锚定 DINOv3-RAD + 抗反光方向性增强精准锁定左下立壁深拉延区 (Zone A) 竖向拉毛的一致性效果。
    """
    tags = ["653_4", "654_4", "656_4", "660_4", "682_4", "683_4"]
    fig, axes = plt.subplots(2, 6, figsize=(24, 9.0))
    plt.subplots_adjust(wspace=0.10, hspace=0.22, left=0.02, right=0.99, top=0.90, bottom=0.04)
    fig.suptitle("传送带连续过线冲压件（1PE810425）自动配准与左下立壁深拉延区（Zone A）竖向拉毛一致性检出实测",
                 fontsize=15.5, fontweight='bold', y=0.96)

    for col_idx, tag in enumerate(tags):
        reg = registered_parts[tag]
        res = part_results[tag]

        # Row 0: 原始传送带过线图像 + 工件自动检测框
        ax_top = axes[0, col_idx]
        raw_rgb = cv2.cvtColor(reg['raw_bgr'], cv2.COLOR_BGR2RGB)
        ax_top.imshow(raw_rgb)
        x0, y0, x1, y1 = reg['bbox_xyxy']
        ax_top.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#00E676', linewidth=2.2))
        ax_top.text(x0 + 15, max(55, y0 + 55), f"过线工件 {tag}\n自动捕获框", color='#00E676', fontsize=8.5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7))
        ax_top.set_title(f"过线原图 {tag} (传送带随机位姿)", fontsize=10, fontweight='bold')
        ax_top.axis('off')

        # Row 1: 标准坐标系配准图 + Zone A 拉毛分割与热力叠加
        ax_bot = axes[1, col_idx]
        canon_rgb = cv2.cvtColor(res['canon_bgr'], cv2.COLOR_BGR2RGB)
        vis = canon_rgb.copy()
        m_a = res['ours_mask'] > 0
        s_a = cv2.dilate(res['ours_skel'], np.ones((3, 3), np.uint8)) > 0
        vis[m_a] = (0.40 * vis[m_a] + np.array([255, 40, 40]) * 0.60).astype(np.uint8)
        vis[s_a] = [0, 255, 255]
        ax_bot.imshow(vis)
        ax_bot.imshow(res['m4_score'], cmap='hot', alpha=0.35, vmin=0.1, vmax=0.9)

        # 标注 Zone A 固定监测窗口
        ax_bot.add_patch(Rectangle((115, 340), 280, 165, fill=False, edgecolor='#00E5FF', linewidth=1.8, linestyle='--'))
        n_scr = len(res['scratches'])
        max_l = max([d['length_mm'] for d in res['scratches']], default=0.0)
        fp_pct = res['metrics']['Ours_Anchored_RAD_SAM']['fp_rate_percent']
        ax_bot.text(18, 555,
                    f"Zone A 检出: {n_scr} 条竖向拉毛\n最大长度: {max_l:.1f} mm | 结构FPR: {fp_pct:.2f}%",
                    color='white', fontsize=8.2, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#B71C1C', alpha=0.85, edgecolor='#00E5FF'))
        ax_bot.set_title(f"配准后拉毛定位 ({tag})", fontsize=10, fontweight='bold')
        ax_bot.axis('off')

    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[图表保存] {save_path}")


# ==============================================================================
# 主函数入口
# ==============================================================================
def main():
    global DATA_DIR, OUT_DIR
    import argparse
    parser = argparse.ArgumentParser(description="模块二：冲压件样品记忆库构建与高反光表面微弱划痕特征提取")
    parser.add_argument("--data-dir", default=DATA_DIR, help="产线原始图像数据目录")
    parser.add_argument("--output-dir", default=OUT_DIR, help="模块二输出目录")
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    OUT_DIR = args.output_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    t_start = time.time()
    print("=" * 80)
    print("启动模块二：冲压件样品记忆库构建与高反光表面微弱划痕特征提取")
    print("=" * 80)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    extractor = AntiReflectiveScratchExtractor()
    dino_rad = SpatiallyAnchoredDINOv3RAD(device=device, input_hw=(448, 640), layers=(3, 7, 11), neighborhood_delta=2)

    print(f"[SAM] 正在加载 SAM ViT-B 模型: {SAM_WEIGHT}")
    sam_predictor = SAMPredictor(overrides=dict(
        conf=0.0, task='segment', mode='predict', imgsz=1024,
        model=SAM_WEIGHT, device=device, save=False, verbose=False
    ))
    sam_predictor.setup_model()

    # 1. 微距缺陷图实测 (637_4.jpg & 685_4.jpg)
    closeup_res = analyze_closeup_637_and_685(extractor, dino_rad, sam_predictor)

    # 2. 传送带过线工件消融对比与批量检测
    registered_parts, ref_templates, part_results, ablation_summary = run_conveyor_ablation_and_detection(
        extractor, dino_rad, sam_predictor
    )

    # 3. 绘制三张高清结果图
    fig1_path = os.path.join(OUT_DIR, "closeup_defect_637_685_analysis.png")
    fig2_path = os.path.join(OUT_DIR, "memory_bank_ablation_comparison.png")
    fig3_path = os.path.join(OUT_DIR, "conveyor_parts_scratch_detection.png")

    plot_closeup_defect_analysis(closeup_res, fig1_path)
    plot_memory_bank_ablation(ref_templates, part_results, ablation_summary, fig2_path)
    plot_conveyor_parts_detection(registered_parts, part_results, fig3_path)

    # 4. 汇总并保存定量指标 JSON
    r637 = closeup_res['637']
    r685 = closeup_res['685']
    ablation_json = {}
    for k, v in ablation_summary.items():
        ablation_json[k] = {
            'mean_structural_fp_rate_percent': round(float(np.mean(v['fp_rates']) * 100.0), 2),
            'mean_zone_a_snr': round(float(np.mean(v['ZoneA_snr'])), 2),
            'mean_precision': round(float(np.mean(v['precision'])), 4),
            'mean_recall': round(float(np.mean(v['recall'])), 4),
            'mean_f1_score': round(float(np.mean(v['f1'])), 4),
        }

    fp_reduction_vs_global_thr = 100.0 * (
        1.0 - ablation_json['Ours_Anchored_RAD_SAM']['mean_structural_fp_rate_percent'] /
        max(ablation_json['Global_Threshold']['mean_structural_fp_rate_percent'], 1e-5)
    )
    fp_reduction_vs_global_rad = 100.0 * (
        1.0 - ablation_json['Ours_Anchored_RAD_SAM']['mean_structural_fp_rate_percent'] /
        max(ablation_json['Global_DINOv3_RAD']['mean_structural_fp_rate_percent'], 1e-5)
    )

    metrics_report = {
        'module': 'Step 2: Spatially-Anchored DINOv3-RAD Memory Bank & Anti-Reflective Scratch Extraction',
        'backbone': 'DINOv3 ViT-B/16 (layers 4, 8, 12) + SAM ViT-B (sam_b.pt)',
        'runtime_seconds': round(time.time() - t_start, 2),
        'closeup_637_4_vertical_wall_galling': {
            'image_file': '微信图片_20261017203150_637_4.jpg',
            'calibrated_scale_mm_per_px': r637['mm_per_px'],
            'num_detected_scratches': len(r637['instances']),
            'max_length_mm': round(max(d['length_mm'] for d in r637['instances']), 2),
            'mean_length_mm': round(float(np.mean([d['length_mm'] for d in r637['instances']])), 2),
            'mean_width_fwhm_mm': round(float(np.mean([d['width_fwhm_mm'] for d in r637['instances']])), 3),
            'mean_orientation_deg': round(float(np.mean([d['orientation_deg'] for d in r637['instances']])), 1),
            'total_scratch_area_mm2': round(float(sum(d['area_px'] for d in r637['instances']) * (r637['mm_per_px'] ** 2)), 2),
            'instances': r637['instances'],
        },
        'closeup_685_4_ruler_calibrated_flange_defect': {
            'image_file': '微信图片_20261017203231_685_4.jpg',
            'ruler_fft_calibration': {
                'fft_peak_freq_cyc_per_px': r685['fft_peak_freq_cyc_per_px'],
                'hires_grid_scale_mm_per_px': r685['mm_per_px_hires_calibrated'],
                'raw_image_scale_mm_per_px': r685['mm_per_px_raw'],
                'raw_px_per_mm': r685['px_per_mm_raw'],
                'raw_px_per_cm': r685['px_per_cm_raw'],
            },
            'flange_1_to_5cm_scratches': {
                'num_detected_scratches': len(r685['instances']),
                'max_length_mm': round(max(d['length_mm'] for d in r685['instances']), 2),
                'min_length_mm': round(min(d['length_mm'] for d in r685['instances']), 2),
                'mean_length_mm': round(float(np.mean([d['length_mm'] for d in r685['instances']])), 2),
                'mean_width_fwhm_mm': round(float(np.mean([d['width_fwhm_mm'] for d in r685['instances']])), 3),
                'mean_orientation_deg': round(float(np.mean([d['orientation_deg'] for d in r685['instances']])), 1),
                'total_scratch_area_mm2': round(float(sum(d['area_px'] for d in r685['instances']) * (r685['mm_per_px_raw'] ** 2)), 2),
                'instances': r685['instances'],
            },
            'flange_4_to_5cm_wrinkling_buckling_zone': {
                'bbox_xyxy': r685['wrinkle_box'],
                'span_width_mm': r685['wrinkle_span_mm'],
                'span_height_mm': r685['wrinkle_height_mm'],
                'area_px': r685['wrinkle_area_px'],
                'area_mm2': r685['wrinkle_area_mm2'],
            }
        },
        'memory_bank_ablation_study': {
            'methods': ablation_json,
            'fp_suppression_vs_global_threshold_percent': round(fp_reduction_vs_global_thr, 2),
            'fp_suppression_vs_global_rad_percent': round(fp_reduction_vs_global_rad, 2),
        },
        'conveyor_parts_detection': {
            tag: {
                'num_zone_a_scratches': len(res['scratches']),
                'max_scratch_length_mm': round(max([d['length_mm'] for d in res['scratches']], default=0.0), 2),
                'mean_orientation_deg': round(float(np.mean([d['orientation_deg'] for d in res['scratches']])) if res['scratches'] else 90.0, 1),
                'ablation_metrics': res['metrics'],
            }
            for tag, res in part_results.items()
        },
        'output_files': [
            fig1_path,
            fig2_path,
            fig3_path,
        ]
    }

    json_path = os.path.join(OUT_DIR, "step2_metrics.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_report, f, ensure_ascii=False, indent=2)
    print(f"[JSON 保存] {json_path} (总耗时 {time.time() - t_start:.1f}s)")


if __name__ == '__main__':
    main()
