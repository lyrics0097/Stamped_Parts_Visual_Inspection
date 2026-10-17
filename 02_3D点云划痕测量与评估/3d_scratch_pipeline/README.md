# 冲压件 3D 划痕宽/深检测（可行方案）

不需要 CAD，也不需要和良品对齐。输入工业 3D 相机（结构光 / 线激光）点云，单位毫米，输出每条划痕的长度、宽度、深度，并按阈值判定异常。

D455 不能测划痕：0.5 m 处大约 1 像素 0.8 mm，深度噪声是毫米级，而冲压划痕通常宽 0.1–0.5 mm、深几十微米。本方案假定相机换成工业 3D 相机。

## 算法（`scratch_3d.py`）

1. 用整体平面把点云摊成高度图，再减掉二次曲面，去掉零件本身的弯曲。
2. 多尺度开运算和闭运算同时抓凸起和凹痕。核宽取 6 / 18 / 30 mm，窄划痕和更宽的压痕都能留下。网格按点距对齐，避免空洞把形态学核掏空。
3. 用 MAD 稳健阈值得到候选，连通域分成每条缺陷，丢掉边缘和噪声碎块。曲面起伏很大时按局部小块再测一次。
4. 沿划痕做横向深度剖面：
   - **深度** = 剖面峰值，再减去沟槽两侧肩部的残差基线（闭运算会把噪声抬进参考面）
   - **宽度** = 半高全宽
   - **长度** = 长轴 1%–99% 跨度
5. 宽 ≥ 0.30 mm 或深 ≥ 0.05 mm 判为异常（阈值可改）。

## 验证数据

公开网上没有「汽车冲压件 + 结构光 + 划痕宽深真值」的完整集。已下载并用于验证的是：

| 数据 | 作用 |
|---|---|
| 本目录 `synthetic_data/` | 弯曲金属板 + 已知宽/深/长的高斯沟槽，定量验证测量公式 |
| IMT Mines Albi [doi:10.57745/C4MSGW](https://doi.org/10.57745/C4MSGW) | 34 个真实机械件 3D 扫描，点级缺陷标注，29 条参考深度。其中 `plaque_g2Metric` 金属平板最接近冲压件局部表面 |
| OSPD / NEU-DET | 只有 2D，不能验证深度 |
| MVTec 3D-AD | 工业 3D，但不是冲压划痕，也没有尺寸真值 |
| 汽车冲压结构光论文 P-NRPCA | 方法贴近，数据未公开 |

原始压缩包在 `..\datasets\defect_3d.zip`，已解压到 `..\datasets\imt_albi\`。检索结论见 `公开数据集说明.txt`。

## 怎么跑

```text
conda activate rad_env
cd 桌面\冲压件检测\3d_scratch_pipeline
python run_benchmark.py
python inspect_cloud.py --input 你的点云.npy --out_dir 结果
```

自己的点云（N×3，单位 mm，也支持 npy / npz / txt / ply）：

```python
from scratch_3d import detect_scratches, to_json
scratches, pred, thr = detect_scratches(xyz_mm)
print(to_json(scratches))
```

换正式相机后，把 `floor_mm` 调到略高于该相机的深度噪声。深拉成形的整件如果缺陷和零件形状缠在一起，先切出局部表面再测。

## 这次验证的结果

判定为可行。数字在 `benchmark_out/summary.json`。

- 合成划痕（8 块弯曲板，宽 0.40–1.50 mm，深 0.06–0.45 mm）：全部检出。宽度相对误差约 15%，深度约 6%，长度约 2%，与真值区域的重合度约 0.83。
- 真实金属平板 `plaque_g2Metric`（9 块，参考深度 0.20–0.66 mm）：全部检出，深度相对误差约 15%，重合度约 0.74。
- 全部 34 个 IMT 扫描的检出率约 94%。深拉钣金 `tole_metallique` 里有几块缺陷和成形面混在一起，深度会对不准，这是当前算法的边界。

## 输出

- `benchmark_out/summary.json` 可行性结论
- `benchmark_out/benchmark.csv` 逐条对照
- `benchmark_out/synthetic_vis/`、`imt_vis/` 俯视对比图
