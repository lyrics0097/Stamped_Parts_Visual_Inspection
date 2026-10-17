"""生成冲压件检测演示用 3D 模型 (STL / PLY) 以及配套资源。"""
from __future__ import annotations

import os
import struct
from pathlib import Path

import numpy as np

OUT_DIR = Path(os.environ["USERPROFILE"]) / "Desktop" / "冲压件检测3D模型"
MODEL_DIR = OUT_DIR / "models"


def write_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """vertices: (N,3) float32, faces: (M,3) int."""
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, np.maximum(lens, 1e-12))
    header = b"Stamping inspection demo mesh".ljust(80, b"\0")
    buf = [header, struct.pack("<I", int(len(f)))]
    rec = np.zeros(len(f), dtype=np.dtype([
        ("n", "<f4", 3),
        ("v0", "<f4", 3),
        ("v1", "<f4", 3),
        ("v2", "<f4", 3),
        ("attr", "<u2"),
    ]))
    rec["n"] = n
    rec["v0"] = a
    rec["v1"] = b
    rec["v2"] = c
    path.write_bytes(header + struct.pack("<I", int(len(f))) + rec.tobytes())


def write_ply_color(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    c = np.clip(np.asarray(colors, dtype=np.float32), 0, 1)
    rgb = (c * 255).astype(np.uint8)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(v)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(f)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for p, col in zip(v, rgb):
        lines.append(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(col[0])} {int(col[1])} {int(col[2])}")
    for tri in f:
        lines.append(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def heatmap_color(t: np.ndarray) -> np.ndarray:
    """0=steel grey, 1=hot red-orange (scratch depth)."""
    t = np.clip(t, 0, 1)
    steel = np.array([0.62, 0.66, 0.70])
    orange = np.array([0.95, 0.35, 0.08])
    red = np.array([0.85, 0.05, 0.05])
    mid = np.clip(t * 2.0, 0, 1)[:, None]
    hi = np.clip(t * 2.0 - 1.0, 0, 1)[:, None]
    col = steel * (1 - mid) + orange * mid
    col = col * (1 - hi) + red * hi
    return col


def box(xmin, xmax, ymin, ymax, zmin, zmax):
    p = np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
        [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
    ], dtype=np.float32)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 4, 7], [0, 7, 3],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return p, faces


def cylinder(radius, z0, z1, cx=0, cy=0, segs=28, caps=True):
    th = np.linspace(0, 2 * np.pi, segs, endpoint=False)
    ring0 = np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th), np.full(segs, z0)], 1)
    ring1 = np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th), np.full(segs, z1)], 1)
    verts = [ring0, ring1]
    if caps:
        verts.append(np.array([[cx, cy, z0]], dtype=np.float32))
        verts.append(np.array([[cx, cy, z1]], dtype=np.float32))
    v = np.vstack(verts).astype(np.float32)
    faces = []
    for i in range(segs):
        j = (i + 1) % segs
        faces.append([i, j, segs + j])
        faces.append([i, segs + j, segs + i])
        if caps:
            bot, top = 2 * segs, 2 * segs + 1
            faces.append([bot, j, i])
            faces.append([top, segs + i, segs + j])
    return v, np.array(faces, dtype=np.int32)


def merge(parts):
    vs, fs = [], []
    off = 0
    for v, f in parts:
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    return np.vstack(vs), np.vstack(fs)


def transform(v, dx=0, dy=0, dz=0, sx=1, sy=1, sz=1):
    out = v.copy()
    out[:, 0] = out[:, 0] * sx + dx
    out[:, 1] = out[:, 1] * sy + dy
    out[:, 2] = out[:, 2] * sz + dz
    return out


def part_height(x, y):
    """冲压件上表面高度 (mm)。含加强筋、沉台、划痕沟槽。"""
    z = np.full_like(x, 3.2, dtype=np.float32)

    # 外缘下翻 / 翻边
    edge = np.maximum(np.abs(x) / 98.0, np.abs(y) / 68.0)
    flange = np.clip((edge - 0.86) / 0.14, 0, 1)
    z -= flange * 1.6

    # 中心沉台
    rec = (np.abs(x) < 52) & (np.abs(y) < 28)
    z = np.where(rec, z - 1.1, z)

    # 四周加强筋
    rib_x = (np.abs(np.abs(x) - 70) < 3.2) & (np.abs(y) < 50)
    rib_y = (np.abs(np.abs(y) - 42) < 3.2) & (np.abs(x) < 80)
    z = np.where(rib_x | rib_y, z + 1.4, z)

    # 安装凸台
    for cx, cy in ((-78, -50), (78, -50), (-78, 50), (78, 50), (0, 0)):
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        boss = np.clip(1.0 - r / 9.0, 0, 1)
        z += 0.9 * boss ** 1.4

    # 轻微整体拱起
    z += 0.18 * np.cos(x / 90.0) * np.cos(y / 70.0)

    # 划痕: 沿折线的高斯沟槽
    # 路径点 (分段折线), 模拟模具刃口拉伤留下的细长划痕
    path = np.array([
        [-62.0, -18.0],
        [-28.0, -6.0],
        [8.0, 4.0],
        [38.0, 14.0],
        [58.0, 22.0],
    ], dtype=np.float32)
    pts = np.stack([x.ravel(), y.ravel()], 1)
    dmin = np.full(pts.shape[0], 1e9, dtype=np.float32)
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        ab = b - a
        t = np.clip(((pts - a) @ ab) / (ab @ ab + 1e-9), 0, 1)
        proj = a + t[:, None] * ab
        dmin = np.minimum(dmin, np.linalg.norm(pts - proj, axis=1))
    dmin = dmin.reshape(x.shape)
    width = 2.15  # 半宽约 1.07, 全宽约 2.15 mm
    depth = 0.82
    groove = depth * np.exp(-0.5 * (dmin / (width * 0.28)) ** 2)
    groove = np.where(dmin < width * 1.4, groove, 0)
    z -= groove
    return z.astype(np.float32), groove.astype(np.float32)


def hole_mask(x, y):
    holes = [(-78, -50, 4.2), (78, -50, 4.2), (-78, 50, 4.2), (78, 50, 4.2),
             (-30, 0, 6.5), (30, 0, 6.5)]
    m = np.zeros_like(x, dtype=bool)
    for cx, cy, r in holes:
        m |= (x - cx) ** 2 + (y - cy) ** 2 < r * r
    return m


def build_stamped_part(nx=180, ny=126):
    xs = np.linspace(-100, 100, nx, dtype=np.float32)
    ys = np.linspace(-70, 70, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    Z, groove = part_height(X, Y)
    holes = hole_mask(X, Y)
    thick = 2.4

    top = np.stack([X, Y, Z], -1)
    bot = np.stack([X, Y, Z - thick], -1)
    vtop = top.reshape(-1, 3)
    vbot = bot.reshape(-1, 3)
    verts = np.vstack([vtop, vbot])
    n = nx * ny

    def vid(i, j, layer=0):
        return layer * n + i * nx + j

    faces = []
    # 上/下表面 (避开孔)
    for i in range(ny - 1):
        for j in range(nx - 1):
            if holes[i, j] or holes[i, j + 1] or holes[i + 1, j] or holes[i + 1, j + 1]:
                continue
            a, b, c, d = vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i + 1, j)
            faces.append([a, b, c])
            faces.append([a, c, d])
            a, b, c, d = vid(i, j, 1), vid(i + 1, j, 1), vid(i + 1, j + 1, 1), vid(i, j + 1, 1)
            faces.append([a, b, c])
            faces.append([a, c, d])

    # 外轮廓侧壁
    for j in range(nx - 1):
        for i, i2 in ((0, 0), (ny - 1, ny - 1)):
            a, b = vid(i, j), vid(i, j + 1)
            faces.append([a, b, b + n] if i == 0 else [a, a + n, b + n])
            faces.append([a, b + n, a + n] if i == 0 else [a, b + n, b])
    for i in range(ny - 1):
        for j in (0, nx - 1):
            a, b = vid(i, j), vid(i + 1, j)
            if j == 0:
                faces.append([a, a + n, b + n])
                faces.append([a, b + n, b])
            else:
                faces.append([a, b, b + n])
                faces.append([a, b + n, a + n])

    # 孔壁
    holes_list = [(-78, -50, 4.2), (78, -50, 4.2), (-78, 50, 4.2), (78, 50, 4.2),
                  (-30, 0, 6.5), (30, 0, 6.5)]
    extra_v = []
    extra_f = []
    base = len(verts)
    segs = 24
    for cx, cy, r in holes_list:
        th = np.linspace(0, 2 * np.pi, segs, endpoint=False)
        zx = float(part_height(np.array([[cx]]), np.array([[cy]]))[0][0, 0])
        ring_t = np.stack([cx + r * np.cos(th), cy + r * np.sin(th), np.full(segs, zx)], 1)
        ring_b = ring_t.copy()
        ring_b[:, 2] = zx - thick
        extra_v.append(ring_t)
        extra_v.append(ring_b)
        off = base
        base += 2 * segs
        for k in range(segs):
            k2 = (k + 1) % segs
            extra_f.append([off + k, off + k2, off + segs + k2])
            extra_f.append([off + k, off + segs + k2, off + segs + k])

    if extra_v:
        verts = np.vstack([verts, np.vstack(extra_v)])
        faces.extend(extra_f)

    faces = np.array(faces, dtype=np.int32)
    t = np.clip(groove.ravel() / 0.82, 0, 1)
    colors = np.vstack([heatmap_color(t), heatmap_color(t)])
    if extra_v:
        extra_n = verts.shape[0] - colors.shape[0]
        steel = np.tile(np.array([[0.62, 0.66, 0.70]]), (extra_n, 1))
        colors = np.vstack([colors, steel])
    return verts, faces, colors


def build_camera():
    parts = [
        box(-28, 28, -18, 18, 0, 22),                 # 机身
        box(-10, 10, -10, 10, 22, 30),                # 颈部
        box(-8, 8, -8, 8, 30, 55),                    # 立柱
        box(-40, 40, -6, 6, 52, 62),                  # 横梁座
    ]
    # 双目镜头 + 中间投影器
    for cx, r, h0, h1 in ((-16, 6.5, -8, 0), (16, 6.5, -8, 0), (0, 4.2, -6, 0)):
        parts.append(cylinder(r, h0, h1, cx=cx, cy=0, segs=20))
    # 镜头玻璃略凸
    parts.append(cylinder(4.8, -9.2, -8.0, cx=-16, segs=16))
    parts.append(cylinder(4.8, -9.2, -8.0, cx=16, segs=16))
    parts.append(cylinder(2.6, -7.5, -6.0, cx=0, segs=14))
    return merge(parts)


def build_gantry():
    parts = [
        box(-110, -98, -80, 80, 0, 8),
        box(98, 110, -80, 80, 0, 8),
        box(-110, -102, -8, 8, 8, 210),
        box(102, 110, -8, 8, 8, 210),
        box(-110, 110, -8, 8, 200, 214),
        box(-16, 16, -16, 16, 188, 200),
    ]
    return merge(parts)


def build_die():
    """上下模, 上模刃口带裂纹。"""
    parts = []
    # 下模座
    parts.append(box(-90, 90, -60, 60, 0, 18))
    parts.append(box(-70, 70, -40, 40, 18, 28))
    # 导柱
    for cx, cy in ((-78, -48), (78, -48), (-78, 48), (78, 48)):
        parts.append(cylinder(6.0, 18, 78, cx=cx, cy=cy, segs=16))
    # 上模
    parts.append(box(-90, 90, -60, 60, 78, 98))
    parts.append(box(-70, 70, -40, 40, 68, 78))
    # 刃口裂纹: 一串细小楔块沿刃口
    crack_x = np.linspace(-8, 22, 10)
    for i, x in enumerate(crack_x):
        w = 0.7 + 0.15 * (i % 3)
        parts.append(box(x, x + 2.4, 39.2, 40.8, 68.0, 69.6 + 0.2 * i))
        parts.append(box(x + 0.4, x + 1.6, 39.6, 40.5, 67.2, 68.4))
    return merge(parts)


def build_table():
    parts = [
        box(-160, 160, -110, 110, -12, 0),
        box(-150, -138, -100, -88, -82, -12),
        box(138, 150, -100, -88, -82, -12),
        box(-150, -138, 88, 100, -82, -12),
        box(138, 150, 88, 100, -82, -12),
    ]
    return merge(parts)


def build_scene():
    part_v, part_f, _ = build_stamped_part()
    cam_v, cam_f = build_camera()
    gan_v, gan_f = build_gantry()
    die_v, die_f = build_die()
    tab_v, tab_f = build_table()

    cam_v = transform(cam_v, dz=168)
    die_v = transform(die_v, dx=250, dy=0, dz=0)
    return merge([
        (tab_v, tab_f),
        (part_v, part_f),
        (gan_v, gan_f),
        (cam_v, cam_f),
        (die_v, die_f),
    ])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("building stamped part...")
    pv, pf, pc = build_stamped_part()
    write_stl(MODEL_DIR / "stamped_part_with_scratch.stl", pv, pf)
    write_ply_color(MODEL_DIR / "stamped_part_scratch_heatmap.ply", pv, pf, pc)

    print("building camera / gantry / die / table...")
    cv, cf = build_camera()
    write_stl(MODEL_DIR / "3d_camera.stl", cv, cf)
    gv, gf = build_gantry()
    write_stl(MODEL_DIR / "gantry.stl", gv, gf)
    dv, df = build_die()
    write_stl(MODEL_DIR / "stamping_die_cracked.stl", dv, df)
    tv, tf = build_table()
    write_stl(MODEL_DIR / "worktable.stl", tv, tf)

    print("building full station...")
    sv, sf = build_scene()
    write_stl(MODEL_DIR / "inspection_station.stl", sv, sf)

    readme = OUT_DIR / "使用说明.txt"
    readme.write_text(
        "冲压件 3D 划痕检测演示模型\n"
        "================================\n\n"
        "1) 双击「打开演示.html」—— 浏览器里旋转、缩放、查看测量与模具报警。\n"
        "2) models 文件夹里是可导入 CAD / 3D 查看器的模型：\n"
        "   - stamped_part_with_scratch.stl     带划痕的冲压件\n"
        "   - stamped_part_scratch_heatmap.ply  同零件，划痕用红橙色标深度\n"
        "   - 3d_camera.stl                     双目/结构光 3D 相机\n"
        "   - gantry.stl                        龙门架\n"
        "   - stamping_die_cracked.stl          带刃口裂纹的上下模\n"
        "   - worktable.stl                     检测台\n"
        "   - inspection_station.stl            整机装配场景\n\n"
        "Windows 可用「3D 查看器」或 Blender 打开 STL / PLY。\n"
        "单位: 毫米。划痕设计值 宽 2.15 mm / 深 0.82 mm。\n",
        encoding="utf-8",
    )
    print("done ->", OUT_DIR)


if __name__ == "__main__":
    main()
