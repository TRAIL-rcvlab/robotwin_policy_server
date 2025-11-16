#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pkl_dir_to_hdf5_batch.py

批量将目录中的 pkl（list[dict]）转换为 hdf5（同名输出，后缀 .hdf5）。
字段映射（若缺失则不创建）：
- base_rgb:         -> /observation/front_camera/rgb        (N,)   |Smaxlen（JPEG字节）
- base_depth:       -> /observation/front_camera/depth_raw  (N,H,W,1) float32
- joint_positions:  -> /joint_action/positions              (N,7)  float32   # 若>7维，丢弃最后一维
- joint_velocities: -> /joint_action/velocities             (N,Jv) float32
- ee_pos_quat:      -> /endpose/ee_pos_quat                 (N,7)  float64
- gripper_position: -> /endpose/gripper & /joint_action/gripper  (N,)或(N,K) float32
- pointcloud:       -> /pointcloud/xyzrgb                   (N,1024,6) float32  [x,y,z,r,g,b] (r,g,b∈[0,1])
                      （可选保留 legacy: /gripper/position）

用法示例：
python pkl_dir_to_hdf5_batch.py --pkl_dir ./pkls --out_dir ./h5s --jpeg_quality 90 --compression gzip \
    --fx 621.3975 --fy 620.6494 --cx 649.6445 --cy 367.9081 --depth_scale 1.0 --k 1024
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import pickle
import numpy as np
import h5py
import cv2
import sys
import traceback

# ------------- helpers -------------
def to_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        if np.isfinite(a).any() and a.max() <= 1.0:
            a *= 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr

def encode_rgb_to_jpeg_bytes(rgb_uint8: np.ndarray, quality: int = 95) -> bytes:
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return bytes(buf)

def to_float32_depth(arr: np.ndarray) -> np.ndarray:
    d = arr
    if d.ndim == 2:
        d = d[..., None]
    if d.ndim == 3 and d.shape[-1] > 1:
        d = d[..., :1]
    if not (d.ndim == 3 and d.shape[-1] == 1):
        raise ValueError(f"Unexpected depth shape: {arr.shape}")
    return d.astype(np.float32)

def parse_ee_pos_quat(ee_obj) -> Optional[np.ndarray]:
    if ee_obj is None:
        return None
    if isinstance(ee_obj, dict):
        pos = ee_obj.get("pos") or ee_obj.get("position")
        quat = ee_obj.get("quat") or ee_obj.get("quaternion")
        if pos is None or quat is None:
            return None
        pos = np.asarray(pos).reshape(-1)
        quat = np.asarray(quat).reshape(-1)
        if pos.size == 3 and quat.size == 4:
            return np.concatenate([pos, quat]).astype(np.float64)
    if isinstance(ee_obj, (tuple, list)) and len(ee_obj) == 2:
        pos = np.asarray(ee_obj[0]).reshape(-1)
        quat = np.asarray(ee_obj[1]).reshape(-1)
        if pos.size == 3 and quat.size == 4:
            return np.concatenate([pos, quat]).astype(np.float64)
    arr = np.asarray(ee_obj).reshape(-1)
    if arr.size == 7:
        return arr.astype(np.float64)
    return None

def first_present_key(data_list, key: str):
    for i, d in enumerate(data_list):
        if isinstance(d, dict) and key in d and d[key] is not None:
            return i, d[key]
    return None, None

# ---- pointcloud helpers ----
def project_depth_to_points(depth_m: np.ndarray, fx: float, fy: float, cx: float, cy: float):
    """depth_m: (H,W) meters -> (H*W,3) xyz float32 & valid mask"""
    H, W = depth_m.shape
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    z = depth_m.astype(np.float32)
    valid = np.isfinite(z) & (z > 0.0)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = valid.reshape(-1)
    return xyz, valid

def downsample_to_k(arr: np.ndarray, k: int, rng: np.random.Generator):
    """arr: (N,D) -> (k,D); 若 N<k 则允许重复采样补齐"""
    N = arr.shape[0]
    if N == 0:
        return np.zeros((k, arr.shape[1]), dtype=arr.dtype)
    if N >= k:
        idx = rng.choice(N, size=k, replace=False)
    else:
        extra = rng.choice(N, size=k - N, replace=True)
        idx = np.concatenate([np.arange(N), extra])
        rng.shuffle(idx)
    return arr[idx]

# ------------- core conversion -------------
def convert_one(pkl_path: Path, out_path: Path, compression: str, jpeg_quality: int, keep_legacy_gripper: bool,
                intrinsics: Optional[Tuple[float,float,float,float]],
                depth_scale: float, k_points: int, pc_min_depth: float, pc_max_depth: float, rng_seed: int):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{pkl_path.name}: 期望 pkl 顶层为非空 list。")

    N = len(data)
    comp = None if compression == "none" else compression

    has, shapes = {}, {}

    # base_rgb → bytes
    idx, sample = first_present_key(data, "base_rgb")
    if idx is not None:
        has["base_rgb"] = True

    # base_depth
    idx, sample = first_present_key(data, "base_depth")
    if idx is not None:
        d0 = to_float32_depth(np.asarray(sample))
        Hd, Wd, _ = d0.shape
        shapes["base_depth"] = (N, Hd, Wd, 1)
        has["base_depth"] = True

    # joint_positions (drop last dim if >7)
    idx, sample = first_present_key(data, "joint_positions")
    if idx is not None:
        J = int(np.asarray(sample).reshape(-1).shape[0])
        if J > 7:
            J = 7
        shapes["joint_positions"] = (N, J)
        has["joint_positions"] = True

    # joint_velocities
    idx, sample = first_present_key(data, "joint_velocities")
    if idx is not None:
        Jv = int(np.asarray(sample).reshape(-1).shape[0])
        shapes["joint_velocities"] = (N, Jv)
        has["joint_velocities"] = True

    # ee_pos_quat
    idx, sample = first_present_key(data, "ee_pos_quat")
    if idx is not None:
        ee = parse_ee_pos_quat(sample)
        if ee is not None and ee.shape == (7,):
            shapes["ee_pos_quat"] = (N, 7)
            has["ee_pos_quat"] = True

    # gripper_position → duplicate
    idx, sample = first_present_key(data, "gripper_position")
    if idx is not None:
        g = np.asarray(sample).reshape(-1)
        shapes["gripper_position"] = (N,) if g.size == 1 else (N, g.size)
        has["gripper_position"] = True

    # pointcloud 需要 base_rgb + base_depth + intrinsics
    enable_pointcloud = has.get("base_rgb") and has.get("base_depth") and (intrinsics is not None)

    if not any(has.values()):
        raise ValueError(f"{pkl_path.name}: 未发现任何可转换字段。")

    # pre-encode rgb bytes per file (for dtype=|Smaxlen)
    rgb_bytes_list, max_len = None, 0
    if has.get("base_rgb"):
        rgb_bytes_list = []
        for i, d in enumerate(data):
            if ("base_rgb" not in d) or (d["base_rgb"] is None):
                raise KeyError(f"{pkl_path.name}: 第 {i} 帧缺少 base_rgb")
            rgb = to_uint8_rgb(np.asarray(d["base_rgb"]))
            buf = encode_rgb_to_jpeg_bytes(rgb, quality=jpeg_quality)
            rgb_bytes_list.append(buf)
            if len(buf) > max_len:
                max_len = len(buf)

    if out_path.exists():
        out_path.unlink()

    rng = np.random.default_rng(rng_seed)

    with h5py.File(out_path, "w") as h5:
        ds_depth = ds_jpos = ds_jvel = ds_ee = None
        ds_grip_endpose = ds_grip_joint = ds_grip_legacy = None
        ds_xyzrgb = None  # (N,K,6) float32

        # RGB bytes → /observation/front_camera/rgb
        if rgb_bytes_list is not None:
            grp_obs = h5.require_group("observation")
            grp_cam = grp_obs.require_group("front_camera")
            dt = h5py.string_dtype(encoding="ascii", length=max_len)  # will show as |S{max_len}
            ds_rgb = grp_cam.create_dataset("rgb", shape=(N,), dtype=dt)
            ds_rgb.attrs["encoding"] = "jpeg"
            ds_rgb.attrs["note"] = "Fixed-length byte strings; JPEG-encoded RGB per frame."
            ds_rgb.attrs["source"] = str(pkl_path)
            for i in range(N):
                ds_rgb[i] = rgb_bytes_list[i]

        # depth_raw
        if has.get("base_depth"):
            grp_obs = h5.require_group("observation")
            grp_fcam = grp_obs.require_group("front_camera")
            ds_depth = grp_fcam.create_dataset(
                "depth_raw", shape=shapes["base_depth"], dtype=np.float32,
                chunks=(1, shapes["base_depth"][1], shapes["base_depth"][2], 1),
                compression=comp
            )
            ds_depth.attrs["desc"] = "Raw depth frames (float32)"
            ds_depth.attrs["unit"] = "meters_after_scale"  # 写入时会乘 depth_scale

        # joints (group: joint_action)
        if has.get("joint_positions"):
            grp_joint = h5.require_group("joint_action")
            ds_jpos = grp_joint.create_dataset(
                "positions", shape=shapes["joint_positions"], dtype=np.float32,
                chunks=(1, shapes["joint_positions"][1]), compression=comp
            )
        if has.get("joint_velocities"):
            grp_joint = h5.require_group("joint_action")
            ds_jvel = grp_joint.create_dataset(
                "velocities", shape=shapes["joint_velocities"], dtype=np.float32,
                chunks=(1, shapes["joint_velocities"][1]), compression=comp
            )

        # endpose
        if has.get("ee_pos_quat"):
            grp_end = h5.require_group("endpose")
            ds_ee = grp_end.create_dataset(
                "ee_pos_quat", shape=shapes["ee_pos_quat"], dtype=np.float64,
                chunks=(1, 7), compression=comp
            )
            ds_ee.attrs["desc"] = "[x,y,z,qx,qy,qz,qw]"

        # gripper duplicated: endpose/gripper & joint_action/gripper
        if has.get("gripper_position"):
            shape = shapes["gripper_position"]
            chunks = (1,) if len(shape) == 1 else (1, shape[1])
            grp_end = h5.require_group("endpose")
            ds_grip_endpose = grp_end.create_dataset(
                "gripper", shape=shape, dtype=np.float32, chunks=chunks, compression=comp
            )
            grp_joint = h5.require_group("joint_action")
            ds_grip_joint = grp_joint.create_dataset(
                "gripper", shape=shape, dtype=np.float32, chunks=chunks, compression=comp
            )
            if keep_legacy_gripper:
                grp_legacy = h5.require_group("gripper")
                ds_grip_legacy = grp_legacy.create_dataset(
                    "position", shape=shape, dtype=np.float32, chunks=chunks, compression=comp
                )

        # pointcloud/xyzrgb
        if enable_pointcloud:
            ds_xyzrgb = h5.create_dataset(
                "pointcloud", shape=(N, k_points, 6), dtype=np.float32,
                chunks=(1, k_points, 6), compression="gzip"
            )
            fx, fy, cx, cy = intrinsics
            ds_xyzrgb.attrs["intrinsics"]  = np.array([fx, fy, cx, cy], dtype=np.float32)
            ds_xyzrgb.attrs["depth_scale"] = np.float32(depth_scale)
            ds_xyzrgb.attrs["K"]           = np.int32(k_points)
            ds_xyzrgb.attrs["color_scale"] = "rgb_in_[0,1]_float32"
            ds_xyzrgb.attrs["min_depth"]   = np.float32(pc_min_depth)
            ds_xyzrgb.attrs["max_depth"]   = np.float32(pc_max_depth)

        # meta
        h5.attrs["source"] = str(pkl_path)
        h5.attrs["num_frames"] = N

        # frame-wise writes
        for i, d in enumerate(data):
            # ---- depth ----
            depth_np = None
            if ds_depth is not None and ("base_depth" in d) and d["base_depth"] is not None:
                depth_np = to_float32_depth(np.asarray(d["base_depth"]))  # (H,W,1)
                ds_depth[i] = depth_np

            # ---- joints ----
            if ds_jpos is not None and ("joint_positions" in d) and d["joint_positions"] is not None:
                jp = np.asarray(d["joint_positions"]).reshape(-1).astype(np.float32)
                ds_jpos[i] = jp[:ds_jpos.shape[1]]
            if ds_jvel is not None and ("joint_velocities" in d) and d["joint_velocities"] is not None:
                jv = np.asarray(d["joint_velocities"]).reshape(-1).astype(np.float32)
                if jv.shape[0] == ds_jvel.shape[1]:
                    ds_jvel[i] = jv

            # ---- ee ----
            if ds_ee is not None and ("ee_pos_quat" in d) and d["ee_pos_quat"] is not None:
                ee = parse_ee_pos_quat(d["ee_pos_quat"])
                if ee is not None and ee.shape == (7,):
                    ds_ee[i] = ee

            # ---- gripper ----
            if (("gripper_position" in d) and d["gripper_position"] is not None and
                (ds_grip_endpose is not None or ds_grip_joint is not None)):
                g = np.asarray(d["gripper_position"]).reshape(-1).astype(np.float32)
                if len(shapes["gripper_position"]) == 1:
                    if ds_grip_endpose is not None: ds_grip_endpose[i] = g[0]
                    if ds_grip_joint   is not None: ds_grip_joint[i]   = g[0]
                    if keep_legacy_gripper and 'ds_grip_legacy' in locals(): ds_grip_legacy[i] = g[0]
                else:
                    if ds_grip_endpose is not None: ds_grip_endpose[i] = g
                    if ds_grip_joint   is not None: ds_grip_joint[i]   = g
                    if keep_legacy_gripper and 'ds_grip_legacy' in locals(): ds_grip_legacy[i] = g

            # ---- pointcloud ----
            if ds_xyzrgb is not None:
                # rgb（直接用当前帧的原始数组而不是重新解码jpeg）
                if ("base_rgb" not in d) or d["base_rgb"] is None or depth_np is None:
                    # 缺任一项则写零占位
                    ds_xyzrgb[i] = np.zeros((k_points, 6), dtype=np.float32)
                else:
                    rgb = to_uint8_rgb(np.asarray(d["base_rgb"]))      # (H,W,3) uint8
                    depth_m = depth_np[..., 0].astype(np.float32)      # (H,W)

                    # 深度单位转换 & 合法范围过滤
                    depth_m = depth_m * float(depth_scale)
                    if np.isfinite(pc_min_depth) or np.isfinite(pc_max_depth):
                        # 拷贝以免影响 ds_depth
                        depth_m = depth_m.copy()
                        if np.isfinite(pc_min_depth):
                            depth_m[depth_m < pc_min_depth] = 0.0
                        if np.isfinite(pc_max_depth):
                            depth_m[depth_m > pc_max_depth] = 0.0

                    # 尺寸不一致时，把 depth 最近邻缩放到 RGB 大小
                    H, W, _ = rgb.shape
                    if depth_m.shape != (H, W):
                        depth_m = cv2.resize(depth_m, (W, H), interpolation=cv2.INTER_NEAREST)

                    fx, fy, cx, cy = intrinsics
                    xyz, valid = project_depth_to_points(depth_m, fx, fy, cx, cy)  # (H*W,3)
                    cols = rgb.reshape(-1, 3).astype(np.float32) / 255.0           # [0,1]

                    xyz_valid = xyz[valid]
                    cols_valid = cols[valid]

                    if xyz_valid.shape[0] == 0:
                        xyz_k = np.zeros((k_points, 3), dtype=np.float32)
                        cols_k = np.zeros((k_points, 3), dtype=np.float32)
                    else:
                        xyz_k = downsample_to_k(xyz_valid, k_points, rng).astype(np.float32)
                        cols_k = downsample_to_k(cols_valid, k_points, rng).astype(np.float32)

                    ds_xyzrgb[i] = np.concatenate([xyz_k, cols_k], axis=1)

def main():
    ap = argparse.ArgumentParser("Batch convert a folder of PKLs to HDF5 with same basenames")
    ap.add_argument("--pkl_dir", required=True, help="输入 pkl 文件夹（不递归）")
    ap.add_argument("--out_dir", required=True, help="输出 hdf5 文件夹（会创建；文件名保持不变，后缀改为 .hdf5）")
    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"], help="非字节数据集压缩方式")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="RGB JPEG 质量 (1-100)")
    ap.add_argument("--keep_legacy_gripper", action="store_true", help="同时保留 /gripper/position 旧路径")
    # ---- 新增：点云相关参数（全部可选；仅在提供完整内参且有rgb+depth时才会生成点云）----
    ap.add_argument("--fx", type=float, default=601.6535 )
    ap.add_argument("--fy", type=float, default=601.79346)
    ap.add_argument("--cx", type=float, default=325.8246)
    ap.add_argument("--cy", type=float, default=237.58635)
    ap.add_argument("--depth_scale", type=float, default=0.001, help="原始深度 * depth_scale -> 米（若深度是毫米，则用 0.001）")
    ap.add_argument("--k", type=int, default=1024, help="每帧点云点数（默认1024）")
    ap.add_argument("--pc_min_depth", type=float, default=0.0, help="点云最小深度（米），小于则视为无效（设为0屏蔽）")
    ap.add_argument("--pc_max_depth", type=float, default=float("inf"), help="点云最大深度（米），大于则视为无效")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（下采样）")
    args = ap.parse_args()

    pkl_dir = Path(args.pkl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 只有当内参全部提供时才启用点云
    intrinsics = None
    if None not in (args.fx, args.fy, args.cx, args.cy):
        intrinsics = (float(args.fx), float(args.fy), float(args.cx), float(args.cy))

    pkl_files = sorted([p for p in pkl_dir.iterdir() if p.suffix.lower() == ".pkl" and p.is_file()])
    if not pkl_files:
        print(f"在 {pkl_dir} 下未找到 .pkl 文件")
        sys.exit(1)

    print(f"将转换 {len(pkl_files)} 个 pkl 文件 → {out_dir}")
    if intrinsics is None:
        print("（未提供完整相机内参 fx,fy,cx,cy，点云部分将被跳过）")

    success, failed = 0, 0
    for idx, pkl_path in enumerate(pkl_files, 1):
        out_path = out_dir / (pkl_path.stem + ".hdf5")
        print(f"[{idx}/{len(pkl_files)}] {pkl_path.name}  ->  {out_path.name}")
        try:
            convert_one(
                pkl_path=pkl_path,
                out_path=out_path,
                compression=args.compression,
                jpeg_quality=args.jpeg_quality,
                keep_legacy_gripper=args.keep_legacy_gripper,
                intrinsics=intrinsics,
                depth_scale=args.depth_scale,
                k_points=args.k,
                pc_min_depth=args.pc_min_depth,
                pc_max_depth=args.pc_max_depth,
                rng_seed=args.seed
            )
            success += 1
        except Exception as e:
            failed += 1
            print(f"❌ 转换失败：{pkl_path.name}  原因：{e}")
            traceback.print_exc(limit=1)

    print(f"\n✅ 完成：成功 {success}，失败 {failed}。输出目录：{out_dir.resolve()}")

if __name__ == "__main__":
    main()
