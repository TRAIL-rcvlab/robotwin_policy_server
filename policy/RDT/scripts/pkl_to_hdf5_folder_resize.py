#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pkl_dir_to_hdf5_batch.py

批量将目录中的 pkl（list[dict]）转换为 hdf5（同名输出，后缀 .hdf5）。
字段映射（若缺失则不创建）：
- base_rgb:         -> /observation/front_camera/rgb        (N,)   |Smaxlen（JPEG字节，已resize）
- base_depth:       -> /observation/front_camera/depth_raw  (N,H,W,1) float32
- joint_positions:  -> /joint_action/positions              (N,7)  float32   # 若>7维，丢弃最后一维
- joint_velocities: -> /joint_action/velocities             (N,Jv) float32
- ee_pos_quat:      -> /endpose/ee_pos_quat                 (N,7)  float64
- gripper_position: -> /endpose/gripper & /joint_action/gripper  (N,)或(N,K) float32
                      （可选保留 legacy: /gripper/position）

用法示例：
python pkl_dir_to_hdf5_batch.py --pkl_dir /path/to/pkls --out_dir /path/to/h5s
python pkl_dir_to_hdf5_batch.py --pkl_dir ./pkls --out_dir ./h5s --jpeg_quality 90 --compression gzip
python pkl_dir_to_hdf5_batch.py --pkl_dir ./pkls --out_dir ./h5s --keep_legacy_gripper
# 指定RGB尺寸（默认 320x240）
python pkl_dir_to_hdf5_batch.py --pkl_dir ./pkls --out_dir ./h5s --rgb_width 320 --rgb_height 240
# 你的路径例子：
# python pkl_dir_to_hdf5_batch.py --pkl_dir /path/to/pkls --out_dir task/demo_clean/data/
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
    """将任意RGB/灰度数组标准化为 uint8 RGB（三通道）。"""
    if arr is None:
        raise ValueError("RGB 输入为 None")
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
    """将 uint8 RGB(BHWC或HWC) 编码为 JPEG 字节。"""
    if rgb_uint8.ndim == 4 and rgb_uint8.shape[0] == 1:
        rgb_uint8 = rgb_uint8[0]
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
        raise ValueError(f"encode_rgb_to_jpeg_bytes 期望形状 (H,W,3)，得到 {rgb_uint8.shape}")
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return bytes(buf)

def to_float32_depth(arr: np.ndarray) -> np.ndarray:
    """将深度图标准化为 float32 (H,W,1)。不做缩放。"""
    d = arr
    if d.ndim == 2:
        d = d[..., None]
    if d.ndim == 3 and d.shape[-1] > 1:
        d = d[..., :1]
    if not (d.ndim == 3 and d.shape[-1] == 1):
        raise ValueError(f"Unexpected depth shape: {arr.shape}")
    return d.astype(np.float32)

def parse_ee_pos_quat(ee_obj) -> Optional[np.ndarray]:
    """解析为 [x,y,z,qx,qy,qz,qw] float64，或返回 None。"""
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
    """在数据列表中找到首个包含 key 的条目，返回 (index, value)。若无则 (None, None)。"""
    for i, d in enumerate(data_list):
        if isinstance(d, dict) and key in d and d[key] is not None:
            return i, d[key]
    return None, None

# ------------- core conversion -------------
def convert_one(
    pkl_path: Path,
    out_path: Path,
    compression: str,
    jpeg_quality: int,
    keep_legacy_gripper: bool,
    rgb_width: int,
    rgb_height: int,
):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError(f"{pkl_path.name}: 期望 pkl 顶层为非空 list。")

    N = len(data)
    comp = None if compression == "none" else compression

    has, shapes = {}, {}

    # base_rgb → bytes（将被resize到 rgb_height x rgb_width）
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

    if not any(has.values()):
        raise ValueError(f"{pkl_path.name}: 未发现任何可转换字段。")

    # 预编码 RGB 为 JPEG 字节（dtype=固定长度字节串）
    rgb_bytes_list, max_len = None, 0
    if has.get("base_rgb"):
        rgb_bytes_list = []
        for i, d in enumerate(data):
            if ("base_rgb" not in d) or (d["base_rgb"] is None):
                raise KeyError(f"{pkl_path.name}: 第 {i} 帧缺少 base_rgb")
            rgb = to_uint8_rgb(np.asarray(d["base_rgb"]))
            # --- Resize 到 (rgb_width, rgb_height) ---
            # 注意：OpenCV resize 的 size=(宽, 高)
            rgb = cv2.resize(rgb, (int(rgb_width), int(rgb_height)), interpolation=cv2.INTER_AREA)
            buf = encode_rgb_to_jpeg_bytes(rgb, quality=jpeg_quality)
            rgb_bytes_list.append(buf)
            if len(buf) > max_len:
                max_len = len(buf)

    if out_path.exists():
        out_path.unlink()

    with h5py.File(out_path, "w") as h5:
        ds_depth = ds_jpos = ds_jvel = ds_ee = None
        ds_grip_endpose = ds_grip_joint = ds_grip_legacy = None

        # RGB bytes → /observation/front_camera/rgb
        if rgb_bytes_list is not None:
            grp_obs = h5.require_group("observation")
            grp_cam = grp_obs.require_group("front_camera")
            dt = h5py.string_dtype(encoding="ascii", length=max_len)  # 固定长度字节串 |S{max_len}
            ds_rgb = grp_cam.create_dataset("rgb", shape=(N,), dtype=dt)
            ds_rgb.attrs["encoding"] = "jpeg"
            ds_rgb.attrs["note"] = "Fixed-length byte strings; JPEG-encoded resized RGB per frame."
            ds_rgb.attrs["resize_hw"] = (int(rgb_height), int(rgb_width))  # (H, W)
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
            ds_depth.attrs["desc"] = "Raw depth frames (float32, no resize)"

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

        # meta
        h5.attrs["source"] = str(pkl_path)
        h5.attrs["num_frames"] = N

        # frame-wise writes
        for i, d in enumerate(data):
            if ds_depth is not None and ("base_depth" in d) and d["base_depth"] is not None:
                depth = to_float32_depth(np.asarray(d["base_depth"]))
                ds_depth[i] = depth
            if ds_jpos is not None and ("joint_positions" in d) and d["joint_positions"] is not None:
                jp = np.asarray(d["joint_positions"]).reshape(-1).astype(np.float32)
                ds_jpos[i] = jp[:ds_jpos.shape[1]]  # 丢掉多余维度（>7）
            if ds_jvel is not None and ("joint_velocities" in d) and d["joint_velocities"] is not None:
                jv = np.asarray(d["joint_velocities"]).reshape(-1).astype(np.float32)
                if jv.shape[0] == ds_jvel.shape[1]:
                    ds_jvel[i] = jv
            if ds_ee is not None and ("ee_pos_quat" in d) and d["ee_pos_quat"] is not None:
                ee = parse_ee_pos_quat(d["ee_pos_quat"])
                if ee is not None and ee.shape == (7,):
                    ds_ee[i] = ee
            if (("gripper_position" in d) and d["gripper_position"] is not None and
                (ds_grip_endpose is not None or ds_grip_joint is not None)):
                g = np.asarray(d["gripper_position"]).reshape(-1).astype(np.float32)
                # 写两份
                if len(shapes["gripper_position"]) == 1:
                    if ds_grip_endpose is not None: ds_grip_endpose[i] = g[0]
                    if ds_grip_joint   is not None: ds_grip_joint[i]   = g[0]
                    if keep_legacy_gripper and 'ds_grip_legacy' in locals(): ds_grip_legacy[i] = g[0]
                else:
                    if ds_grip_endpose is not None: ds_grip_endpose[i] = g
                    if ds_grip_joint   is not None: ds_grip_joint[i]   = g
                    if keep_legacy_gripper and 'ds_grip_legacy' in locals(): ds_grip_legacy[i] = g

def main():
    ap = argparse.ArgumentParser("Batch convert a folder of PKLs to HDF5 with same basenames")
    ap.add_argument("--pkl_dir", required=True, help="输入 pkl 文件夹（不递归）")
    ap.add_argument("--out_dir", required=True, help="输出 hdf5 文件夹（会创建；文件名保持不变，后缀改为 .hdf5）")
    ap.add_argument("--compression", default="gzip", choices=["gzip", "lzf", "none"], help="非字节数据集压缩方式")
    ap.add_argument("--jpeg_quality", type=int, default=95, help="RGB JPEG 质量 (1-100)")
    ap.add_argument("--keep_legacy_gripper", action="store_true", help="同时保留 /gripper/position 旧路径")
    # 新增：RGB 目标尺寸（默认 320x240）
    ap.add_argument("--rgb_width", type=int, default=320, help="RGB 目标宽度（默认 320）")
    ap.add_argument("--rgb_height", type=int, default=240, help="RGB 目标高度（默认 240）")
    args = ap.parse_args()

    pkl_dir = Path(args.pkl_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_files = sorted([p for p in pkl_dir.iterdir() if p.suffix.lower() == ".pkl" and p.is_file()])
    if not pkl_files:
        print(f"在 {pkl_dir} 下未找到 .pkl 文件")
        sys.exit(1)

    print(f"将转换 {len(pkl_files)} 个 pkl 文件 → {out_dir}（RGB 将被 resize 到 {args.rgb_height}×{args.rgb_width}）")
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
                rgb_width=args.rgb_width,
                rgb_height=args.rgb_height,
            )
            success += 1
        except Exception as e:
            failed += 1
            print(f"❌ 转换失败：{pkl_path.name}  原因：{e}")
            traceback.print_exc(limit=1)

    print(f"\n✅ 完成：成功 {success}，失败 {failed}。输出目录：{out_dir.resolve()}")

if __name__ == "__main__":
    main()
