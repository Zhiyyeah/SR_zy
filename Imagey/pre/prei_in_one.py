#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tarfile
import re
import numpy as np
from osgeo import gdal
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

# ========== 用户配置 ==========
# 原始 .tar 文件所在目录
TAR_DIR = '/public/home/zyye/SR_organized/Imagey'

IMG_DIR = TAR_DIR

# 解压后各场景临时存放目录
EXTRACT_ROOT   = os.path.join(TAR_DIR, 'WaterLand_unzipped')
# 计算好的 TOA 输出目录
TOA_DIR        = os.path.join(TAR_DIR, 'WaterLand_TOA')
# 高分辨率切片输出目录
HR_TILES_DIR   = os.path.join(TAR_DIR, 'WaterLand_TOA_tiles_hr')
# 生成的低分辨率切片输出目录
LR_TILES_DIR   = os.path.join(TAR_DIR, 'WaterLand_TOA_tiles_lr')

# 需要处理的波段列表
BANDS = [1, 2, 3, 4, 5, 6, 7]
# 切片大小
TILE_SIZE = 512
# 低分辨率降采样倍数
SCALE = 8
# ===============================

def extract_archives():
    """解压所有 .tar，输出到 EXTRACT_ROOT/<scene_name>/"""
    os.makedirs(EXTRACT_ROOT, exist_ok=True)
    for fname in os.listdir(IMG_DIR):
        if not fname.lower().endswith('.tar'):
            continue
        tar_path = os.path.join(IMG_DIR, fname)
        scene = fname[:-4]
        out_dir = os.path.join(EXTRACT_ROOT, scene)
        os.makedirs(out_dir, exist_ok=True)
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=out_dir)
        print(f"✅ 已解压 {fname} → {out_dir}")

def get_band_paths(folder, bands):
    """在 folder 中匹配 _B{b}.TIF 文件，返回按 bands 排序的路径列表"""
    paths = []
    for b in bands:
        matches = [f for f in os.listdir(folder) if f.endswith(f"_B{b}.TIF")]
        if not matches:
            raise FileNotFoundError(f"⚠️ 在 {folder} 中未找到 Band {b}")
        paths.append(os.path.join(folder, matches[0]))
    return paths

def parse_mtl(mtl_path, bands):
    """解析 MTL.txt，提取反射率增益与偏移，以及太阳高度角"""
    txt = open(mtl_path, 'r').read()
    gains, offsets = [], []
    for b in bands:
        g = re.search(rf'REFLECTANCE_MULT_BAND_{b}\s*=\s*([0-9E.+-]+)', txt).group(1)
        o = re.search(rf'REFLECTANCE_ADD_BAND_{b}\s*=\s*([0-9E.+-]+)', txt).group(1)
        gains.append(float(g)); offsets.append(float(o))
    sun_elev = float(re.search(r'SUN_ELEVATION\s*=\s*([0-9E.+-]+)', txt).group(1))
    return gains, offsets, sun_elev

def read_image_stack(band_paths):
    """用 GDAL 读入多波段 stack，返回 numpy 数组和参考 Dataset"""
    ds0 = gdal.Open(band_paths[0])
    rows, cols = ds0.RasterYSize, ds0.RasterXSize
    stack = np.zeros((rows, cols, len(band_paths)), dtype=np.float32)
    for i, p in enumerate(band_paths):
        ds = gdal.Open(p)
        stack[:, :, i] = ds.GetRasterBand(1).ReadAsArray()
    return stack, ds0

def apply_calibration(stack, gains, offsets, sun_elev):
    """按公式 (DN * gain + offset) / sin(sun_elev) 计算 TOA"""
    toa = np.empty_like(stack, dtype=np.float32)
    s = np.sin(np.deg2rad(sun_elev))
    for i in range(stack.shape[2]):
        toa[:, :, i] = (stack[:, :, i] * gains[i] + offsets[i]) / s
    # 把原始 DN==0 的像素设为 NaN
    toa[stack == 0] = np.nan
    return toa

def write_tiff(image, ref_ds, out_path):
    """用 GDAL 将 image (rows×cols×bands) 写成 GeoTIFF"""
    driver = gdal.GetDriverByName('GTiff')
    out = driver.Create(out_path,
                        ref_ds.RasterXSize, ref_ds.RasterYSize,
                        image.shape[2], gdal.GDT_Float32)
    out.SetGeoTransform(ref_ds.GetGeoTransform())
    out.SetProjection(ref_ds.GetProjection())
    for i in range(image.shape[2]):
        out.GetRasterBand(i+1).WriteArray(image[:, :, i])
    out.FlushCache()
    out = None

def generate_toa():
    """遍历 EXTRACT_ROOT 下每个场景，生成 TOA.tif 到 TOA_DIR"""
    os.makedirs(TOA_DIR, exist_ok=True)
    for scene in os.listdir(EXTRACT_ROOT):
        folder = os.path.join(EXTRACT_ROOT, scene)
        if not os.path.isdir(folder):
            continue
        try:
            print(f"🚀 计算 TOA: {scene}")
            band_paths = get_band_paths(folder, BANDS)
            stack, ref_ds = read_image_stack(band_paths)
            mtl = [f for f in os.listdir(folder) if f.endswith('_MTL.txt')]
            if not mtl:
                print(f"❌ 未找到 MTL，跳过 {scene}")
                continue
            gains, offsets, sun_elev = parse_mtl(os.path.join(folder, mtl[0]), BANDS)
            toa = apply_calibration(stack, gains, offsets, sun_elev)
            out_path = os.path.join(TOA_DIR, f"{scene}_TOA.tif")
            write_tiff(toa, ref_ds, out_path)
            print(f"✅ 输出 TOA: {out_path}\n")
        except Exception as e:
            print(f"❌ 错误 {scene}: {e}\n")

def tile_hr():
    """将 TOA_DIR 中每个 .tif 切成 512×512 的无 NaN HR 片，存到 HR_TILES_DIR"""
    os.makedirs(HR_TILES_DIR, exist_ok=True)
    total = 0
    for fname in os.listdir(TOA_DIR):
        if not fname.lower().endswith('.tif'):
            continue
        path = os.path.join(TOA_DIR, fname)
        with rasterio.open(path) as src:
            H, W = src.height, src.width
            bands = src.count
            count = 0
            for i in range(0, H, TILE_SIZE):
                for j in range(0, W, TILE_SIZE):
                    if i + TILE_SIZE > H or j + TILE_SIZE > W:
                        continue
                    win = Window(j, i, TILE_SIZE, TILE_SIZE)
                    block = src.read(window=win)
                    if np.isnan(block).any():
                        continue
                    meta = src.meta.copy()
                    meta.update({
                        'height': TILE_SIZE,
                        'width' : TILE_SIZE,
                        'transform': src.window_transform(win),
                        'dtype': block.dtype
                    })
                    out_name = f"{os.path.splitext(fname)[0]}_r{i:05d}_c{j:05d}.tif"
                    out_path = os.path.join(HR_TILES_DIR, out_name)
                    with rasterio.open(out_path, 'w', **meta) as dst:
                        dst.write(block)
                    count += 1
                    total += 1
            print(f"✅ {fname} 生成 HR 切片 {count} 张")
    print(f"\n🎉 共生成 HR 切片 {total} 张")

def tile_lr():
    """对 HR_TILES_DIR 中每个切片降采样 SCALE 倍，存到 LR_TILES_DIR"""
    os.makedirs(LR_TILES_DIR, exist_ok=True)
    cnt = 0
    for fname in os.listdir(HR_TILES_DIR):
        if not fname.lower().endswith('.tif'):
            continue
        hr_path = os.path.join(HR_TILES_DIR, fname)
        lr_path = os.path.join(LR_TILES_DIR, fname)
        with rasterio.open(hr_path) as src:
            h, w = src.height, src.width
            new_h, new_w = h // SCALE, w // SCALE
            data = src.read(
                out_shape=(src.count, new_h, new_w),
                resampling=Resampling.bilinear
            )
            transform = src.transform * src.transform.scale(
                (w / new_w), (h / new_h)
            )
            meta = src.meta.copy()
            meta.update({
                'height': new_h,
                'width' : new_w,
                'transform': transform
            })
            with rasterio.open(lr_path, 'w', **meta) as dst:
                dst.write(data)
        cnt += 1
        
    print(f"\n✅ 共生成 LR 切片 {cnt} 张 至：{LR_TILES_DIR}")

if __name__ == '__main__':
    extract_archives()
    generate_toa()
    tile_hr()
    tile_lr()
    print("\n✔️ 全流程完成！")
