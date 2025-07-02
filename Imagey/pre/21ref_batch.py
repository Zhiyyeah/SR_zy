import os
import numpy as np
import re
from osgeo import gdal

def get_band_paths(folder, bands):
    files = []
    for b in bands:
        fname = [f for f in os.listdir(folder) if f.endswith(f"_B{b}.TIF")]
        if fname:
            files.append(os.path.join(folder, fname[0]))
        else:
            raise FileNotFoundError(f"⚠️ Band {b} not found in {folder}")
    return files

def read_image_stack(band_paths):
    ds = gdal.Open(band_paths[0])
    if ds is None:
        raise RuntimeError(f"❌ 无法打开影像: {band_paths[0]}")
    rows, cols = ds.RasterYSize, ds.RasterXSize
    stack = np.zeros((rows, cols, len(band_paths)), dtype=np.float32)

    for i, path in enumerate(band_paths):
        band_ds = gdal.Open(path)
        stack[:, :, i] = band_ds.GetRasterBand(1).ReadAsArray()
    return stack, ds

def parse_mtl(mtl_path, bands):
    with open(mtl_path, 'r') as f:
        content = f.read()

    gains, offsets = [], []
    for b in bands:
        g = re.search(f'REFLECTANCE_MULT_BAND_{b} *= *([\dE.+-]+)', content)
        o = re.search(f'REFLECTANCE_ADD_BAND_{b} *= *([\dE.+-]+)', content)
        gains.append(float(g.group(1)))
        offsets.append(float(o.group(1)))

    sun_elev = float(re.search(r'SUN_ELEVATION *= *([\dE.+-]+)', content).group(1))
    return gains, offsets, sun_elev

def apply_calibration(image_stack, gains, offsets, sun_elevation):
    result = np.zeros_like(image_stack, dtype=np.float32)
    sin_theta = np.sin(np.deg2rad(sun_elevation))
    for i in range(image_stack.shape[2]):
        result[:, :, i] = (image_stack[:, :, i] * gains[i] + offsets[i]) / sin_theta
        result[image_stack[:, :, i] == 0] = np.nan
    return result

def write_output(image, ref_ds, out_path):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(out_path, ref_ds.RasterXSize, ref_ds.RasterYSize, image.shape[2], gdal.GDT_Float32)
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    for i in range(image.shape[2]):
        out_ds.GetRasterBand(i + 1).WriteArray(image[:, :, i])
    out_ds.FlushCache()
    del out_ds

# ---------- 主批处理入口 ----------
if __name__ == '__main__':
    root = "/home/zyye/SR/SRCNN_my/Imagery/Water"
    bands = [1, 2, 3, 4, 5, 6, 7]

    subfolders = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    for folder in subfolders:
        try:
            print(f"🚀 处理: {os.path.basename(folder)}")
            band_paths = get_band_paths(folder, bands)
            image_stack, ref_ds = read_image_stack(band_paths)

            mtl_txt = [f for f in os.listdir(folder) if f.endswith("_MTL.txt")]
            if not mtl_txt:
                print(f"❌ MTL 文件未找到，跳过 {folder}")
                continue
            mtl_path = os.path.join(folder, mtl_txt[0])

            gains, offsets, sun_elev = parse_mtl(mtl_path, bands)
            calibrated = apply_calibration(image_stack, gains, offsets, sun_elev)

            output_path = os.path.join(folder, f"{os.path.basename(folder)}_TOA.tif")
            write_output(calibrated, ref_ds, output_path)
            print(f"✅ 输出完成: {output_path}\n")

        except Exception as e:
            print(f"❌ 错误：{folder}\n→ {e}\n")
