import os
import numpy as np
import re
from osgeo import gdal

def get_band_paths(folder, bands):
    return [os.path.join(folder, f"LC08_L1GT_116038_20230131_20230208_02_T2_B{b}.TIF") for b in bands]

def read_image_stack(band_paths):
    ds = gdal.Open(band_paths[0])
    rows, cols = ds.RasterYSize, ds.RasterXSize
    stack = np.zeros((rows, cols, len(band_paths)), dtype=np.float32)

    for i, path in enumerate(band_paths):
        band_ds = gdal.Open(path)
        stack[:, :, i] = band_ds.GetRasterBand(1).ReadAsArray()
    return stack, ds

def parse_mtl(mtl_path, bands, mode='reflectance'):
    with open(mtl_path, 'r') as f:
        content = f.read()
    gains, offsets = [], []

    for b in bands:
        if mode == 'reflectance':
            g = re.search(f'REFLECTANCE_MULT_BAND_{b} *= *([\dE.+-]+)', content)
            o = re.search(f'REFLECTANCE_ADD_BAND_{b} *= *([\dE.+-]+)', content)
        else:
            g = re.search(f'RADIANCE_MULT_BAND_{b} *= *([\dE.+-]+)', content)
            o = re.search(f'RADIANCE_ADD_BAND_{b} *= *([\dE.+-]+)', content)
        gains.append(float(g.group(1)))
        offsets.append(float(o.group(1)))

    sun_elev = None
    if mode == 'reflectance':
        match = re.search(r'SUN_ELEVATION *= *([\dE.+-]+)', content)
        sun_elev = float(match.group(1))
    return gains, offsets, sun_elev

def apply_calibration(image_stack, gains, offsets, sun_elevation=None):
    calibrated = np.zeros_like(image_stack, dtype=np.float32)
    for i in range(image_stack.shape[2]):
        calibrated[:, :, i] = image_stack[:, :, i] * gains[i] + offsets[i]
        if sun_elevation is not None:
            calibrated[:, :, i] /= np.sin(np.deg2rad(sun_elevation))
        calibrated[image_stack[:, :, i] == 0] = np.nan
    return calibrated

def write_multiband_tiff(image, ref_ds, output_path):
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, ref_ds.RasterXSize, ref_ds.RasterYSize, image.shape[2], gdal.GDT_Float32)
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    for i in range(image.shape[2]):
        out_ds.GetRasterBand(i + 1).WriteArray(image[:, :, i])
    out_ds.FlushCache()
    del out_ds

# ---------- 主程序 ----------
if __name__ == '__main__':
    folder = "/home/zyye/SR/SRCNN_my/Imgaery/Water/LC08_L1GT_116038_20230131_20230208_02_T2"
    bands = [1, 2, 3, 4, 5, 6, 7]

    band_paths = get_band_paths(folder, bands)
    print("🔍 拼接后的路径：")
    for path in band_paths:
        print(path)

    image_stack, ref_ds = read_image_stack(band_paths)

    mtl_path = os.path.join(folder, "LC08_L1GT_116038_20230131_20230208_02_T2_MTL.txt")
    gains, offsets, sun_elev = parse_mtl(mtl_path, bands, mode='reflectance')

    calibrated = apply_calibration(image_stack, gains, offsets, sun_elev)
    output_path = os.path.join(folder, "LC08_B1toB7_TOA.tif")
    write_multiband_tiff(calibrated, ref_ds, output_path)

    print(f"✅ 辐射定标完成，输出路径：{output_path}")
