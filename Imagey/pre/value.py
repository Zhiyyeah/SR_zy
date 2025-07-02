import os
import glob
import rasterio
import numpy as np

folder = "/home/zyye/SR/SRCNN_my/Imgaery/Water_tiles_all"

tif_files = sorted(glob.glob(os.path.join(folder, "*.tif")))

for tif_path in tif_files[0:10]:
    print(f"\n📂 正在处理：{os.path.basename(tif_path)}")
    with rasterio.open(tif_path) as src:
        for i in range(1, src.count + 1):
            band = src.read(i)
            print(f"  波段 {i}: min={band.min()}, max={band.max()}")
