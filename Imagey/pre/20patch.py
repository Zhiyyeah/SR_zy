from rasterio.windows import Window
import os
import rasterio
import numpy as np

# ===== 用户配置 =====
tif_root_dir    = "/public/home/zyye/SR_organized/Imagey"      # 各场景父目录
output_root_dir = "/public/home/zyye/SR_organized/Imagey_tiles"  # 切片输出父目录
tile_size       = 512                                       # 切片大小（像素）

# 要提取的光谱波段编号
spectral_idxs  = [1,2,3,4,5,6,7,9]
# 要提取的角度波段后缀
angle_suffixes = ["VZA", "VAA", "SZA", "SAA"]
# =====================

os.makedirs(output_root_dir, exist_ok=True)

for scene_name in os.listdir(tif_root_dir):
    scene_dir = os.path.join(tif_root_dir, scene_name)
    if not os.path.isdir(scene_dir):
        continue

    # 创建输出场景目录
    out_scene_dir = os.path.join(output_root_dir, scene_name)
    os.makedirs(out_scene_dir, exist_ok=True)

    # 列出所有文件名
    all_files = os.listdir(scene_dir)

    # 1) 筛选光谱波段文件
    band_files = []
    for idx in spectral_idxs:
        pat = f"_B{idx}.TIF"
        matches = [f for f in all_files if f.upper().endswith(pat)]
        if not matches:
            raise FileNotFoundError(f"场景 `{scene_name}` 中未找到波段 B{idx} 的 TIF 文件")
        band_files.append(matches[0])

    # 2) 筛选角度波段文件
    for suf in angle_suffixes:
        pat = f"_{suf}.TIF"
        matches = [f for f in all_files if f.upper().endswith(pat)]
        if not matches:
            raise FileNotFoundError(f"场景 `{scene_name}` 中未找到角度波段 `{suf}` 的 TIF 文件")
        band_files.append(matches[0])

    # 按用户指定顺序排列：先 B1–B7、B9，再四个角度
    # （已经按上面添加顺序，不再排序）

    # 打开所有波段
    src_files = [
        rasterio.open(os.path.join(scene_dir, bf))
        for bf in band_files
    ]

    # 校验尺寸一致性
    H, W = src_files[0].height, src_files[0].width
    for src in src_files:
        assert src.height == H and src.width == W, (
            f"{scene_name} 中文件 {src.name} 尺寸不一致"
        )

    # 用第一个波段的 dataset_mask 作为有效像素掩膜
    mask = src_files[0].dataset_mask()  # 255=有效，0=无效

    # 拷贝元数据模板并更新波段数
    meta = src_files[0].meta.copy()
    meta.update({
        "count": len(src_files),
        "dtype": src_files[0].dtypes[0],
    })

    cnt = 0
    # 开始按网格裁切
    for i in range(0, H, tile_size):
        for j in range(0, W, tile_size):
            # 跳过边缘不足的
            if i + tile_size > H or j + tile_size > W:
                continue

            block = mask[i:i+tile_size, j:j+tile_size]
            # 仅保留全 255（全有效像素）的窗口
            if not (block == 255).all():
                continue

            # 读取每个波段的这一窗口
            data_stack = [
                src.read(1, window=Window(j, i, tile_size, tile_size))
                for src in src_files
            ]
            tile = np.stack(data_stack, axis=0)  # (bands, H, W)

            # 更新地理变换和尺寸
            meta.update({
                "height": tile_size,
                "width" : tile_size,
                "transform": src_files[0].window_transform(
                    Window(j, i, tile_size, tile_size)
                ),
            })

            # 写出文件
            out_fp = os.path.join(
                out_scene_dir,
                f"{scene_name}_tile_{i:05d}_{j:05d}.tif"
            )
            with rasterio.open(out_fp, 'w', **meta) as dst:
                dst.write(tile)

            cnt += 1

    # 关闭所有打开的文件
    for src in src_files:
        src.close()

    print(f"✅ 场景 `{scene_name}`：共生成 {cnt} 个 {tile_size}×{tile_size} 切片")

print("🎉 全部场景处理完成！")
