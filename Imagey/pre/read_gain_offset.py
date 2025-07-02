import re

def parse_mtl_multipliers(mtl_path, band_nums, mode='reflectance'):
    """
    从 Landsat MTL 文件中提取 gain、offset、太阳高度角

    参数:
        mtl_path : str        MTL 文件路径
        band_nums : List[int] 要提取的波段编号，例如 [1,2,3,4,5,6,7]
        mode : 'reflectance' 或 'radiance'

    返回:
        gain_list : 每个波段的乘数（list）
        offset_list : 每个波段的加数（list）
        sun_elevation : float（仅 reflectance 模式需要）
    """
    with open(mtl_path, 'r') as f:
        content = f.read()

    gain_list = []
    offset_list = []

    for b in band_nums:
        if mode == 'reflectance':
            g = re.search(f'REFLECTANCE_MULT_BAND_{b} *= *([\dE.+-]+)', content)
            o = re.search(f'REFLECTANCE_ADD_BAND_{b} *= *([\dE.+-]+)', content)
        elif mode == 'radiance':
            g = re.search(f'RADIANCE_MULT_BAND_{b} *= *([\dE.+-]+)', content)
            o = re.search(f'RADIANCE_ADD_BAND_{b} *= *([\dE.+-]+)', content)
        else:
            raise ValueError("mode must be 'reflectance' or 'radiance'")

        gain_list.append(float(g.group(1)))
        offset_list.append(float(o.group(1)))

    sun_elev = None
    # if mode == 'reflectance':
    match = re.search(r'SUN_ELEVATION *= *([\dE.+-]+)', content)
    sun_elev = float(match.group(1)) if match else None

    return gain_list, offset_list, sun_elev

mtl_path = "Imgaery/Water/LC08_L1GT_116038_20230131_20230208_02_T2/LC08_L1GT_116038_20230131_20230208_02_T2_MTL.txt"
band_nums = [1,2,3,4,5,6,7]  # 你想提取的波段

# 获取反射率定标参数和太阳高度角
gains, offsets, sun_elev = parse_mtl_multipliers(mtl_path, band_nums, mode='reflectance')

# 输出举例
for i, b in enumerate(band_nums):
    print(f"Band {b}: gain = {gains[i]:.8f}, offset = {offsets[i]:.8f}")
print(f"Sun Elevation: {sun_elev:.3f}°")
