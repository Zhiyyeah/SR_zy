import tifffile
import numpy as np

# 把这里换成之前报错的那个HR文件的完整路径
problem_file_path = r"D:\Py_Code\img_match\SR_Imagery\tif\HR\HR_LC09_L1TP_116036_20250504_20250504_02_T1_r05888_c03840.tif"

print(f"Checking file: {problem_file_path}")

try:
    # 模拟 data_loader 中的读取过程
    image_array = tifffile.imread(problem_file_path)
    
    print(f"Successfully read. Original shape: {image_array.shape}")
    print(f"Data type: {image_array.dtype}")
    print(f"Number of dimensions (ndim): {image_array.ndim}")

    if image_array.ndim == 3:
        # 尝试转换
        transposed_array = image_array.transpose(2, 0, 1)
        print(f"Transposed shape: {transposed_array.shape}")
    else:
        print("Error: Image is not 3-dimensional, cannot transpose.")

except Exception as e:
    print(f"An error occurred while reading the file: {e}")