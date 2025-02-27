
import time, sys
from functools import wraps
from pathlib import Path
from hashlib import md5
from typing import List, Dict,Tuple

import numpy as np
import pandas as pd
import cv2
import h5py
from anndata import AnnData
import json

def write_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def read_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)
    
def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Start {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Finish {func.__name__} in {elapsed_time:.4f} s")
        return result
    return wrapper

def progress_bar(title, iterable, total):
    sys.stdout.write(title + '\n')
    step = total // 1000
    def iter_with_bar(): 
        for i, item in enumerate(iterable, 1):
            yield item
            if i % step == 0:
                percent = (i / total) * 100
                bar = '█' * (i * 50 // total)
                spaces = ' ' * (50 - len(bar))
                sys.stdout.write(f"\r[{bar}{spaces}] {percent:.1f}%")
                sys.stdout.flush()
        sys.stdout.write('\n')
    return iter_with_bar

def fast_to_csv(df: pd.DataFrame, file: Path, sep="\t"):
    with open(file, "w") as f:
        f.write(sep.join(df.columns) + "\n")
        for _, row in df.iterrows():
            f.write(sep.join(map(str, row)) + "\n")

class AffineTransform:
    def __init__(self, A_points, B_points):
        """
        初始化仿射变换模型。
        :param A_points: N x 2 数组，表示坐标系 A 下的点
        :param B_points: N x 2 数组，表示坐标系 B 下的对应点
        """
        self.A_points = np.array(A_points)
        self.B_points = np.array(B_points)
        self.M = None  # 变换矩阵
        self.T = None  # 平移向量
        self._compute_transformation()

    def _compute_transformation(self):
        """ 计算仿射变换矩阵和偏移量 """
        N = self.A_points.shape[0]
        X = np.hstack([self.A_points, np.ones((N, 1))])  # 添加一列 1 进行平移
        
        # 目标向量
        U = self.B_points[:, 0]  # B 坐标系的 u
        V = self.B_points[:, 1]  # B 坐标系的 v
        
        # 计算最小二乘解
        params_u, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        params_v, _, _, _ = np.linalg.lstsq(X, V, rcond=None)
        
        # 提取参数
        a, b, c = params_u  # u = ax + by + c
        d, e, f = params_v  # v = dx + ey + f
        
        # 存储变换矩阵和偏移量
        self.M = np.array([[a, b], [d, e]])
        self.T = np.array([c, f])
    
    def transform(self, A_point):
        """ 将 A 坐标系的点变换到 B 坐标系 """
        A_point = np.array(A_point)
        return self.M @ A_point + self.T

    def transform_batch(self, A_points):
        """ 对多个点进行变换 """
        A_points = np.array(A_points)
        return (self.M @ A_points.T).T + self.T
    
    def get_parameters(self):
        """ 获取仿射变换矩阵和偏移量 """
        return self.M, self.T
    
    @property
    def resolution(self):
        s_x = np.linalg.norm(self.M[:, 0])  # 第1列
        s_y = np.linalg.norm(self.M[:, 1])  # 第2列
        return 1/np.mean((s_x,s_y))

def hash_to_dna(index, length=16, suffix="-1"):
    """ 通过哈希值生成 DNA 序列 """
    bases = "ACGT"
    hash_val = md5(str(index).encode()).hexdigest()  # 生成哈希
    dna_seq = "".join(bases[int(c, 16) % 4] for c in hash_val[:length])
    return f"{dna_seq}{suffix}"

def write_10X_h5(adata:AnnData, file:Path, metadata={}) -> None:
    """\
    Writes an AnnData object to a 10X Genomics formatted HDF5 file.
    
    This function creates a file compatible with Seurat's Read10X_h5 function. 
    It writes the sparse matrix data and associated metadata while ensuring correct data types and attributes.
    
    Args:
        adata: AnnData object containing the matrix and metadata.
        file: Output file path. Appends '.h5' if not present.
        
    Raises:
        FileExistsError: If the output file already exists.
    copy from: http://xuzhougeng.com/archives/write10xh5foradata
    """
    
    # Helper function to calculate max integer size
    def int_max(x):
        return int(max(np.floor(np.log10(max(x)+1)), 1) * 4)
    
    # Helper function to calculate max string size
    def str_max(x):
        return max(len(str(i)) for i in x) + 1  # +1 for null termination
    
    # must transpose
    X = adata.X.T
    # Create file and write data
    with h5py.File(file, 'w') as w:

        for key, value in metadata.items():
            w.attrs[key] = value

        grp = w.create_group("matrix")

        grp.create_dataset("data", data=X.data, dtype=np.float32)
        grp.create_dataset("indices", data=X.indices, dtype=np.int32)
        grp.create_dataset("indptr", data=X.indptr, dtype=np.int32)
        grp.create_dataset("shape", data=np.array(X.shape, dtype=np.int32))

        # Handling barcodes and features
        grp.create_dataset("barcodes", data=np.array(adata.obs_names, dtype=f'S{str_max(adata.obs_names)}'))
        
        ftrs = grp.create_group("features")
        ftrs.create_dataset("id", data=np.array(adata.var_names, dtype=f'S{str_max(adata.var_names)}'))
        ftrs.create_dataset("name", data=np.array(adata.var_names, dtype=f'S{str_max(adata.var_names)}'))

        # Optionally add more metadata fields if needed
        if 'genome' in adata.var.columns:
            ftrs.create_dataset("genome", data=np.array(adata.var['genome'], dtype=f'S{str_max(adata.var["genome"])}'))
        # set feature_types
        if 'feature_types' in adata.var.columns:
            ftrs.create_dataset("feature_type", data=np.array(adata.var['feature_types'], dtype=f'S{str_max(adata.var["feature_types"])}'))
        else:
            adata.var['feature_types'] = 'Gene Expression'
            ftrs.create_dataset("feature_type", data=np.array(adata.var['feature_types'], dtype=f'S{str_max(adata.var["feature_types"])}'))

def auto_tissue_mask(img : np.ndarray,
              CANNY_THRESH_1 = 100,
              CANNY_THRESH_2 = 200,
              apertureSize=5,
              L2gradient = True) -> np.ndarray:
    if len(img.shape)==3:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    elif len(img.shape)==2:
        gray=(img*((1, 255)[np.max(img)<=1])).astype(np.uint8)
    else:
        print("Image format error!")
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2,apertureSize = apertureSize, L2gradient = L2gradient)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    cnt_info = []
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        cnt_info.append((c,cv2.isContourConvex(c),cv2.contourArea(c),))
    cnt_info.sort(key=lambda c: c[2], reverse=True)
    cnt=cnt_info[0][0]
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], contourIdx=-1, color=255, thickness=-1)
    return mask

def image_resize(img, scalef:float=None, shape=None):
    if shape == None and scalef != None:
        resized_image = cv2.resize(img, None,fx=scalef, fy=scalef, interpolation=cv2.INTER_AREA)
    elif shape != None:
        resized_image = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    else:
        raise ValueError("Please input scale factor or target shape")
    return resized_image

def calculate_scale_factor(original, target):
    """
    计算原始图像到目标图像的缩放比例。

    :param original: 原始图像
    :param target: 目标图像
    :return: 缩放比例
    """
    original_width, original_height = original.shape[:2]
    target_width, target_height = target.shape[:2]

    # 计算宽度和高度方向的缩放比例
    width_scale = target_width / original_width
    height_scale = target_height / original_height

    # 选择较小的缩放比例以保持图像纵横比
    scale_factor = np.linalg.norm([width_scale, height_scale]) / np.sqrt(2)

    return scale_factor, (width_scale, scale_factor)

def image_pad(img, shape):
        pad_h = shape[0] - img.shape[0]
        pad_w = shape[1] - img.shape[1]
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT), (top,left)

def mask_outside_rectangle(image, rect) -> np.ndarray:
    """
    将指定矩形范围外的像素设为黑色
    :param image: 单通道图像
    :param rect: 矩形 (x, y, w, h)
    :return: 处理后的图像
    """
    mask = np.zeros_like(image)
    x, y, w, h = rect
    mask[x:x+h, y:y+w] = 255  # 在矩形区域内设为白色
    result = cv2.bitwise_and(image, mask)
    return result

def draw_points(image: np.ndarray, points: np.ndarray, radius: int = 3, color: tuple = (0, 255, 0), thickness: int = -1) -> np.ndarray:
    """
    在图像上批量绘制点

    :param image: 输入图像（numpy 数组）
    :param points: 点的坐标数组，形状为 (N, 2)，每个点的格式为 (x, y)
    :param radius: 绘制点的半径，默认为 3
    :param color: 绘制点的颜色，默认为绿色 (B, G, R) 格式
    :param thickness: 点的线宽，-1 表示填充（实心圆）
    :return: 绘制完点的图像
    """
    # 遍历所有点，并在图像上绘制
    for pt in points:
        cv2.circle(image, tuple(pt), radius, color, thickness)
    return image


def get_corner(x,y,h,w):
    a = h/2; b = w/2
    return [[x-a,y-b],[x-a,y+b],[x+a,y+b],[x+a,y-b]]
    
def crop_single_patch(image:np.ndarray, corners):
    H, W = image.shape[:2]
    if len(image.shape) == 2:
        image_channels = 1
    else:
        image_channels = image.shape[2]

    x_min = int(np.floor(min(corners[:, 0])))
    x_max = int(np.ceil(max(corners[:, 0])))
    y_min = int(np.floor(min(corners[:, 1])))
    y_max = int(np.ceil(max(corners[:, 1])))

    pad_left = max(0, -y_min)
    pad_right = max(0, y_max - W)
    pad_top = max(0, -x_min)
    pad_bottom = max(0, x_max - H)

    x_min, x_max = max(0, x_min), min(H, x_max)
    y_min, y_max = max(0, y_min), min(W, y_max)

    white_value = 255 if image_channels == 1 else [255] * image_channels
    patch_filled = np.full(
        (
            x_max - x_min + pad_top + pad_bottom,
            y_max - y_min + pad_left + pad_right,
            image_channels
        ),
        white_value, dtype=image.dtype
    )
    if x_max > x_min and y_max > y_min:
        patch = image[x_min:x_max, y_min:y_max]

        patch_filled[
            pad_top:pad_top + patch.shape[0], 
            pad_left:pad_left + patch.shape[1]
        ] = patch

    return patch_filled

def get_outside_indices(shape, dx, dy, num_row, num_col):
    rows, cols = shape
    row_grid, col_grid = np.ogrid[0:rows, 0:cols]
    
    inside_row = (row_grid >= dx) & (row_grid < dx + num_row)
    inside_col = (col_grid >= dy) & (col_grid < dy + num_col)
    inside_mask = inside_row & inside_col
    
    outside_mask = ~inside_mask
    outside_rows, outside_cols = np.where(outside_mask)
    return list(zip(outside_rows, outside_cols))
    
def reconstruct_image(patch_array):
    # 获取 patch_array 的维度信息
    if patch_array.ndim == 4:
        # 单通道，形状为 (rows, cols, ph, pw)
        rows, cols, ph, pw = patch_array.shape
        channels = 1
    elif patch_array.ndim == 5:
        # 多通道，形状为 (rows, cols, ph, pw, channels)
        rows, cols, ph, pw, channels = patch_array.shape
    else:
        raise ValueError("Unexpected patch_array shape")
    
    # 创建空白的大图
    if channels > 1:
        reconstructed = np.zeros((rows * ph, cols * pw, channels), dtype=patch_array.dtype)
    else:
        reconstructed = np.zeros((rows * ph, cols * pw), dtype=patch_array.dtype)
    
    # 遍历每个 patch 并填充到大图中
    for i in range(rows):
        for j in range(cols):
            if channels > 1:
                patch = patch_array[i, j]
            else:
                patch = patch_array[i, j]
            # 计算当前位置在大图中的坐标
            x_start = i * ph
            x_end = x_start + ph
            y_start = j * pw
            y_end = y_start + pw
            # 将 patch 放入对应位置
            reconstructed[x_start:x_end, y_start:y_end] = patch
    
    return reconstructed.squeeze()  # 去除单通道情况下的多余维度