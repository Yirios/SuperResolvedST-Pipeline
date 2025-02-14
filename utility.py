
import time
from functools import wraps
from pathlib import Path
from hashlib import md5
from typing import List, Dict,Tuple

import numpy as np
import pandas as pd
import cv2
import h5py
from anndata import AnnData

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
        # set feature_type
        if 'feature_type' in adata.var.columns:
            ftrs.create_dataset("feature_type", data=np.array(adata.var['feature_type'], dtype=f'S{str_max(adata.var["feature_type"])}'))
        else:
            adata.var['feature_type'] = 'Gene Expression'
            ftrs.create_dataset("feature_type", data=np.array(adata.var['feature_type'], dtype=f'S{str_max(adata.var["feature_type"])}'))

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

    return scale_factor

def image_pad(img, shape):
        pad_h = shape[0] - img.shape[0]
        pad_w = shape[1] - img.shape[1]
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT), (top,left)

def mask_outside_rectangle(image, rect):
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

class PerspectiveTransformer:
    def __init__(self, image: np.ndarray, corners: np.ndarray):
        """
        初始化透视变换类

        :param image: 原始图像 (numpy 数组)
        :param corners: 四个顶点坐标，形状为 (4, 2)，顺序为左上、右上、右下、左下
        """
        self.image = image
        self.corners = corners.astype("float32")
        self.warped_image = None
        self.M = None       # 正向透视变换矩阵
        self.M_inv = None   # 逆透视变换矩阵
        if len(self.image.shape) == 2:
            self.channels = 1
        else:
            self.channels = self.image.shape[2]

    def crop_image(self):
        """
        利用给定的四个顶点裁剪并校正图像，
        同时计算正向和逆向透视变换矩阵。

        :return: 裁剪后的图像和正向透视变换矩阵 M
        """
        # 计算目标矩形的宽度：取左下到右下和左上到右上的距离的最大值
        widthA = np.linalg.norm(self.corners[2] - self.corners[3])
        widthB = np.linalg.norm(self.corners[1] - self.corners[0])
        maxWidth = int(max(widthA, widthB))

        # 计算目标矩形的高度：取右上到右下和左上到左下的距离的最大值
        heightA = np.linalg.norm(self.corners[1] - self.corners[2])
        heightB = np.linalg.norm(self.corners[0] - self.corners[3])
        maxHeight = int(max(heightA, heightB))

        # 定义目标矩形的四个顶点
        target_corner = np.array([
            [0, 0],                         # 左上角
            [0, maxHeight - 1],             # 右上角
            [maxWidth - 1, maxHeight - 1],  # 右下角
            [maxWidth - 1, 0]               # 右下角
        ], dtype="float32")

        # 计算正向透视变换矩阵 M
        self.M = cv2.getPerspectiveTransform(self.corners, target_corner)
        # 利用 M 进行透视变换，得到裁剪并校正后的图像
        if self.channels == 1:
            borderValue = 0
        else:
            borderValue = tuple([255] * self.image.shape[2])
        self.warped_image = cv2.warpPerspective(self.image, self.M,
                                                (maxWidth, maxHeight),
                                                borderValue=borderValue)
        # 计算逆向透视变换矩阵，用于将 warped 图像中的点映射回原图
        self.M_inv = np.linalg.inv(self.M)
        return self.warped_image, self.M

    def reverse_map_points(self, points: np.ndarray) -> np.ndarray:
        """
        批量将 warped 图像中的点反向映射回原图像

        :param points: numpy 数组，形状为 (N, 2)，每行表示一个点 (x, y)
        :return: 映射回原图像后的点的 numpy 数组，形状为 (N, 2)
        """
        if self.M_inv is None:
            raise ValueError("请先调用 crop_image 方法计算透视变换矩阵。")

        # 将点转换为齐次坐标 (x, y, 1)
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack([points, ones])
        
        # 利用逆矩阵映射
        mapped_points = homogeneous_points.dot(self.M_inv.T)
        
        # 归一化（除以第三个分量）
        mapped_points /= mapped_points[:, 2][:, np.newaxis]
        return mapped_points[:, :2]
    
    def map_points(self, points: np.ndarray) -> np.ndarray:
        """
        将原图中的多个点映射到透视变换后的图像中

        :param points: numpy 数组，形状为 (N, 2)，每行表示一个点 (x, y)
        :return: 映射后的点的 numpy 数组，形状为 (N, 2)
        """
        # 将每个点扩展为齐次坐标 (x, y, 1)
        ones = np.ones((points.shape[0], 1))
        homogeneous_points = np.hstack((points, ones))  # 形状 (N, 3)
        
        # 使用矩阵乘法进行映射（注意使用 M 的转置）
        mapped_points = homogeneous_points.dot(self.M.T)  # 形状 (N, 3)
        
        # 对每个点归一化（除以第三个分量）
        mapped_points /= mapped_points[:, 2, np.newaxis]
        
        # 返回前两列，即映射后的 (x, y)
        return mapped_points[:, :2]