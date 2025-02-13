
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
        return np.mean((s_x,s_y))

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
    return resized_image