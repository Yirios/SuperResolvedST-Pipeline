
import argparse, os, math, json
from pathlib import Path
from typing import List, Dict,Tuple

import pandas as pd
import numpy as np
import scanpy as sc
import cv2 as cv
from PIL import Image
import tifffile


def labels_from_spots(dst: np.ndarray, spots: List) -> None:
    r"""Fills `dst` with labels enumerated from `spots`"""
    for i, s in enumerate(spots, 1):
        x, y, radius = [int(round(x)) for x in (s.x, s.y, s.r)]
        dst[
            tuple(
                zip(
                    *(
                        (y - dy, x - dx)
                        for dy in range(-radius, radius + 1)
                        for dx in range(-radius, radius + 1)
                        if dy ** 2 + dx ** 2 <= s.r ** 2
                    )
                )
            )
        ] = i

class VisiumProfile:
    def __init__(self):
        self.spot_diameter = 55
        self.spot_step = 99
        self.row_range = 64     # even numbers from 0 to 126 for even rows, and odd numbers from 1 to 127 for odd rows with each row (even or odd) resulting in 64 spots.
        self.col_range = 78

    @property
    def frame(self):
        w = self.spot_step * (self.row_range + 0.5)
        h = self.spot_step * (self.col_range-1) * math.sqrt(3)/2 + self.spot_diameter
        return w, h
    
    @property
    def spots(self):
        n = 0
        radius = self.spot_diameter / 2
        for i in range(self.col_range):
            bias = int(i/2) * self.spot_step / 2
            for j in range(self.row_range):
                x = radius + j*self.spot_step + bias
                y = radius + i*self.spot_step*math.sqrt(3)/2
                yield n, x, y, radius
                n += 1
    

class VisiumHDProfile:
    def __init__(self):
        self.bin_size = 2
        self.row_range = 3350
        self.col_range = 3350
    
    @property
    def frame(self):
        w = self.bin_size * self.row_range
        h = self.bin_size * self.col_range
        return w, h
    
    def set_spots(self, profile:VisiumProfile, mode = "center"):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        spot_label = np.zeros((self.row_range,self.col_range))
        bias = 1
        for id, x, y, r in profile.spots:
            int(x - r)
            y - r
            
        return spot_label



class VisiumHD:
    def __init__(self, path:Path, bin_size=2):
        self.path = path
        self.bin_size = bin_size
        self.locDF = pd.read_parquet(path/"spatial/tissue_positions.parquet")
        self.adata = sc.read_10x_h5(path/f'binned_outputs/square_{bin_size:03}um/filtered_feature_bc_matrix.h5')
        self.adata.var_names_make_unique()
        with open(path/f'binned_outputs/square_{bin_size:03}um/spatial/scalefactors_json.json','r') as f:
            self.scaleF = json.load(f)
        # self.image_raw = tifffile.imread(path/'tissue_image.tif')


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--prefix', type=str)
    parser.add_argument('--rawdata', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    # prefix = Path(args.prefix)
    rawdata = Path(args.rawdata)
    # VisiumHD(rawdata)
    # image = tifffile.imread("/data/datasets/Visium_HD_Human_Tonsil_Fresh_Frozen/Visium_HD_Human_Tonsil_Fresh_Frozen_tissue_image.tif")
    # image = tifffile.imread("/data/datasets/Visium_HD_Mouse_Brain_Fresh_Frozen/Visium_HD_Mouse_Brain_Fresh_Frozen_tissue_image.tif")
    # pil_image = Image.fromarray(image)
    # scale_factor = 0.01  # 缩小比例
    # new_size = (int(pil_image.width * scale_factor), int(pil_image.height * scale_factor))
    # resized_image = pil_image.resize(new_size, Image.ANTIALIAS)
    # Resolution

    # 将 Pillow 图像转换回 NumPy 数组
    # resized_image = np.array(resized_image)

    # 保存缩小后的图像
    # tifffile.imwrite("resized_image.tif", resized_image)
    a = VisiumHDProfile()
    profile = VisiumProfile()
    print(profile.frame,a.frame)
    print(a.set_spots(profile=profile))

if __name__ == "__main__":
    main()