
import argparse, os, math, json
from pathlib import Path
from typing import List, Dict,Tuple
from itertools import product
import warnings 

import pandas as pd
import numpy as np
import scanpy as sc
import cv2 as cv
from PIL import Image
import tifffile
import imageio.v2 as ii


class VisiumProfile:
    def __init__(self):
        self.spot_diameter = 55.0
        self.spot_step = 100.0
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
            bias = i%2 * self.spot_step / 2
            for j in range(self.row_range):
                y = radius + j*self.spot_step + bias
                x = radius + i*self.spot_step*math.sqrt(3)/2
                yield n, x, y, radius
                n += 1
    

class VisiumHDProfile:
    def __init__(self):
        self.bin_size = 2.0
        self.row_range = 3350
        self.col_range = 3350
    
    @property
    def frame(self):
        w = self.bin_size * self.row_range
        h = self.bin_size * self.col_range
        return w, h
    
    @property
    def bins(self):
        n = 0
        r = self.bin_size/2
        for i in range(self.col_range):
            for j in range(self.row_range):
                x = i*self.bin_size + r
                y = j*self.bin_size + r
                yield n, x, y, r
                n += 1
    
    def reset(self, bin_size:int):
        '''Set bin size
        The suggested bin size is 2*n
        '''
        self.row_range = int(self.row_range*self.bin_size/bin_size)
        self.col_range = int(self.col_range*self.bin_size/bin_size)
        self.bin_size = bin_size
    
    def __get_frame_center(self, frame, mode, args:Dict):
        w, h = frame
        W, H = self.frame
        if mode == "corner":
            corner = args.get("corner", None)
            if corner in (0,1,2,3):
                if corner == 0: return h/2, w/2
                if corner == 1: return h/2, W-w/2
                if corner == 2: return H-h/2, W-w/2
                if corner == 3: return H-h/2, w/2
            else:
                raise ValueError("supported corner in 0,1,2,3")
        elif mode == "manual":
            center = args.get("center", None)
            if isinstance(center, (tuple, list)) and len(center)==2:
                return center
            else:
                raise ValueError("")
        elif mode == "center":
            return W/2, H/2
        elif mode == "adaptive":
            y = W/2 if w<W else w/2
            x = H/2 if h<H else h/2
            return x,y
        else:
            raise ValueError("unsupported mode")
    
    def set_spots(self, profile:VisiumProfile, mode="adaptive", **args) -> np.ndarray:
        '''
        mode: 
        - center
        - corner
        - adaptive
        - manual
        '''
        center = self.__get_frame_center(profile.frame, mode, args)
        spot_label = np.zeros((self.row_range,self.col_range))
        x0 = center[0]-profile.frame[1]/2
        y0 = center[1]-profile.frame[0]/2

        bin_r = self.bin_size/2
        bin_iter = lambda a: range(int((a-r)/self.bin_size),int((a+r)/self.bin_size)+2)
        d2 = lambda x,y,a,b: (x-a)*(x-a) + (y-b)*(y-b)
        for id, x, y, r in profile.spots:
            x+=x0; y+=y0
            uncovered = 0
            # label bin in spot
            for i, j in product(bin_iter(x), bin_iter(y)):
                bin_x = i*self.bin_size + bin_r
                bin_y = j*self.bin_size + bin_r
                if d2(bin_x,bin_y,x,y) < r*r:
                    if i<0 or j<0 or i>self.col_range-1 or j>self.row_range-1:
                        uncovered += 1
                        continue
                    spot_label[i,j] = id + 1
            if uncovered:
                warnings.warn(f"spot {id} cover {uncovered} empty bins")
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

def vision_label(image):
    image[image>0]=255
    ii.imwrite("label.png",image.astype(np.uint8))

def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--prefix', type=str)
    # parser.add_argument('--rawdata', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    # prefix = Path(args.prefix)
    # rawdata = Path(args.rawdata)
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
    lable = a.set_spots(profile=profile)
    # image = a.set_spots(profile=profile, mode="corner", corner=3)
    # image = a.set_spots(profile=profile, mode="manual", center=(3000,2000))
    vision_label(lable)

if __name__ == "__main__":
    main()