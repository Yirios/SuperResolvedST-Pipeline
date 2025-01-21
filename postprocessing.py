
import argparse
import pickle
import time
from pathlib import Path
from typing import List, Dict,Tuple
from functools import wraps
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch
import cv2 as cv
import h5py
from PIL import Image

import imageio.v2 as ii
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


class SRresult:
    def __init__(self, prefix:Path):
        self.prefix = prefix
        self.image_shape = (-1,-1)

    @timer
    def load_xfuse(self):

        def find_min_bbox( mask: np.ndarray) -> Tuple[float, float]:
            """"Finds the mininum bounding box enclosing a given image mask
            copy from xfuse/convert/utility.py"""
            contour, _ = cv.findContours(
                mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            x, y, _, _ = cv.boundingRect(np.concatenate(contour))
            return (x,y)
        
        # read scale factor
        with open(self.prefix/"scale.txt") as f:
            scale = float(f.read())
        # rebuild mask
        mask = Image.open(self.prefix/"mask.png")
        mask = mask.resize([round(x * scale) for x in mask.size], resample=Image.NEAREST)
        mask = np.array(mask)[1:-1, 1:-1]
        rect = find_min_bbox(mask)
        
        with h5py.File(self.prefix/"data/data.h5", "r") as f:
            buildin_mask = np.where(f["label"][:]==1,1,0).astype("int8")
        
        mask = np.pad(buildin_mask,
                      pad_width=(
                          (rect[1],mask.shape[0]-buildin_mask.shape[0]-rect[1]),
                          (rect[0],mask.shape[1]-buildin_mask.shape[1]-rect[0])
                      ),
                      mode='constant', constant_values=1)
        # select genes
        with open(self.prefix/'gene-names.txt', 'r') as file:
            genes = [line.rstrip() for line in file]
        # select unmasked super pixel 
        Xs,Ys = np.where(mask==0)
        self.image_shape = mask.shape[:2]
        data = {"x":Xs, "y":Ys}
        # genes = ["HES4","VWA1","AL645728.1","GABRD"] # test genes
        for gene in genes:
            cnts = torch.load(self.prefix/f"result/analyses/final/gene_maps/section1/{gene}.pt")
            cnts = np.mean(cnts,axis=0)
            data[gene]=[float(f"{x:.8f}") for x in np.round(cnts[Xs, Ys], decimals=8)]
        self.data = pd.DataFrame(data)
        # print(self.data)
        
    @timer
    def load_istar(self):
        mask = ii.imread(self.prefix/'mask-small.png') > 0
        # select genes
        with open(self.prefix/'gene-names.txt', 'r') as file:
            genes = [line.rstrip() for line in file]
        # select unmasked super pixel 
        Xs,Ys = np.where(mask)
        self.image_shape = mask.shape[:2]
        data = {"x":Xs, "y":Ys}
        for gene in genes:
            with open(self.prefix/f'cnts-super/{gene}.pickle', 'rb') as file:
                cnts = pickle.load(file)
            data[gene]=[float(f"{x:.8f}") for x in np.round(cnts[Xs, Ys], decimals=8)]
        self.data = pd.DataFrame(data)

    @timer
    def load_TESLA(self):
        pass

    @timer
    def to_csv(self, file=None, sep="\t"):
        if not file:
            file = self.prefix/"super-resolution.csv"
        with open(file, "w") as f:
            header = self.data.columns.to_list()
            header[0] = f"x:{self.image_shape[0]}"
            header[1] = f"y:{self.image_shape[1]}"
            f.write(sep.join(header) + "\n")
            for _, row in self.data.iterrows():
                f.write(sep.join(map(str, row)) + "\n")

    @timer
    def to_h5ad(self):
        adata = AnnData(self.data.iloc[:,2:])
        adata.obs = self.data.iloc[:, :2]
        adata.var.index = self.data.columns[2:]
        adata.uns["shape"] = list(self.image_shape)
        adata.uns["project_dir"] = str(self.prefix.resolve())
        adata.write_h5ad(self.prefix/"super-resolution.h5ad")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    prefix = Path(args.prefix)
    result = SRresult(prefix)
    # result.load_istar()
    result.load_xfuse()
    # result.to_csv()
    # result.to_h5ad()

if __name__ == "__main__":
    main()