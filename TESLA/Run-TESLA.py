
import warnings
warnings.filterwarnings('ignore')
import argparse
from pathlib import Path
import json

import numpy as np
from scipy.sparse import issparse
import scanpy as sc
import cv2
from TESLA.imputation import imputation, imputation_sudo, sum_patch
from anndata import AnnData
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    args = parser.parse_args()
    return args

def load_npz(file:Path):
    with np.load(file, allow_pickle=True) as loaded:
        dictf = {key: loaded[key].copy() for key in loaded.files}
    return dictf

def main():
    args = get_args()
    prefix = Path(args.prefix)

    with open(prefix/"super_resolution_config.json", "r", encoding="utf-8") as f:
        config =  json.load(f)
    top,left,height,width = config["capture_area"]

    counts = sc.read(prefix/"data.h5ad")
    counts = counts[counts.obs["in_tissue"]==1]
    counts.var_names_make_unique()
    counts.raw=counts
    sc.pp.log1p(counts) # impute on log scale
    if issparse(counts.X):
        counts.X=counts.X.toarray()
    
    if config["mode"] == "VisiumHD":

        spot_patchs = load_npz(prefix/"spot_image.npz")
        # bin_patchs = load_npz(prefix/"bin_image.npz")
        bin_patch_array = np.load(prefix/"bin_image.npy")
        bin_positions = pd.read_pickle(prefix/"bin_positions.pkl")
        bin_positions = bin_positions[bin_positions["in_tissue"]==1]
        sudo:pd.DataFrame = bin_positions.loc[
            bin_positions["in_tissue"]==1,
            ["array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]
        ].copy()
        sudo.columns = ["i","j","x","y"]
        i_indices = sudo["i"].astype(int).values + top
        j_indices = sudo["j"].astype(int).values + left
        bin_patchs = bin_patch_array[i_indices, j_indices]
        sudo["color"] = sum_patch(bin_patchs)
        z_scale=np.max([np.std(sudo["x"]), np.std(sudo["y"])])
        sudo["z"]=(sudo["color"]-np.mean(sudo["color"]))/np.std(sudo["color"])*z_scale
        counts.obs["color"] = sum_patch(
            neighborhoods=spot_patchs,
            index = counts.obs.index, 
            RGB=False
        )
        counts.obs["z"]=(counts.obs["color"]-np.mean(counts.obs["color"]))/np.std(counts.obs["color"])*z_scale
        counts.obs.rename(columns={"pixel_x": "x", "pixel_y": "y"}, inplace=True)
        adata:AnnData = imputation_sudo(
            sudo=sudo,
            known_adata=counts,
            k=2,
            num_nbs=3
            # num_nbs=10
        )
        adata.obs.rename(columns={"i":"x_super","j":"y_super"}, inplace=True)
        adata.obs["x_super"] += top
        adata.obs["y_super"] += left
        adata.uns["shape"] = [height, width]

    elif config["mode"] == "Image":
        with open(prefix/"pixel_step.txt","r") as f:
            res=int(f.read())

        counts=sc.read(prefix/"data.h5ad")
        img=cv2.imread(prefix/"image.jpg")
        mask=cv2.imread(prefix/"mask.png")

        shape = [
            int(np.floor((img.shape[0]-res)/res)+1),
            int(np.floor((img.shape[1]-res)/res)+1)
            ]

        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cnts, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        adata:AnnData = imputation(
            img = img,
            raw = counts,
            cnt=cnts[0], 
            genes=counts.var.index.tolist(), 
            shape="None", 
            res=res, 
            s=1, k=2, num_nbs=10
        )
        adata.obs["x_super"] = adata.obs["x"]/res
        adata.obs["y_super"] = adata.obs["y"]/res
        adata.uns["shape"] = shape
    
    adata.X = np.expm1(adata.X)
    adata.write_h5ad(prefix/"enhanced_exp.h5ad")

if __name__ == "__main__":
    main()