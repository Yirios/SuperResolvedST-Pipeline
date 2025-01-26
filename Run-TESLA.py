
import warnings
warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
import scanpy as sc
import cv2
import TESLA as tesla
from anndata import AnnData

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    prefix = Path(args.prefix)

    with open(prefix/"pixel_step.txt","r") as f:
        res=int(f.read())

    counts=sc.read(prefix/"data.h5ad")
    img=cv2.imread(prefix/"image.jpg")
    mask=cv2.imread(prefix/"mask.png")

    shape = [
        int(np.floor((img.shape[0]-res)/res)+1),
        int(np.floor((img.shape[1]-res)/res)+1)
        ]
    print(f"super shape {shape}")
    
    counts.var_names_make_unique()
    counts.raw=counts
    sc.pp.log1p(counts) # impute on log scale
    if issparse(counts.X):counts.X=counts.X.A

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    adata:AnnData = tesla.imputation(img=img, raw=counts, cnt=cnts[0], genes=counts.var.index.tolist(), shape="None", res=res, s=1, k=2, num_nbs=10)
    adata.X = np.expm1(adata.X)
    adata.obs["x_spuer"] = adata.obs["x"]/res
    adata.obs["y_spuer"] = adata.obs["y"]/res
    adata.uns["shape"] = shape
    adata.write_h5ad(prefix/"enhanced_exp.h5ad")

if __name__ == "__main__":
    main()