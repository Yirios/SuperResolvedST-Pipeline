
import argparse
import json
from pathlib import Path
from typing import List, Dict
from itertools import product
import shutil

import numpy as np
import cv2
import pandas as pd
import h5py
import scanpy as sc
import imageio.v2 as ii
from anndata import AnnData

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Variable names are not unique.*")

    
def fast_to_csv(df: pd.DataFrame, file: Path, sep="\t") -> None:
    with open(file, "w") as f:
        f.write(sep.join(df.columns) + "\n")
        for _, row in df.iterrows():
            f.write(sep.join(map(str, row)) + "\n")

def write_10X_h5(adata, file:Path) -> None:
    """Writes an AnnData object to a 10X Genomics formatted HDF5 file.
    
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
        grp = w.create_group("matrix")
        #
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

class rawData:
    """
    """
    image_extensions = {'.jpg', '.png', '.tiff'}
    def __init__(self,path:Path, pixel_size=8, auto_mask=True):
        self.path = path
        self.prefix = None
        self.pixel_size=pixel_size
        self.__read_feature_bc_matrix()
        self.__read_scalefactors()
        self.__read_location()
        self.__read_raw_image()
        self.__read_mask(auto_mask)

    def __read_raw_image(self):
        self.images = [
            ii.imread(f) for f in self.path.iterdir() 
            if f.is_file() and f.suffix.lower() in rawData.image_extensions and 'mask' not in f.name.lower()
        ]
        if len(self.images) == 0:
            raise ValueError("Can't find image")
        self.images.sort(key=lambda i:np.sum(i.shape), reverse=True)
        
    def __read_location(self) -> pd.DataFrame:
        file = self.path/"spatial/tissue_positions_list.csv"
        self.locDF = pd.read_csv(file,header=None)
        return self.locDF

    def __read_scalefactors(self) -> Dict:
        file = self.path/"spatial/scalefactors_json.json"
        with open(file) as f:
            self.scaleF = json.load(f)
        return self.scaleF

    def __read_feature_bc_matrix(self) -> AnnData:
        file = self.path/"filtered_feature_bc_matrix.h5"
        self.adata = sc.read_10x_h5(file)
        self.adata.var_names_make_unique()
        sc.pp.filter_genes(self.adata, min_counts=10)

    def __read_mask(self, auto_mask=True) -> np.ndarray:
        self.masks = [
            ii.imread(f) for f in self.path.iterdir() 
            if f.is_file() and f.suffix.lower() in rawData.image_extensions and 'mask' in f.name.lower()
        ]
        self.masks.sort(key=lambda i:np.sum(i.shape), reverse=True)
        # find match image
        self.mask2image=[]
        if len(self.masks) != 0:
            for (i,image),(j,mask) in product(enumerate(self.images),enumerate(self.masks)):
                if mask.shape[0] == image.shape[0] and mask.shape[1] == image.shape[1]:
                    self.mask2image.append((j,i))
        elif auto_mask:
            print("Can't find mask image, auto masking.")
            self.masks.append(self.auto_mask(self.images[0]))
            self.mask2image.append((0,0))
        else:
            self.masks = []
        return self.masks

    def auto_mask(self, img : np.ndarray,
                  CANNY_THRESH_1 = 100, CANNY_THRESH_2 = 200, apertureSize=5, L2gradient = True) -> np.ndarray:
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
        return cnt

    def select_HVG(self,n_top_genes=2000) -> None:
        sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes, subset=True, flavor='seurat_v3')

    def require_genes(self,genes:List[str]) -> None:

        genes = [gene for gene in genes if gene in self.adata.var_names]

        if genes:
            self.adata = self.adata[:,genes]
        else:
            print("No genes from the list are found in the data.")

    def convert(self):
        if self.prefix:
            shutil.copytree(self.path, self.prefix)

    def save(self, prefix:Path):
        prefix.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        print("Start convert")
        # write selected gene names
        if (self.path/"gene-names.txt").exists():
            shutil.copy(self.path/"gene-names.txt",self.prefix/"gene-names.txt")
        else:
            with open(self.prefix/"gene-names.txt","w") as f:
                f.write("\n".join(self.adata.var.index.values))
        self.convert()
        print("Finish convert")

class XfuseData(rawData):

    def convert(self):
        # save image.png
        ii.imsave(self.prefix/"image.png", self.images[self.mask2image[0][1]])
        # save mask.png
        mask = self.masks[self.mask2image[0][0]] > 0
        mask = np.where(mask, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
        ii.imsave(self.prefix/"mask.png", mask)
        # save h5
        write_10X_h5(self.adata, self.prefix/"filtered_feature_bc_matrix.h5")
        # copy tissue_positions_list.csv
        shutil.copy(self.path/"spatial/tissue_positions_list.csv", self.prefix/"tissue_positions_list.csv")
        # copy scale-factors
        shutil.copy(self.path/"spatial/scalefactors_json.json", self.prefix/"scalefactors_json.json")
        # calculate scale 
        with open(self.prefix/"scale.txt","w") as f:
            f.write(str(self.scaleF["tissue_hires_scalef"]*4/self.pixel_size))
            # f.write(str(65*self.pixel_size/self.scaleF["spot_diameter_fullres"]))

class iStarData(rawData):

    def transfer_loc(self) -> pd.DataFrame:
        df = self.locDF.copy(True)
        df.columns = ["barcode","in_tissue","array_row","array_col","x","y"]
        df = df[df["in_tissue"]==1]
        del df["in_tissue"]
        df["spot"] = df["array_row"].astype(str) + "x" + df["array_col"].astype(str)
        return df
    
    def transfer_cnts(self,locDF:pd.DataFrame) -> pd.DataFrame:
        cntDF = pd.DataFrame(self.adata.X.toarray(), index=self.adata.obs_names, columns=self.adata.var_names)
        cntDF["barcode"] = self.adata.obs_names
        mergedDF = pd.merge(locDF,cntDF, left_on='barcode', right_on='barcode', how='inner')
        return mergedDF.iloc[:, 5:]

    def convert(self):
        # save he-raw.jpg
        ii.imsave(self.prefix/"he-raw.jpg", self.images[self.mask2image[0][1]])
        # save mask-raw.png
        mask = self.masks[self.mask2image[0][0]]
        ii.imsave(self.prefix/"mask-raw.png", mask)
        # wirte number of pixels per spot radius
        with open(self.prefix/"radius-raw.txt","w") as f:
            f.write(str(self.scaleF["spot_diameter_fullres"]/2))
        # write side length (in micrometers) of pixels
        with open(self.prefix/"pixel-size-raw.txt","w") as f:
            f.write(str(self.scaleF["tissue_hires_scalef"]*4))
            # f.write(str(65/scaleF["spot_diameter_fullres"]))
        # save spot locations
        locDF = self.transfer_loc()
        locDF[["spot","x","y"]].to_csv(self.prefix/"locs-raw.tsv", sep="\t", index=False)
        # save gene count matrix
        fast_to_csv(self.transfer_cnts(locDF),self.prefix/"cnts.tsv")

class soScopeData(rawData):
    pass

class TESLAData(rawData):

    def transfer_h5ad(self) -> AnnData:
        # select in tissue
        df = self.locDF.copy(True)
        df.columns = ["barcode","in_tissue","array_row","array_col","pixel_x","pixel_y"]
        df = df[df["in_tissue"]==1]
        df.index = df["barcode"]
        del df["in_tissue"], df["barcode"]
        # merge to h5d
        adata = self.adata.copy()
        adata.obs = df
        return adata

    def convert(self):
        # save image.jpg
        ii.imsave(self.prefix/"image.jpg", self.images[self.mask2image[0][1]])
        # save mask.png
        mask = self.masks[self.mask2image[0][0]]
        ii.imsave(self.prefix/"mask.png", mask)
        # save data.h5ad
        self.transfer_h5ad().write_h5ad(self.prefix/"data.h5ad")

class ImSpiRE(rawData):
    
    def convert(self):
        pass

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--rawdata', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    prefix = Path(args.prefix)
    rawdata = Path(args.rawdata)
    data = XfuseData(path=rawdata)
    # data = iStarData(path=rawdata)
    # data = TESLAData(path=rawdata)
    data.select_HVG(n_top_genes=2000)
    if (rawdata/"gene_names.txt").exists():
        genes = pd.read_csv(rawdata/"gene_names.txt", sep='\t', header=0)
        data.require_genes(genes[0].values.tolist())
    data.save(prefix/"xfuse")
    # data.save(prefix/"istar")
    # data.save(prefix/"TESLA")

if __name__ == "__main__":
    main()