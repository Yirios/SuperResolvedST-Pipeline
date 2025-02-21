from pathlib import Path
import pickle
from typing import List, Dict, Tuple
from datetime import datetime
import shutil

import numpy as np
import cv2
import pandas as pd
import imageio.v2 as ii
from anndata import AnnData, read_h5ad
from scipy.sparse import csr_matrix
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utility import *
from profiles import Profile
from datasets import VisiumData, VisiumHDData

class SRtools(VisiumData):
    '''
    Image base: 
    VisiumHD base:
    '''
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.prefix:Path = None
        self.super_pixel_size:float = None
        self.HDData:VisiumHDData = None
        self.mask:np.ndarray = None

        self.SRresult:pd.DataFrame = None
        self.super_image_shape = [None, None]
        self.capture_area = [None, None, None, None]

    def set_super_pixel_size(self, super_pixel_size:float=8.0):
        self.super_pixel_size = super_pixel_size
    
    def set_target_VisiumHD(self, HDData:VisiumHDData):
        self.super_pixel_size = HDData.bin_size
        self.HDData = HDData
    
    def tissue_mask(self, mask:np.ndarray=None, mask_image_path:Path=None, auto_mask=False, **kwargs):
        
        if mask != None:
            pass
        elif mask_image_path != None:
            mask = ii.imread(mask_image_path)
        elif auto_mask:
            self.mask = auto_tissue_mask(self.image,**kwargs)
            return self.mask
        else:
            raise ValueError("Please provide a mask or set auto_mask=True to apply masking automatically.")
        
        if self.image.shape[:2] == mask.shape[:2]:
            self.mask = mask
        else:
            raise ValueError("The mask must have the same shape as the image.")
        return self.mask
    

    def convert(self):
        super().save(self, self.prefix)
        ii.imsave(self.prefix/"mask.png", self.mask)

    def save_params(self):
        parameters = {
            "mode": "VisiumHD" if self.HDData else "Image",
            "super_resolution_tool":type(self).__name__,
            "super_image_shape": list(self.super_image_shape),
            "super_pixel_size":self.super_pixel_size,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_dir": str(self.prefix.resolve())
        }
        if None not in self.capture_area:
            parameters["capture_area"] = self.capture_area
        else:
            parameters["capture_area"] = [0,0,*self.super_image_shape]
        write_json(self.prefix/"super_resolution_config.json",parameters)

    def load_params(self):
        parameters = read_json(self.prefix/"super_resolution_config.json")
        self.super_pixel_size = parameters["super_pixel_size"]
        self.super_image_shape = parameters["super_image_shape"]
        self.capture_area = parameters["capture_area"]

    def save_input(self, prefix:Path):
        prefix = Path(prefix)
        prefix.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        # write selected gene names
        with open(self.prefix/"gene-names.txt","w") as f:
            f.write("\n".join(self.adata.var.index.values))
        self.convert()
        self.save_params()
    
    def load_output(self, prefix:Path=None):
        '''
        if self.prefix exist, will be recovered 
        '''
        if not (self.prefix or prefix):
            raise ValueError("Run save_inpout frist or set prefix")
        else: 
            self.prefix = Path(prefix)
        self.load_params()
    
    @timer
    def to_VisiumHD(self, HDprefix:Path, superHD_demo:VisiumHDData=None):
        if not (self.HDData or superHD_demo ):
            raise ValueError("Run set_target_VisiumHD frist or set superHD_demo")
        elif superHD_demo:
            self.HDData = superHD_demo
        
        merged = self.HDData.locDF.merge(self.SRresult, left_on=['array_row', 'array_col'], right_on=['x', 'y'])
        self.HDData.locDF.loc[self.HDData.locDF.index.isin(merged.index), 'in_tissue'] = 1

        genes = self.SRresult.columns[2:]
        self.SRresult = merged.set_index('barcode')[genes]

        self.HDData.adata = AnnData(
            X=csr_matrix(self.SRresult.to_numpy()),
            obs=pd.DataFrame(index=self.SRresult.index),
            var=self.HDData.adata.var.loc[genes,:])
        self.HDData.save(HDprefix)

    @timer
    def to_csv(self, file=None, sep="\t"):
        if not file:
            file = self.prefix/"super-resolution.csv"
        with open(file, "w") as f:
            header = self.SRresult.columns.to_list()
            header[0] = f"x:{self.super_image_shape[0]}"
            header[1] = f"y:{self.super_image_shape[1]}"
            f.write(sep.join(header) + "\n")
            for _, row in self.SRresult.iterrows():
                f.write(sep.join(map(str, row)) + "\n")
    
    @timer
    def to_h5ad(self):
        adata = AnnData(self.SRresult.iloc[:,2:])
        adata.obs = self.SRresult.iloc[:, :2]
        adata.var.index = self.SRresult.columns[2:]
        adata.uns["shape"] = list(self.super_image_shape)
        adata.uns["project_dir"] = str(self.prefix.resolve())
        adata.write_h5ad(self.prefix/"super-resolution.h5ad")

class Xfuse(SRtools):

    def convert(self):
        # save image.png
        ii.imsave(self.prefix/"image.png", self.image)
        # save mask.png
        mask = self.mask > 0
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
            f.write(str(self.pixel_size/self.super_pixel_size))
    
    def load_output(self, prefix:Path=None):
        super().load_output(prefix)

class iStar(SRtools):
    
    def transfer_cnts(self,locDF:pd.DataFrame) -> pd.DataFrame:
        cntDF = pd.DataFrame(self.adata.X.toarray(), index=self.adata.obs_names, columns=self.adata.var_names)
        cntDF["barcode"] = self.adata.obs_names
        mergedDF = pd.merge(locDF,cntDF, left_on='barcode', right_on='barcode', how='inner')
        return mergedDF.iloc[:, 5:]

    def transfer_loc_base(self, scaleF) -> pd.DataFrame:
        df = self.locDF.copy(True)
        df.columns = ["barcode","in_tissue","array_row","array_col","y","x"]
        df = df[df["in_tissue"]==1]
        del df["in_tissue"]
        df["spot"] = df["array_row"].astype(str) + "x" + df["array_col"].astype(str)
        df.loc[:, ["y", "x"]] = (df[["y", "x"]].values * scaleF).astype(int)
        return df

    def transfer_image_base(self, img:np.ndarray):
        scalef = 16*self.pixel_size/self.super_pixel_size
        img = image_resize(img, scalef=scalef)
        H256 = (img.shape[0] + 255) // 256 * 256
        W256 = (img.shape[1] + 255) // 256 * 256
        img, _ = image_pad(img, (H256,W256))
        return img, scalef

    def transfer_mask_base(self, img:np.ndarray):
        img, _ = self.transfer_image_base(img)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        return img
    
    def transfer_image_HD(self, img:np.ndarray):
        num_row = self.HDData.profile.row_range
        num_col = self.HDData.profile.col_range
        H16 = (num_row + 15) // 16 * 16
        W16 = (num_col + 15) // 16 * 16
        HDdx = (H16-num_row)//2
        HDdy = (W16-num_col)//2
        Hframe = H16*self.HDData.bin_size; Wframe = W16*self.HDData.bin_size
        corner = pd.DataFrame(
            {
                "id":[0,1,2,3],
                "array_row":[0,0,1,1],
                "array_col":[0,1,1,0],
                "frame_row":[0,0,Hframe,Hframe],
                "frame_col":[0,Wframe,Wframe,0]
            }
        )

        image_profile = Profile(corner,(Wframe, Hframe))
        transformer = self.HDData.get_HDlikeImage(image_profile, img,(HDdx*self.HDData.bin_size, HDdy*self.HDData.bin_size))
        HDlikeImage, _ = transformer.crop_image()
        H256 = H16*16
        W256 = W16*16
        img = image_resize(HDlikeImage, shape=(W256,H256))
        scalef, scalefyx = calculate_scale_factor(HDlikeImage, img)
        capture_area = (HDdx*16, HDdy*16, num_row*16, num_col*16)
        return img, scalef, scalefyx, transformer, capture_area
    
    def transfer_mask_HD(self, img):
        img, _, _, _, capture_area = self.transfer_image_HD(img)
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
        img = mask_outside_rectangle(img, rect=capture_area)
        return img

    def transfer_loc_HD(self, mapper:PerspectiveTransformer, scaleFyx) -> pd.DataFrame:
        df = self.locDF.copy(True)
        df.columns = ["barcode","in_tissue","array_row","array_col","y","x"]
        df = df[df["in_tissue"]==1]
        del df["in_tissue"]
        df["spot"] = df["array_row"].astype(str) + "x" + df["array_col"].astype(str)
        df = df.astype({"y": float, "x": float})
        df.loc[:, ["y", "x"]] = mapper.map_points(df[["y", "x"]].values) 
        df["x"] = np.round(df["x"] * scaleFyx[0]).astype(int)
        df["y"] = np.round(df["x"] * scaleFyx[0]).astype(int)
        return df

    def convert(self):
        if self.HDData == None:
            image, scaleF = self.transfer_image_base(self.image)
            mask = self.transfer_mask_base(self.mask)
            locDF = self.transfer_loc_base(scaleF)
            self.super_image_shape = [i//16 for i in mask.shape]
        else:
            image, scaleF, scaleFyx, crop_mapper, capture_area = self.transfer_image_HD(self.image)
            mask = self.transfer_mask_HD(self.mask)
            locDF = self.transfer_loc_HD(crop_mapper, scaleFyx)
            self.super_image_shape = [i//16 for i in mask.shape]
            self.capture_area = [i//16 for i in capture_area]
            
        ii.imsave(self.prefix/"he.jpg", image)
        # save mask.png
        ii.imsave(self.prefix/"mask.png", mask)
        # save spot locations
        locDF[["spot","x","y"]].to_csv(self.prefix/"locs.tsv", sep="\t", index=False)

        # wirte number of pixels per spot radius
        radius = self.scaleF["spot_diameter_fullres"]/2*scaleF
        pixel_size_raw = self.pixel_size*scaleF
        pixel_size = self.super_pixel_size/16
        with open(self.prefix/"radius.txt","w") as f:
            f.write(str(int(np.round(radius))))
        # write side length (in micrometers) of pixels
        with open(self.prefix/"pixel-size-raw.txt","w") as f:
            f.write(str(pixel_size_raw))
        with open(self.prefix/"pixel-size.txt", "w") as f:
            f.write(str(pixel_size))
        # save gene count matrix
        fast_to_csv(self.transfer_cnts(locDF),self.prefix/"cnts.tsv")

    def corp_capture_area(self, img):
        top,left,heigth,width = self.capture_area
        return img[top:top+heigth,left:left+width]

    def load_output(self, prefix:Path=None):
        super().load_output(prefix)
        mask = ii.imread(self.prefix/'mask.png')

        if mask.shape != self.super_image_shape:
            mask = image_resize(mask, shape=self.super_image_shape)
        mask = self.corp_capture_area(mask)
        mask = mask > 127

        # select unmasked super pixel 
        Xs,Ys = np.where(mask)
        data = {"x":Xs, "y":Ys}

        # select genes
        with open(self.prefix/'gene-names.txt', 'r') as file:
            genes = [line.rstrip() for line in file]
        gene_iter = progress_bar(
            title="Reading iStar output",
            iterable=genes,
            total=len(genes)
        )
        for gene in gene_iter():
            with open(self.prefix/f'cnts-super/{gene}.pickle', 'rb') as file:
                cnts = pickle.load(file)
            data[gene]=[x for x in np.round(cnts[Xs, Ys], decimals=8)]
            # data[gene]=[float(f"{x:.8f}") for x in np.round(cnts[Xs, Ys], decimals=8)]
        self.SRresult = pd.DataFrame(data)

class soScope(SRtools):
    pass

class TESLA(SRtools):

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
        ii.imsave(self.prefix/"image.jpg", self.image)
        # save mask.png
        mask = self.mask
        ii.imsave(self.prefix/"mask.png", mask,)
        # save data.h5ad
        self.transfer_h5ad().write_h5ad(self.prefix/"data.h5ad")
        # calculate super pixel step
        with open(self.prefix/"pixel_step.txt","w") as f:
            scale = self.scaleF["tissue_hires_scalef"]*4/self.pixel_size
            f.write(str(int(np.round(1/scale))))

    def load_output(self, prefix:Path=None):
        super().load_output(prefix)
        adata = read_h5ad(self.prefix/"enhanced_exp.h5ad")
        self.SRresult = adata.to_df()
        self.SRresult.insert(0, 'y', adata.obs["y_spuer"].astype(int))
        self.SRresult.insert(0, 'x', adata.obs["x_spuer"].astype(int))

class ImSpiRE(SRtools):
    
    def convert(self):
        # save image.tif
        ii.imsave(self.prefix/"image.tif", self.image)
        # save h5
        write_10X_h5(self.adata, self.prefix/"filtered_feature_bc_matrix.h5", self.metadata)
        # copy spatial folder
        shutil.copytree(self.path / "spatial", self.prefix / "spatial")
        # calculate patch size
        with open(self.prefix/"patch_size.txt","w") as f:
            scale = self.scaleF["tissue_hires_scalef"]*4/self.pixel_size
            f.write(str(int(np.round(1/scale))))

    def load_output(self, prefix:Path=None):
        super().load_output(prefix)
        adata = read_h5ad(self.prefix/"result/result_ResolutionEnhancementResult.h5ad")
        self.SRresult = adata.to_df()
        locDF = pd.read_csv(self.prefix/"result/result_PatchLocations.txt", sep="\t",)
        locDF.columns = ['index', 'row', 'col', 'pxl_row', 'pxl_col', 'in_tissue']
        self.SRresult.index = self.SRresult.index.astype(int)
        with open(self.prefix/'patch_size.txt') as f:
            patch_size = int(f.read())
        locDF["x"] = np.floor(locDF["pxl_row"]/patch_size).astype(int)
        locDF["y"] = np.floor(locDF["pxl_col"]/patch_size).astype(int)
        self.SRresult = pd.merge(locDF, self.SRresult, left_index=True, right_index=True).iloc[:, 6:]
