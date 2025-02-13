
import json
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import product
import shutil

import numpy as np
import cv2
import pandas as pd
import h5py
import scanpy as sc
import imageio.v2 as ii
import tifffile
from anndata import AnnData
from scipy.sparse import csr_matrix

from utility import *
from profiles import Profile, VisiumProfile, VisiumHDProfile

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Variable names are not unique.*")


class rawData:

    def __init__(self,**kwargs):

        if "path" in kwargs:
            self.load(kwargs["path"])
        if "tissue_positions" in kwargs:
            self.locDF:pd.DataFrame = kwargs["tissue_positions"]
        if "feature_bc_matrix" in kwargs:
            self.adata:AnnData = kwargs["feature_bc_matrix"]
        if "scalefactors" in kwargs:
            self.scaleF:Dict = kwargs["scalefactors"]
        if "metadata" in kwargs:
            self.metadata:Dict = kwargs["metadata"]

    def _read_location(self) -> pd.DataFrame:
        file = self.path/"spatial/tissue_positions.csv"
        self.locDF = pd.read_csv(file,header=None)
        self.locDF.columns = Profile.RawColumes
        return self.locDF

    def _read_scalefactors(self) -> Dict:
        file = self.path/"spatial/scalefactors_json.json"
        with open(file) as f:
            self.scaleF = json.load(f)
        return self.scaleF

    def _read_feature_bc_matrix(self) -> AnnData:
        file = self.path/"filtered_feature_bc_matrix.h5"
        self.adata = sc.read_10x_h5(file)
        # self.adata.var_names_make_unique()
        with h5py.File(file, mode="r") as f:
            self.metadata = dict(f.attrs)
        sc.pp.filter_genes(self.adata, min_counts=10)

    def read_image(self, file:Path):
        self.image = tifffile.imread(file)
    
    def load(self, path:Path, source_image_path:Path=None):
        self.path = Path(path)
        self._read_feature_bc_matrix()
        self._read_scalefactors()
        self._read_location()
        if source_image_path :
            self.read_image(Path(source_image_path))

    def match2profile(self, profile:Profile):
        self.profile = profile
        # keep order
        order = self.profile.tissue_positions[["array_row", "array_col"]].values.tolist()
        raw_order = self.locDF[["array_row", "array_col"]].values.tolist()
        if not np.array_equal(order, raw_order):
            temp = self.locDF.set_index(["array_row", "array_col"])
            self.locDF = temp.loc[order]
            self.locDF.reset_index(inplace=True)
            # TODO match adata
        
        # map to image
        PointsOnFrame = self.profile.tissue_positions[["frame_row","frame_col"]].values
        PointsOnImage = self.locDF[["pxl_row_in_fullres","pxl_col_in_fullres"]].values
        self.mapper = AffineTransform(PointsOnFrame, PointsOnImage)
        self.pixel_size = 1/self.mapper.resolution

    def select_HVG(self,n_top_genes=2000) -> None:
        sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes, subset=True, flavor='seurat_v3')

    def require_genes(self,genes:List[str]) -> None:

        genes = [gene for gene in genes if gene in self.adata.var_names]

        if genes:
            self.adata = self.adata[:,genes]
        else:
            warnings.warn("No genes from the list are found in the data.")

class VisiumData(rawData):

    def load(self, path:Path, profile=VisiumProfile(), source_image_path:Path=None):
        super().load(path, source_image_path)
        self.match2profile(profile)
        self.profile:VisiumProfile
    
    def tissue_mask(self, auto_mask=False, mask_path:Path=None, **kwargs):
        if auto_mask or mask_path == None:
            self.mask = auto_tissue_mask(self.image,**kwargs)
        else:
            self.mask = ii.imread(mask_path)
        return self.mask
    
    def convert(self):
        # feature_bc_matrix
        write_10X_h5(self.adata, self.prefix/'filtered_feature_bc_matrix.h5', self.metadata)
        # raw image
        tifffile.imwrite(self.prefix/"image.tif", self.image, bigtiff=True)
        # spatial output
        (self.prefix/"spatial").mkdir(parents=True, exist_ok=True)
        self.locDF.to_csv(self.prefix/"spatial/tissue_positions.csv", index=False, header=False)
        with open(self.prefix/"spatial/scalefactors_json.json", "w") as f:
            json.dump(self.scaleF, f, ensure_ascii=False, indent=4)
        lowres_image = image_resize(self.image, scalef=self.scaleF["tissue_lowres_scalef"])
        ii.imsave(self.prefix/"spatial/tissue_lowres_image.png", lowres_image)
        hires_image = image_resize(self.image, scalef=self.scaleF["tissue_hires_scalef"])
        ii.imsave(self.prefix/"spatial/tissue_hires_image.png", hires_image)

    def save(self, prefix:Path):
        prefix.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        print("Start convert")
        # write selected gene names
        with open(self.prefix/"gene-names.txt","w") as f:
            f.write("\n".join(self.adata.var.index.values))
        self.convert()
        print("Finish convert")

    def Image2VisiumHD(self, profile:VisiumHDProfile):

        _, frame_center = self.profile.set_spots(profile)
        return None

    def Visium2HD(self, HDporfile:VisiumHDProfile, SRmodel):
        HDlikeImage = self.Image2VisiumHD(HDporfile)

        metadata = self.profile.metadata.copy()
        metadata["chemistry_description"] = self.metadata["chemistry_description"]
        
        FullImage = np.max(self.image.shape)
        scaleF = {
            "spot_diameter_fullres": HDporfile.bin_size/self.pixel_size,
            "bin_size_um": HDporfile.bin_size,
            "microns_per_pixel": self.pixel_size,
            "tissue_lowres_scalef": HDporfile.LowresImage/FullImage,
            "fiducial_diameter_fullres": self.scaleF["fiducial_diameter_fullres"],
            "tissue_hires_scalef": HDporfile.HiresImage/FullImage,
            "regist_target_img_scalef": HDporfile.HiresImage/FullImage,
        }
        adata = AnnData()
        tissue_positions = HDporfile.tissue_positions[["barcode","array_row","array_col"]].copy()
        superHD_demo = VisiumHDData(
                tissue_positions = tissue_positions,
                feature_bc_matrix = adata,
                scalefactors = scaleF,
                metadata = metadata
            )
        return superHD_demo

class VisiumHDData(rawData):

    def _read_location(self) -> pd.DataFrame:
        file = self.path/"spatial/tissue_positions.parquet"
        self.locDF = pd.read_parquet(file)
        return self.locDF

    def load(self, path:Path, profile=VisiumHDProfile(), source_image_path:Path=None):
        super().load(path, source_image_path)
        
        self.bin_size = self.scaleF["bin_size_um"]
        self.pixel_size = self.scaleF["microns_per_pixel"]
        if self.bin_size != 2.0:
            warnings.warn(f"Using data in bin size of {self.bin_size}, recommed 2 um.")
            profile.reset(self.bin_size)
        
        self.match2profile(profile)
        self.profile:VisiumHDProfile

    def rebining(self, profile:VisiumHDProfile) -> "VisiumHDData":
        '''\
        TODO bin in user define profile
        '''
        pass

    def __spot2image(self, profile:VisiumProfile, frame_center:Tuple[float, float]):
        '''\
        add pixel_x, pixel_y in profile.tissue_positions
        '''
        x0 = frame_center[0]-self.profile.frame[1]/2
        y0 = frame_center[1]-self.profile.frame[0]/2
        spotOnFrame = profile.tissue_positions[["frame_row","frame_col"]].values + np.array([[x0,y0]])
        spotsOnImage = self.mapper.transform_batch(spotOnFrame)
        profile.tissue_positions[["pxl_row_in_fullres","pxl_col_in_fullres"]] = spotsOnImage

        return spotsOnImage
    
    def generate_Visium(self, profile:VisiumProfile, uncover_thresholds=0) -> VisiumData:

        _, frame_center = self.profile.set_spots(profile)
        self.__spot2image(profile, frame_center)
        
        X_indptr = [0]
        X_indices = np.zeros(0)
        X_data = np.zeros(0)
        spot_in_tissue = np.zeros(len(profile), dtype=int)
        
        mask_in_tissue = self.locDF["in_tissue"] == 1
        for id in range(len(profile)):
            bin_out = profile.tissue_positions.loc[id,"num_bin_out_spot"]
            bin_in = profile.tissue_positions.loc[id,"num_bin_in_spot"]
            if bin_out and bin_out/(bin_in+bin_out) > uncover_thresholds:
                spot_in_tissue[id] = 0
                continue

            mask_in_spot = self.profile.tissue_positions["spot_label"] == id + 1
            mask = mask_in_spot[mask_in_tissue].values
            if mask.any():
                bin_in_spot = self.adata.X[mask]
                spot_data = bin_in_spot.sum(axis=0).A1
                gene_index = np.where(spot_data>0)[0]
                X_indices = np.hstack((X_indices, gene_index))
                X_data = np.hstack((X_data, spot_data[gene_index]))
                X_indptr.append(X_indptr[-1]+len(gene_index))
                spot_in_tissue[id] = 1
            else:
                spot_in_tissue[id] = 0
            
            if id%100==0: print(id)
        
        tissue_positions = profile.tissue_positions[["barcode","array_row","array_col"]].copy()
        tissue_positions["pxl_row_in_fullres"] = np.round(profile.tissue_positions["pxl_row_in_fullres"].values).astype(int)
        tissue_positions["pxl_col_in_fullres"] = np.round(profile.tissue_positions["pxl_col_in_fullres"].values).astype(int)
        tissue_positions["in_tissue"] = spot_in_tissue
        tissue_positions = tissue_positions[profile.RawColumes]
        
        X_sparse = csr_matrix((X_data, X_indices, X_indptr), shape=(np.sum(spot_in_tissue), len(self.adata.var)))
        mask_in_tissue = spot_in_tissue == 1
        adata = AnnData(
            X=X_sparse,
            var=self.adata.var,
            obs=pd.DataFrame(index=tissue_positions.loc[mask_in_tissue,"barcode"].values)
        )

        metadata = self.profile.metadata.copy()
        metadata["chemistry_description"] = self.metadata["chemistry_description"]

        FullImage = np.max(self.image.shape)
        scaleF = {
            "spot_diameter_fullres": (profile.spot_diameter+10)/self.pixel_size,
            "tissue_lowres_scalef": profile.LowresImage/FullImage,
            "fiducial_diameter_fullres": self.scaleF["fiducial_diameter_fullres"],
            "tissue_hires_scalef": profile.HiresImage[profile.serial]/FullImage
        }

        emulate_visium = VisiumData(
            tissue_positions = tissue_positions,
            feature_bc_matrix = adata,
            scalefactors = scaleF,
            metadata = metadata
        )
        emulate_visium.match2profile(profile)
        emulate_visium.image = self.image
        emulate_visium.pixel_size = self.pixel_size

        return emulate_visium

class XfuseData(VisiumData):

    def convert(self, image_index=0):
        # save image.png
        ii.imsave(self.prefix/"image.png", self.images[self.mask2image[image_index][1]])
        # save mask.png
        mask = self.masks[self.mask2image[image_index][0]] > 0
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

class iStarData(VisiumData):

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

    def convert(self, image_index=0):
        # save he-raw.jpg
        ii.imsave(self.prefix/"he-raw.jpg", self.images[self.mask2image[image_index][1]])
        # save mask-raw.png
        mask = self.masks[self.mask2image[image_index][0]]
        ii.imsave(self.prefix/"mask-raw.png", mask)
        # wirte number of pixels per spot radius
        with open(self.prefix/"radius-raw.txt","w") as f:
            f.write(str(self.scaleF["spot_diameter_fullres"]/2))
        # write side length (in micrometers) of pixels
        with open(self.prefix/"pixel-size-raw.txt","w") as f:
            f.write(str(self.scaleF["tissue_hires_scalef"]*4))
            # f.write(str(65/scaleF["spot_diameter_fullres"]))
        with open(self.prefix/"pixel-size.txt", "w") as f:
            f.write(str(self.pixel_size/16))
        # save spot locations
        locDF = self.transfer_loc()
        locDF[["spot","x","y"]].to_csv(self.prefix/"locs-raw.tsv", sep="\t", index=False)
        # save gene count matrix
        fast_to_csv(self.transfer_cnts(locDF),self.prefix/"cnts.tsv")

class soScopeData(VisiumData):
    pass

class TESLAData(VisiumData):

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

    def convert(self, image_index=0):
        # save image.jpg
        ii.imsave(self.prefix/"image.jpg", self.images[self.mask2image[image_index][1]])
        # save mask.png
        mask = self.masks[self.mask2image[image_index][0]]
        ii.imsave(self.prefix/"mask.png", mask,)
        # save data.h5ad
        self.transfer_h5ad().write_h5ad(self.prefix/"data.h5ad")
        # calculate super pixel step
        with open(self.prefix/"pixel_step.txt","w") as f:
            scale = self.scaleF["tissue_hires_scalef"]*4/self.pixel_size
            f.write(str(int(np.round(1/scale))))

class ImSpiREData(VisiumData):
    
    def convert(self, image_index=0):
        # save image.jpg
        ii.imsave(self.prefix/"image.tif", self.images[self.mask2image[image_index][1]])
        # save h5
        write_10X_h5(self.adata, self.prefix/"filtered_feature_bc_matrix.h5", self.metadata)
        # copy spatial folder
        shutil.copytree(self.path / "spatial", self.prefix / "spatial")
        # calculate patch size
        with open(self.prefix/"patch_size.txt","w") as f:
            scale = self.scaleF["tissue_hires_scalef"]*4/self.pixel_size
            f.write(str(int(np.round(1/scale))))