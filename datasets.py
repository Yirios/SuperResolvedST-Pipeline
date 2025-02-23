
from pathlib import Path
from typing import List, Dict, Tuple
import copy

import numpy as np
import cv2
import pandas as pd
import h5py
import scanpy as sc
import imageio.v2 as ii
import tifffile
from anndata import AnnData
from scipy.sparse import csr_matrix
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utility import *
from profiles import *

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
        self.locDF.columns = Profile.RawColumns
        return self.locDF

    def _read_scalefactors(self) -> Dict:
        file = self.path/"spatial/scalefactors_json.json"
        self.scaleF = read_json(file)
        return self.scaleF

    def _read_feature_bc_matrix(self) -> AnnData:
        file = self.path/"filtered_feature_bc_matrix.h5"
        self.adata = sc.read_10x_h5(file)
        with h5py.File(file, mode="r") as f:
            self.metadata = dict(f.attrs)

    def read_image(self, file:Path):
        self.image = tifffile.imread(file)
        if len(self.image.shape) == 2:
            self.image_channels = 1
        else:
            self.image_channels = self.image.shape[2]
    
    @timer
    def load(self, path:Path, source_image_path:Path=None):
        self.path = Path(path)
        self._read_feature_bc_matrix()
        self._read_scalefactors()
        self._read_location()
        if source_image_path :
            self.read_image(Path(source_image_path))

    def _save_location(self,path):
        self.locDF.to_csv(path/"spatial/tissue_positions.csv", index=False, header=False)

    def _save_scalefactors(self, path):
        write_json(path/"spatial/scalefactors_json.json", self.scaleF)

    def _save_feature_bc_matrix(self, path):
        file = path/'filtered_feature_bc_matrix.h5'
        write_10X_h5(self.adata, file, self.metadata)

    def _save_images(self, path):
        # raw image
        tifffile.imwrite(path/"image.tif", self.image, bigtiff=True)
        # spatial image
        lowres_image = image_resize(self.image, scalef=self.scaleF["tissue_lowres_scalef"])
        ii.imsave(path/"spatial/tissue_lowres_image.png", lowres_image)
        hires_image = image_resize(self.image, scalef=self.scaleF["tissue_hires_scalef"])
        ii.imsave(path/"spatial/tissue_hires_image.png", hires_image)

    @timer
    def save(self, path:Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path/"spatial").mkdir(parents=True, exist_ok=True)
        self._save_images(path)
        self._save_feature_bc_matrix(path)
        self._save_scalefactors(path)
        self._save_location(path)

    def match2profile(self, profile:Profile):
        self.profile = profile
        # keep order
        order = self.profile.tissue_positions[["array_row", "array_col"]].values.tolist()
        raw_order = self.locDF[["array_row", "array_col"]].values.tolist()
        if not np.array_equal(order, raw_order):
            print("Unable to match the profile, transferring to the specified profile.")
            temp = self.locDF.set_index(["array_row", "array_col"])
            self.locDF = temp.loc[order]
            self.locDF.reset_index(inplace=True)
            self.locDF = self.locDF[profile.RawColumns]
        # map to image
        PointsOnFrame = self.profile.tissue_positions[["frame_row","frame_col"]].values
        PointsOnImage = self.locDF[["pxl_row_in_fullres","pxl_col_in_fullres"]].values
        self.mapper = AffineTransform(PointsOnFrame, PointsOnImage)
        # add in location 
        self.profile.tissue_positions[["pxl_row_in_fullres","pxl_col_in_fullres"]] = PointsOnImage
        self.pixel_size = self.mapper.resolution

    def select_HVG(self,n_top_genes=2000, min_counts=10) -> None:
        self.adata.var_names_make_unique()
        sc.pp.filter_genes(self.adata, min_counts=min_counts)
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

    def vision_spots(self, in_tissue=False):
        image = self.image.copy()
        if not in_tissue:
            spots = self.locDF[["pxl_col_in_fullres","pxl_row_in_fullres"]].values
        else:
            spots = self.locDF.loc[self.locDF['in_tissue']==1, ["pxl_col_in_fullres","pxl_row_in_fullres"]].values
        for pt in spots:
            cv2.circle(image, tuple(pt), 50, (255,0,0), -1)
        return image
    
    def _bin2image(self, profile:VisiumHDProfile, frame_center:Tuple[float, float]):
        if {"pxl_row_in_fullres", "pxl_col_in_fullres"} <= set(profile.tissue_positions.columns):
            return profile.tissue_positions[["pxl_row_in_fullres","pxl_col_in_fullres"]]
        
        x0 = frame_center[0]-self.profile.frame[1]/2
        y0 = frame_center[1]-self.profile.frame[0]/2
        binsOnFrame = profile.tissue_positions[["frame_row","frame_col"]].values - np.array([[x0,y0]])
        binsOnImage = self.mapper.transform_batch(binsOnFrame)
        profile.tissue_positions[["pxl_row_in_fullres","pxl_col_in_fullres"]] = binsOnImage

        return binsOnImage
    
    def drop_spots(self, spot_ids:List, in_place=False) :
        if not in_place:
            dropped = copy.deepcopy(self)
            dropped.drop_spots(spot_ids,in_place=True)
            return dropped
        drop_mask = self.profile.tissue_positions["id"].isin(spot_ids)
        tissue_mask = self.locDF["in_tissue"].values == 1
        drop_mask = drop_mask & tissue_mask
        sys.stdout.write(f"{sum(drop_mask)} spots will be dropped.\n")

        self.locDF.loc[drop_mask,"in_tissue"] = 0
        undrop_mask = tissue_mask & ~drop_mask
        undrop_barcode = self.locDF.loc[undrop_mask,"barcode"]
        self.adata = self.adata[undrop_barcode,:]

    def Visium2HD(self, HDprofile:VisiumHDProfile, **kwargs) -> "VisiumHDData":

        _, frame_center =  align_profile(HDprofile, self.profile, **kwargs)
        self._bin2image(HDprofile, frame_center)

        # Get demo VisiumHD: without feature_bc_matrix
        metadata = HDprofile.metadata.copy()
        metadata["library_ids"] = self.metadata["library_ids"]
        
        FullImage = np.max(self.image.shape)
        scaleF = {
            "spot_diameter_fullres": HDprofile.bin_size/self.pixel_size,
            "bin_size_um": HDprofile.bin_size,
            "microns_per_pixel": self.pixel_size,
            "tissue_lowres_scalef": HDprofile.LowresImage/FullImage,
            "fiducial_diameter_fullres": self.scaleF["fiducial_diameter_fullres"],
            "tissue_hires_scalef": HDprofile.HiresImage/FullImage,
            "regist_target_img_scalef": HDprofile.HiresImage/FullImage,
        }

        adata = AnnData(
            X=csr_matrix((0, len(self.adata.var))),
            var=self.adata.var
        )

        cols = [col for col in HDprofile.RawColumns if col != "in_tissue"]
        tissue_positions = HDprofile.tissue_positions[cols].copy()
        tissue_positions["in_tissue"] = np.repeat(0,len(tissue_positions))
        tissue_positions = tissue_positions[HDprofile.RawColumns]
        superHD_demo = VisiumHDData(
                tissue_positions = tissue_positions,
                feature_bc_matrix = adata,
                scalefactors = scaleF,
                metadata = metadata
            )
        superHD_demo.image = self.image
        superHD_demo.image_channels = self.image_channels
        superHD_demo.pixel_size = self.pixel_size
        superHD_demo.bin_size = HDprofile.bin_size
        superHD_demo.match2profile(HDprofile)

        return superHD_demo

class VisiumHDData(rawData):

    def _read_location(self) -> pd.DataFrame:
        file = self.path/"spatial/tissue_positions.parquet"
        self.locDF = pd.read_parquet(file)
        return self.locDF
    
    def _save_location(self,path):
        file = path/"spatial/tissue_positions.parquet"
        self.locDF.to_parquet(file, index=False)

    def load(self, path:Path, profile:VisiumHDProfile, source_image_path:Path=None):
        super().load(path, source_image_path)
        
        self.bin_size = self.scaleF["bin_size_um"]
        self.pixel_size = self.scaleF["microns_per_pixel"]
        if profile.bin_size != self.bin_size:
            warnings.warn(f"bin size of VisiumHD is {self.bin_size}, but profile recommend {profile.bin_size}, Start Rebinnig")
            self.rebining()
        self.match2profile(profile)

    def rebining(self, profile:VisiumHDProfile) -> "VisiumHDData":
        '''\
        TODO bin in user define profile
        '''
        pass

    def crop_patch(self, patch_size=None, patch_shape=None):
        if not patch_size:
            patch_size = self.bin_size
        if not patch_shape:
            patch_pixel = int(patch_size/self.pixel_size+0.5)
            patch_shape = (patch_pixel, patch_pixel)
        
        bins = progress_bar(
            title="Cropping patch image of each bin",
            iterable=self.profile.bins,
            total=len(self.profile)
        )
        
        bin_patch_shape = [
            self.profile.row_range,
            self.profile.col_range,
            *patch_shape
            ]
        if self.image_channels > 1:
            bin_patch_shape.append(self.image_channels)
        
        patch_array = np.full(bin_patch_shape, fill_value=255, dtype=np.uint8)
        for id,(i,j),(x,y,_) in bins():
            corners = get_corner(x,y,*patch_shape)
            cornerOnImage = self.mapper.transform_batch(np.array(corners))
            patchOnImage = crop_single_patch(self.image, cornerOnImage)
            patch_array[i,j] = image_resize(patchOnImage, shape=patch_shape)

        return patch_array


    def _spot2image(self, profile:VisiumProfile, frame_center:Tuple[float, float]):
        '''\
        add pxl_row_in_fullres, pxl_col_in_fullres in profile.tissue_positions
        '''
        if {"pxl_row_in_fullres", "pxl_col_in_fullres"} <= set(profile.tissue_positions.columns):
            return profile.tissue_positions[["pxl_row_in_fullres","pxl_col_in_fullres"]]
        x0 = frame_center[0]-self.profile.frame[1]/2
        y0 = frame_center[1]-self.profile.frame[0]/2
        spotsOnFrame = profile.tissue_positions[["frame_row","frame_col"]].values + np.array([[x0,y0]])
        spotsOnImage = self.mapper.transform_batch(spotsOnFrame)
        profile.tissue_positions[["pxl_row_in_fullres","pxl_col_in_fullres"]] = spotsOnImage

        return spotsOnImage
    
    def HD2Visium(self, profile:VisiumProfile, uncover_thresholds=0, **kwargs) -> VisiumData:
        readyHD = {'spot_label', "pxl_row_in_fullres", "pxl_col_in_fullres"}  <= \
            set(self.profile.tissue_positions.columns)
        readyVisium = {"num_bin_out_spot","num_bin_in_spot"}  <= \
            set(profile.tissue_positions.columns)
        if not (readyHD and readyVisium):
            _, frame_center =  align_profile(self.profile, profile, **kwargs)
            self._spot2image(profile, frame_center)
        
        X_indptr = [0]
        X_indices = np.zeros(0)
        X_data = np.zeros(0)
        spot_in_tissue = np.zeros(len(profile), dtype=int)
        
        mask_in_tissue = self.locDF["in_tissue"] == 1

        spot_iter = progress_bar(
            title="Merge the gene expression from bins to the spot",
            iterable=range(len(profile)),
            total=len(profile)
        )
        for id in spot_iter():
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
        
        tissue_positions = profile.tissue_positions[["barcode","array_row","array_col"]].copy()
        tissue_positions["pxl_row_in_fullres"] = np.round(profile.tissue_positions["pxl_row_in_fullres"].values).astype(int)
        tissue_positions["pxl_col_in_fullres"] = np.round(profile.tissue_positions["pxl_col_in_fullres"].values).astype(int)
        tissue_positions["in_tissue"] = spot_in_tissue
        tissue_positions = tissue_positions[profile.RawColumns]
        
        X_sparse = csr_matrix((X_data, X_indices, X_indptr), shape=(np.sum(spot_in_tissue), len(self.adata.var)))
        mask_in_tissue = spot_in_tissue == 1
        adata = AnnData(
            X=X_sparse,
            var=self.adata.var,
            obs=pd.DataFrame(index=tissue_positions.loc[mask_in_tissue,"barcode"].values)
        )

        metadata = profile.metadata.copy()
        metadata["library_ids"] = self.metadata["library_ids"]

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
