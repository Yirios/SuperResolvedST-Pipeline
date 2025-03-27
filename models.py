from pathlib import Path
import pickle
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from itertools import product
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import cv2
import pandas as pd
import imageio.v2 as ii
from anndata import AnnData, read_h5ad
from scipy.sparse import csr_matrix
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from utility import *
from profiles import VisiumProfile
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
        
        if isinstance(mask, np.ndarray):
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

    def save_params(self, **kvargs):
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
        write_json(self.prefix/"super_resolution_config.json", parameters)
    
    def update_params(self,**kvargs):
        parameters = read_json(self.prefix/"super_resolution_config.json")
        parameters.update(kvargs)
        write_json(self.prefix/"super_resolution_config.json", parameters)

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
    
    def corp_capture_area(self, img):
        top,left,height,width = self.capture_area
        return img[top:top+height,left:left+width]
    
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
    def to_VisiumHD(self, HDprefix:Path=None, superHD_demo:VisiumHDData=None):
        if not (self.HDData or superHD_demo ):
            raise ValueError("Run set_target_VisiumHD frist or set superHD_demo")
        elif superHD_demo:
            self.HDData = superHD_demo
        
        merged = self.HDData.locDF.reset_index().merge(
            self.SRresult,
            left_on=['array_row', 'array_col'],
            right_on=['x', 'y']
        )

        self.HDData.locDF['in_tissue'] = 0
        if not merged.empty:
            self.HDData.locDF.loc[merged['index'], 'in_tissue'] = 1

        genes = self.SRresult.columns[2:]
        self.SRresult = merged.set_index('barcode')[genes]

        self.HDData.adata = AnnData(
            X=csr_matrix(self.SRresult.to_numpy()),
            obs=pd.DataFrame(index=self.SRresult.index),
            var=self.HDData.adata.var.loc[genes,:])
        if HDprefix != None:
            self.HDData.save(HDprefix)
        return self.HDData

    @timer
    def to_csv(self, file:Path=None, sep="\t"):
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
    def to_h5ad(self, file:Path=None):
        if not file:
            file = self.prefix/"super-resolution.h5ad"
        adata = AnnData(self.SRresult.iloc[:,2:])
        adata.obs = self.SRresult.iloc[:, :2]
        adata.var.index = self.SRresult.columns[2:]
        adata.uns["shape"] = list(self.super_image_shape)
        adata.uns["project_dir"] = str(self.prefix.resolve())
        adata.write_h5ad(file)

class Xfuse(SRtools):

    def write_in_xfuse_format(
            path: Path,
            counts: pd.DataFrame,
            image: np.ndarray,
            label: np.ndarray,
            type_label: Optional[str]="ST",
        ):
        '''
        copy and edit base on https://github.com/ludvb/xfuse/blob/master/xfuse/convert/utility.py
        unsupport annotation
        '''
        def _normalize(x: np.ndarray, axis=None) -> np.ndarray:
            import warnings
            x = x - x.min(axis, keepdims=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x = x / x.max(axis, keepdims=True)
                x = np.nan_to_num(x)
            return x
        
        image = _normalize(image.astype(np.float32), axis=(0, 1)) * 2 - 1
        image = 0.9 * image

        with h5py.File(path, "w") as data_file:
            data = csr_matrix(counts.values.astype(float))
            data_file.create_dataset(
                "counts/data", data.data.shape, float, data.data.astype(float)
            )
            data_file.create_dataset(
                "counts/indices",
                data.indices.shape,
                data.indices.dtype,
                data.indices,
            )
            data_file.create_dataset(
                "counts/indptr", data.indptr.shape, data.indptr.dtype, data.indptr
            )
            data_file.create_dataset(
                "counts/columns",
                counts.columns.shape,
                h5py.string_dtype(),
                counts.columns.values,
            )
            data_file.create_dataset(
                "counts/index", counts.index.shape, int, counts.index.astype(int)
            )
            data_file.create_dataset("image", image.shape, np.float32, image)
            data_file.create_dataset("label", label.shape, np.int16, label)
            data_file.create_group("annotation", track_order=True)
            data_file.create_dataset(
                "type", data=type_label, dtype=h5py.string_dtype()
            )


    def transfer_image_HD(self, patch_pixel):
        patch_shape = (patch_pixel, patch_pixel)
        num_row = self.HDData.profile.row_range
        num_col = self.HDData.profile.col_range
        visium_w,visium_h = self.profile.frame
        Hsilde = max(num_row, int(visium_h/self.HDData.bin_size+1)) + 4
        Wslide = max(num_col, int(visium_w/self.HDData.bin_size+1)) + 4
        HDdx = (Hsilde-num_row)//2
        HDdy = (Wslide-num_col)//2
        bin_patch_shape = [
            Hsilde,Wslide,*patch_shape
            ]
        if self.HDData.image_channels > 1:
            bin_patch_shape.append(self.HDData.image_channels)
        
        patch_array = np.full(bin_patch_shape, fill_value=255, dtype=np.uint8)
        patch_array[HDdx:HDdx+num_row,HDdy:HDdy+num_col] = self.HDData.crop_patch(patch_shape=patch_shape)
        for i,j in get_outside_indices((Hsilde,Wslide), HDdx, HDdy, num_row, num_col):
            x,y,_ = self.HDData.profile[i-HDdx,j-HDdy]
            corners = get_corner(x,y,self.HDData.bin_size,self.HDData.bin_size)
            cornerOnImage = self.HDData.mapper.transform_batch(np.array(corners))
            image_flips = self.HDData.mapper.check_flips()
            patchOnImage = crop_single_patch(image=self.HDData.image, corners=cornerOnImage, flips=image_flips)
            patch_array[i,j] = image_resize(patchOnImage, shape=patch_shape)
        img = reconstruct_image(patch_array)
        img = image_resize(img, shape=(Hsilde,Wslide)) # resize to silde shape
        binsOnImage = self.HDData.profile.tissue_positions[["pxl_row_in_fullres","pxl_col_in_fullres"]].values
        # binsOnHD = self.HDData.profile.tissue_positions[["array_row","array_col"]].values*patch_pixel  \
        #     + (np.array((HDdx,HDdy))+0.5)*np.array(patch_shape)
        binsOnHD = self.HDData.profile.tissue_positions[["array_row","array_col"]].values  \
            + (np.array((HDdx,HDdy))+0.5)
        HDmapper = AffineTransform(binsOnImage, binsOnHD)
        capture_area = (HDdx, HDdy, num_row, num_col)
        scaleF = 1/HDmapper.resolution
        return img, capture_area, HDmapper, scaleF
    
    def transfer_image_mask_HD(self, patch_pixel):
        mask = 255 - self.mask
        mask = mask.astype(np.uint8)[..., np.newaxis]
        self.HDData.image_channels += 1
        self.HDData.image = np.concatenate([self.HDData.image, mask], axis=2)
        img, capture_area, HDmapper, scaleF = self.transfer_image_HD(patch_pixel)
        self.HDData.image_channels -= 1
        mask = 255 - img[:,:,3]
        img = img[:,:,:3]
        self.HDData.image = self.HDData.image[:,:,:3]
        return img, mask, capture_area, HDmapper, scaleF
    
    def transfer_label_HD(self, image_shape, spot_radius , mapper:AffineTransform, mask) -> np.ndarray:
        df = self.locDF.copy(True)
        df.columns = self.profile.RawColumns
        df = df[df["in_tissue"]==1]
        spotOnImage = mapper.transform_batch(df[["pxl_row_in_fullres","pxl_col_in_fullres"]].values) 
        label = np.where(mask, 0, 1)
        bin_iter = lambda a: range(int(a-spot_radius),int((a+spot_radius)+1.5))
        d2 = lambda x,y,a,b: (x-a)*(x-a) + (y-b)*(y-b)
        for spot_id, spot_center in enumerate(spotOnImage, start=2):
            spot_x,spot_y = spot_center.tolist()
            for i,j in product(bin_iter(spot_x), bin_iter(spot_y)):
                bin_x = i+0.5
                bin_y = j+0.5
                if d2(bin_x,bin_y,spot_x,spot_y) < spot_radius*spot_radius:
                    if i<0 or j<0 or i>=image_shape[0] or j>=image_shape[1]:
                        continue
                    label[i, j] = spot_id
        return label

    def convert(self):
        if self.HDData == None:
            # Image mode use xfuse covert to preprocess
            # save image.png
            ii.imsave(self.prefix/"image.png", self.image)
            # save mask.png
            mask = self.mask > 0
            mask = np.where(mask, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
            ii.imsave(self.prefix/"mask.png", mask)
            # save h5
            write_10X_h5(self.adata, self.prefix/"filtered_feature_bc_matrix.h5")
            
            # calculate scale 
            with open(self.prefix/"scale.txt","w") as f:
                f.write(str(self.pixel_size/self.super_pixel_size))
        else:
            # Visium HD mode self define preprocess
            patch_pixel_size = int(self.super_pixel_size/self.pixel_size+0.5)
            img, mask, capture_area, HDmapper, scaleF = self.transfer_image_mask_HD(patch_pixel_size)
            self.super_image_shape = img.shape[:2]
            self.capture_area = capture_area
            radius = self.scaleF["spot_diameter_fullres"]*scaleF/2
            label = self.transfer_label_HD(self.super_image_shape, radius, HDmapper, mask>0)
            # label_image = np.zeros(label.shape, dtype=np.uint8)
            # label_image[label>0]=255
            # ii.imsave("test_xfuse.png", label_image)
            counts = self.adata.to_df()
            counts.index = pd.Index([*range(1, counts.shape[0] + 1)], name="n") + 1
            counts = pd.concat(
                [
                    pd.DataFrame(
                        [np.repeat(0, counts.shape[1])],
                        columns=counts.columns,
                        index=[1],
                    ).astype(pd.SparseDtype("float", 0)),
                    counts,
                ]
            )
            (self.prefix/"data").mkdir(parents=True, exist_ok=True)
            Xfuse.write_in_xfuse_format(
                path=self.prefix/"data/data.h5",
                counts=counts,
                image=img,
                label=label
            )

    def load_output(self, prefix:Path=None):
        super().load_output(prefix)
        mask = ii.imread(self.prefix/'mask.png')

        if mask.shape != self.super_image_shape:
            mask = image_resize(mask, shape=self.super_image_shape)
        mask = self.corp_capture_area(mask)
        mask = mask > 127
    
    def load_output(self, prefix:Path=None):
        super().load_output(prefix)
        import torch
        
        with h5py.File(self.prefix/"data/data.h5", "r") as f:
            mask = f["label"][:,:]==1
        mask = self.corp_capture_area(mask)
        
        # select unmasked super pixel 
        Xs,Ys = np.where(np.logical_not(mask))
        data = {"x":Xs, "y":Ys}
        
        # select genes
        with open(self.prefix/'gene-names.txt', 'r') as file:
            genes = [line.rstrip() for line in file]
        # genes = ["HES4","VWA1","AL645728.1","GABRD"] # test genes
        gene_iter = progress_bar(
            title="Reading xfuse output",
            iterable=genes,
            total=len(genes)
        )
        for gene in gene_iter():
            cnts = torch.load(self.prefix/f"result/analyses/final/gene_maps/section1/{gene}.pt")
            cnts = np.mean(cnts, axis=0)
            data[gene]=[float(f"{x:.8f}") for x in np.round(cnts[Xs, Ys], decimals=8)]
        self.SRresult = pd.DataFrame(data)

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

    def transfer_image_HD(self):
        patch_shape=(16,16)
        num_row = self.HDData.profile.row_range
        num_col = self.HDData.profile.col_range
        H16 = (num_row + 15) // 16 * 16
        W16 = (num_col + 15) // 16 * 16
        HDdx = (H16-num_row)//2
        HDdy = (W16-num_col)//2
        bin_patch_shape = [
            H16,W16,*patch_shape
            ]
        if self.HDData.image_channels > 1:
            bin_patch_shape.append(self.HDData.image_channels)
        
        patch_array = np.full(bin_patch_shape, fill_value=255, dtype=np.uint8)
        patch_array[HDdx:HDdx+num_row,HDdy:HDdy+num_col] = self.HDData.crop_patch(patch_shape=patch_shape)
        for i,j in get_outside_indices((H16,W16), HDdx, HDdy, num_row, num_col):
            x,y,_ = self.HDData.profile[i-HDdx,j-HDdy]
            corners = get_corner(x,y,self.HDData.bin_size,self.HDData.bin_size)
            cornerOnImage = self.HDData.mapper.transform_batch(np.array(corners))
            image_flips = self.HDData.mapper.check_flips()
            patchOnImage = crop_single_patch(image=self.HDData.image, corners=cornerOnImage, flips=image_flips)
            patch_array[i,j] = image_resize(patchOnImage, shape=patch_shape)
        img = reconstruct_image(patch_array)
        binsOnImage = self.HDData.profile.tissue_positions[["pxl_row_in_fullres","pxl_col_in_fullres"]].values
        binsOnHD = self.HDData.profile.tissue_positions[["array_row","array_col"]].values*16  \
            + (np.array((HDdx,HDdy))+0.5)*np.array(patch_shape)
        HDmapper = AffineTransform(binsOnImage, binsOnHD)
        capture_area = (HDdx, HDdy, num_row, num_col)
        scaleF = 1/HDmapper.resolution
        return img, capture_area, HDmapper, scaleF
    
    def transfer_image_mask_HD(self):
        mask = 255 - self.mask
        mask = mask.astype(np.uint8)[..., np.newaxis]
        self.HDData.image_channels += 1
        self.HDData.image = np.concatenate([self.HDData.image, mask], axis=2)
        img, capture_area, HDmapper, scaleF = self.transfer_image_HD()
        self.HDData.image_channels -= 1
        mask = 255 - img[:,:,3]
        img = img[:,:,:3]
        self.HDData.image = self.HDData.image[:,:,:3]
        return img, mask, capture_area, HDmapper, scaleF

    def transfer_loc_HD(self, mapper:AffineTransform) -> pd.DataFrame:
        df = self.locDF.copy(True)
        df.columns = ["barcode","in_tissue","array_row","array_col","y","x"]
        df = df[df["in_tissue"]==1]
        del df["in_tissue"]
        df["spot"] = df["array_row"].astype(str) + "x" + df["array_col"].astype(str)
        df = df.astype({"y": float, "x": float})
        df.loc[:, ["y", "x"]] = mapper.transform_batch(df[["y", "x"]].values) 
        return df

    def convert(self):
        if self.HDData == None:
            image, scaleF = self.transfer_image_base(self.image)
            mask = self.transfer_mask_base(self.mask)
            locDF = self.transfer_loc_base(scaleF)
            self.super_image_shape = [i//16 for i in mask.shape]
        else:
            image, mask, capture_area, HDmapper, scaleF = self.transfer_image_mask_HD()
            locDF = self.transfer_loc_HD(HDmapper)
            self.super_image_shape = [i//16 for i in mask.shape]
            self.capture_area = capture_area
            
        ii.imsave(self.prefix/"he.jpg", image)
        # save mask.png
        ii.imsave(self.prefix/"mask.png", mask)
        # save spot locations
        locDF[["spot","x","y"]].to_csv(self.prefix/"locs.tsv", sep="\t", index=False)

        # wirte number of pixels per spot radius
        radius = self.scaleF["spot_diameter_fullres"]/2*scaleF
        pixel_size_raw = self.pixel_size/scaleF
        pixel_size = self.super_pixel_size
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
        top,left,height,width = self.capture_area
        return img[top:top+height,left:left+width]

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
        from anndata import concat as ann_concat
        df = self.locDF.copy()
        df.columns = ["barcode","in_tissue","array_row","array_col","pixel_x","pixel_y"]
        df.index = df["barcode"]
        adata = ann_concat(
            [
                AnnData(
                    X=csr_matrix((len(df),0)),
                    obs=df
                ),
                self.adata
            ],
            join="inner", 
            merge="first",
            axis=1
        )
        return adata
    
    def mask_in_loc(self, patch_array, capture_area):
        mask = np.mean(patch_array[:,:,:,:,3],axis=(2,3)) < 128
        margin = np.full_like(mask, False)
        top,left,height,width = capture_area
        margin[top:top+height,left:left+width] = True
        mask = np.logical_and(mask, margin)
        rows, cols = np.where(mask)
        rows -= top
        cols -= left
        temp_df = pd.DataFrame({'array_row': rows, 'array_col': cols})
        merged = self.HDData.locDF.reset_index().merge(
            temp_df, on=['array_row', 'array_col'], how='inner'
        )
        self.HDData.locDF.loc[merged['index'], 'in_tissue'] = 1
        return mask
    
    def transfer_image_HD(self, patch_pixel):
        patch_shape = (patch_pixel, patch_pixel)
        num_row = self.HDData.profile.row_range
        num_col = self.HDData.profile.col_range
        
        patch_array = self.HDData.crop_patch(patch_shape=patch_shape)
        capture_area = (0, 0, num_row, num_col)
        # mask in channel 3
        if patch_array.shape[4] == 4 :
            self.mask_in_loc(patch_array, capture_area)
            patch_array = patch_array[:,:,:,:,:3]
        # cols = ["barcode","array_row","array_col"]
        # bin_patchs = {
        #     str(barcode):patch_array[i,j].reshape(-1,3)
        #     for barcode, i,j in self.HDData.locDF[cols].values
        # }
        spot_patchs = self.crop_patch(in_tissue=True)
        # return bin_patchs, spot_patchs, capture_area
        return patch_array, spot_patchs, capture_area
    
    def transfer_image_mask_HD(self, patch_pixel):
        mask = 255 - self.mask
        mask = mask.astype(np.uint8)[..., np.newaxis]
        self.HDData.image_channels += 1
        self.HDData.image = np.concatenate([self.HDData.image, mask], axis=2)
        # bin_patchs, spot_patchs, capture_area = self.transfer_image_HD(patch_pixel)
        patch_array, spot_patchs, capture_area = self.transfer_image_HD(patch_pixel)
        self.HDData.image_channels -= 1
        self.mask = 255 - self.HDData.image[:,:,3]
        self.HDData.image = self.HDData.image[:,:,:3]
        # return bin_patchs, spot_patchs, capture_area
        return patch_array, spot_patchs, capture_area

    def convert(self):
        if self.HDData == None:
            # save image.jpg
            ii.imsave(self.prefix/"image.jpg", self.image)
            # save mask.png
            ii.imsave(self.prefix/"mask.png", self.mask)
            # save data.h5ad
            self.transfer_h5ad().write_h5ad(self.prefix/"data.h5ad")
            # calculate super pixel step
            with open(self.prefix/"pixel_step.txt","w") as f:
                f.write(str(self.pixel_size/self.super_pixel_size))
        else:
            patch_pixel_size = int(self.super_pixel_size/self.pixel_size+0.5)
            # bin_patchs, spot_patchs, capture_area = self.transfer_image_mask_HD(patch_pixel_size)
            patch_array, spot_patchs, capture_area = self.transfer_image_mask_HD(patch_pixel_size)
            # # bin patch images
            # np.savez_compressed(self.prefix/"bin_image.npz", **bin_patchs)
            np.save(self.prefix/"bin_image.npy", patch_array)
            # spot patch images
            np.savez_compressed(self.prefix/"spot_image.npz", **spot_patchs)
            # bin positions
            pd.to_pickle(
                self.HDData.locDF,
                self.prefix/"bin_positions.pkl"
            )
            self.transfer_h5ad().write_h5ad(self.prefix/"data.h5ad")
            self.super_image_shape = capture_area[2:]
            self.capture_area = capture_area

    def load_output(self, prefix:Path=None):
        super().load_output(prefix)
        adata = read_h5ad(self.prefix/"enhanced_exp.h5ad")
        self.SRresult = adata.to_df()

        top,left,height,width = self.capture_area
        mask_x = (adata.obs["x_super"] >= top) & (adata.obs["x_super"] < top + height)
        mask_y = (adata.obs["y_super"] >= left) & (adata.obs["y_super"] < left+ width)
        mask = np.logical_and(mask_x,mask_y)

        self.SRresult = self.SRresult.loc[mask,:]
        self.SRresult.insert(0, 'x', adata.obs.loc[mask,"x_super"].astype(int)-top)
        self.SRresult.insert(0, 'y', adata.obs.loc[mask,"y_super"].astype(int)-left)

class ImSpiRE(SRtools):
    
    def transfer_image_HD(self, patch_pixel):
        patch_shape = (patch_pixel, patch_pixel)
        num_row = self.HDData.profile.row_range
        num_col = self.HDData.profile.col_range
        visium_w,visium_h = self.profile.frame
        Hsilde = max(num_row, int(visium_h/self.HDData.bin_size+1)) + 4
        Wslide = max(num_col, int(visium_w/self.HDData.bin_size+1)) + 4
        HDdx = (Hsilde-num_row)//2
        HDdy = (Wslide-num_col)//2
        bin_patch_shape = [
            Hsilde,Wslide,*patch_shape
            ]
        if self.HDData.image_channels > 1:
            bin_patch_shape.append(self.HDData.image_channels)
        
        patch_array = np.full(bin_patch_shape, fill_value=255, dtype=np.uint8)
        patch_array[HDdx:HDdx+num_row,HDdy:HDdy+num_col] = self.HDData.crop_patch(patch_shape=patch_shape)
        for i,j in get_outside_indices((Hsilde,Wslide), HDdx, HDdy, num_row, num_col):
            x,y,_ = self.HDData.profile[i-HDdx,j-HDdy]
            corners = get_corner(x,y,self.HDData.bin_size,self.HDData.bin_size)
            cornerOnImage = self.HDData.mapper.transform_batch(np.array(corners))
            image_flips = self.HDData.mapper.check_flips()
            patchOnImage = crop_single_patch(image=self.HDData.image, corners=cornerOnImage, flips=image_flips)
            patch_array[i,j] = image_resize(patchOnImage, shape=patch_shape)
        img = reconstruct_image(patch_array)
        binsOnImage = self.HDData.profile.tissue_positions[["pxl_row_in_fullres","pxl_col_in_fullres"]].values
        binsOnHD = self.HDData.profile.tissue_positions[["array_row","array_col"]].values*patch_pixel  \
            + (np.array((HDdx,HDdy))+0.5)*np.array(patch_shape)
        HDmapper = AffineTransform(binsOnImage, binsOnHD)
        capture_area = (HDdx, HDdy, num_row, num_col)
        scaleF = 1/HDmapper.resolution
        return img, capture_area, HDmapper, scaleF
    

    def transfer_image_mask_HD(self, patch_pixel):
        mask = 255 - self.mask
        mask = mask.astype(np.uint8)[..., np.newaxis]
        self.HDData.image_channels += 1
        self.HDData.image = np.concatenate([self.HDData.image, mask], axis=2)
        img, capture_area, HDmapper, scaleF = self.transfer_image_HD(patch_pixel)
        self.HDData.image_channels -= 1
        mask = 255 - img[:,:,3]
        img = img[:,:,:3]
        self.HDData.image = self.HDData.image[:,:,:3]
        return img, mask, capture_area, HDmapper, scaleF

    def transfer_loc_HD(self, mapper:AffineTransform) -> pd.DataFrame:
        df = self.locDF.copy(True)
        df.columns = self.profile.RawColumns
        df[["pxl_row_in_fullres","pxl_col_in_fullres"]] = \
            mapper.transform_batch(df[["pxl_row_in_fullres","pxl_col_in_fullres"]].values) 
        return df
    
    def convert(self):
        patch_pixel_size = int(self.super_pixel_size/self.pixel_size+0.5)

        if self.HDData == None:
            self.save(self.prefix)
            self.super_image_shape = [i//patch_pixel_size for i in self.image.shape[:2]]
            mask = image_resize(self.mask, shape=self.super_image_shape)
            np.save(self.prefix/"mask.npy", mask>127)
        else:
            img, mask, capture_area, HDmapper, scaleF = self.transfer_image_mask_HD(patch_pixel_size)
            mask = image_resize(mask, shape=capture_area[2:])
            np.save(self.prefix/"mask.npy", mask>127)
            locDF = self.transfer_loc_HD(HDmapper)
            FullImage = np.max(img.shape)
            scaleFs = {
                "spot_diameter_fullres": self.scaleF["spot_diameter_fullres"]*scaleF,
                "tissue_lowres_scalef": self.profile.LowresImage/FullImage,
                "fiducial_diameter_fullres": self.scaleF["fiducial_diameter_fullres"]*scaleF,
                "tissue_hires_scalef": self.profile.HiresImage[self.profile.serial]/FullImage
            }
            temp_visium = VisiumData(
                tissue_positions = locDF,
                feature_bc_matrix = self.adata,
                scalefactors = scaleFs,
                metadata = self.metadata.copy()
            )
            temp_visium.metadata["software_version"] = "spaceranger-1.3.0"
            temp_visium.image = img
            temp_visium.match2profile(VisiumProfile(slide_serial=1), quiet=True)
            temp_visium.save(self.prefix)
            self.super_image_shape = [i//patch_pixel_size for i in img.shape[:2]]
            self.capture_area = capture_area
        
        with open(self.prefix/"patch_size.txt","w") as f:
            f.write(str(patch_pixel_size))

    def load_output(self, prefix:Path=None):
        super().load_output(prefix)
        adata = read_h5ad(self.prefix/"result/result_ResolutionEnhancementResult.h5ad")
        self.SRresult = adata.to_df()
        locDF = pd.read_csv(self.prefix/"result/result_PatchLocations.txt", sep="\t",)
        locDF.columns = ['index', 'row', 'col', 'pxl_row', 'pxl_col', 'in_tissue']
        self.SRresult.index = self.SRresult.index.astype(int)
        # crop capture area
        top,left,height,width = self.capture_area
        capture_mask = (
            locDF["row"].between(top, top + height, inclusive="left") & \
            locDF["col"].between(left, left + width, inclusive="left")
        )
        locDF = locDF[capture_mask]
        locDF["x"] = locDF["row"] - top
        locDF["y"] = locDF["col"] - left
        self.SRresult = pd.merge(locDF, self.SRresult, left_index=True, right_index=True).iloc[:, 6:]
