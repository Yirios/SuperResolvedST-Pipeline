
import argparse, math, json
from pathlib import Path
from typing import List, Dict,Tuple
from itertools import product
import hashlib
import warnings 

import pandas as pd
import numpy as np
import scanpy as sc
import h5py
from anndata import AnnData
from scipy.sparse import csr_matrix
import tifffile
import imageio.v2 as ii

class AffineTransform:
    def __init__(self, A_points, B_points):
        """
        初始化仿射变换模型。
        :param A_points: N x 2 数组，表示坐标系 A 下的点
        :param B_points: N x 2 数组，表示坐标系 B 下的对应点
        """
        self.A_points = np.array(A_points)
        self.B_points = np.array(B_points)
        self.M = None  # 变换矩阵
        self.T = None  # 平移向量
        self._compute_transformation()

    def _compute_transformation(self):
        """ 计算仿射变换矩阵和偏移量 """
        N = self.A_points.shape[0]
        X = np.hstack([self.A_points, np.ones((N, 1))])  # 添加一列 1 进行平移
        
        # 目标向量
        U = self.B_points[:, 0]  # B 坐标系的 u
        V = self.B_points[:, 1]  # B 坐标系的 v
        
        # 计算最小二乘解
        params_u, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        params_v, _, _, _ = np.linalg.lstsq(X, V, rcond=None)
        
        # 提取参数
        a, b, c = params_u  # u = ax + by + c
        d, e, f = params_v  # v = dx + ey + f
        
        # 存储变换矩阵和偏移量
        self.M = np.array([[a, b], [d, e]])
        self.T = np.array([c, f])
    
    def __call__(self, A_point):
        """ 将 A 坐标系的点变换到 B 坐标系 """
        A_point = np.array(A_point)
        return self.M @ A_point + self.T

    def transform_batch(self, A_points):
        """ 对多个点进行变换 """
        A_points = np.array(A_points)
        return (self.M @ A_points.T).T + self.T
    
    def get_parameters(self):
        """ 获取仿射变换矩阵和偏移量 """
        return self.M, self.T

def hash_to_dna(index, length=16, suffix="-1"):
    """ 通过哈希值生成 DNA 序列 """
    bases = "ACGT"
    hash_val = hashlib.md5(str(index).encode()).hexdigest()  # 生成哈希
    dna_seq = "".join(bases[int(c, 16) % 4] for c in hash_val[:length])
    return f"{dna_seq}{suffix}"

def write_10X_h5(adata, file:Path, metadata={}) -> None:
    """\
    Writes an AnnData object to a 10X Genomics formatted HDF5 file.
    
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

        for key, value in metadata.items():
            w.attrs[key] = value

        grp = w.create_group("matrix")

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

class VisiumProfile:
    def __init__(self):
        self.spot_diameter = 55.0
        self.spot_step = 100.0
        self.row_range = 64     # even numbers from 0 to 126 for even rows, and odd numbers from 1 to 127 for odd rows with each row (even or odd) resulting in 64 spots.
        self.col_range = 78
        self.tissue_positions =  self.__get_dataframe()
        self.metadata = {
            'chemistry_description': 'custom',
            'filetype': 'matrix',
            'original_gem_groups': np.array([1]),
            'software_version': '4509.5.5',
            'version': 2}

    @property
    def frame(self):
        """\
        Compute the overall frame dimensions of the Visium capture area.
        
        Returns
        -------
        tuple of (width:float, height:float)
        """
        w = self.spot_step * (self.row_range-0.5) + self.spot_diameter
        h = self.spot_step * (self.col_range-1) * math.sqrt(3)/2 + self.spot_diameter
        return w, h
    
    @property
    def spots(self):
        id = 0
        for i in range(self.col_range):
            for j in range(self.row_range):
                yield id, (i,j), self[i,j]
                id += 1
    
    def __len__(self):
        return self.row_range*self.col_range
    
    def __getitem__(self, args):
        i,j = args
        bias = i%2 * self.spot_step / 2
        r = self.spot_diameter/2
        y = r + j*self.spot_step + bias
        x = r + i*self.spot_step*math.sqrt(3)/2
        return x, y, r
    
    def __get_dataframe(self):
        return pd.DataFrame(
            [
                (id, *self.id2array(id), x, y)
                for id, _, (x,y,_) in self.spots
            ],
            columns =["id","array_row","array_col","frame_row","frame_col"]
            )

    def reset(self, diameter, step):
        self.spot_diameter = diameter
        self.spot_step = step
        self.tissue_positions =  self.__get_dataframe()
    
    def id2array(self, id:int):
        """\
        Convert a linear identifier into Visium coordinates.

        Parameters
        ----------
        The spot id : int

        Returns
        -------
        tuple of (array_row:int, array_col:int)
        """
        array_row = id//(self.row_range*2)
        array_col = 2 * (id%self.row_range) - array_row%2
        return array_row, array_col

class VisiumHDProfile:
    def __init__(self):
        self.bin_size = 2.0
        self.row_range = 3350
        self.col_range = 3350
        self.tissue_positions = self.__get_dataframe()
        self.metadata = {
            'chemistry_description': 'Visium HD v1',
            'filetype': 'matrix',
            'original_gem_groups': np.array([1]),
            'software_version': 'spaceranger-3.1.1',
            'version': 2}
    
    @property
    def frame(self):
        """\
        Compute the overall frame dimensions of the Visium HD capture area.

        Returns
        -------
        tuple of (width:float, height:float)
        """
        w = self.bin_size * self.row_range
        h = self.bin_size * self.col_range
        return w, h
    
    @property
    def bins(self):
        id = 0
        for i in range(self.row_range):
            for j in range(self.col_range):
                yield id, (i,j), self[i,j]
                id += 1

    def __len__(self):
        return self.row_range*self.col_range
    
    def __getitem__(self, args):
        i,j = args
        r = self.bin_size/2
        x = i*self.bin_size + r
        y = j*self.bin_size + r
        return x, y, r
    
    def __get_dataframe(self):
        return pd.DataFrame(
            [
                (id, i, j, x, y)
                for id, (i,j), (x,y,_) in self.bins
            ],
            columns=["id","array_row","array_col","frame_row","frame_col"],
            )

    def id2array(self, id:int):
        """\
        Convert a linear identifier into Visium HD coordinates.

        Parameters
        ----------
        The spot id : int

        Returns
        -------
        tuple of (array_row:int, array_col:int)
        """
        array_row = id//self.row_range
        array_col = id%self.row_range
        return array_row, array_col
    
    def reset(self, bin_size:float):
        """\
        Reset the bin size and update the grid ranges accordingly.

        The suggested bin size is of the form 2*n.

        Parameters
        ----------
        bin_size : float
        """
        self.row_range = int(self.row_range*self.bin_size/bin_size)
        self.col_range = int(self.col_range*self.bin_size/bin_size)
        self.bin_size = bin_size
        self.tissue_positions = self.__get_dataframe()
    
    def __get_frame_center(self, frame, mode, kwargs:Dict):
        w, h = frame
        W, H = self.frame
        if mode == "corner":
            corner = kwargs.get("corner", None)
            if corner in (0,1,2,3):
                # Define corners: 0-top-left, 1-top-right, 2-bottom-right, 3-bottom-left
                if corner == 0: return h/2, w/2
                if corner == 1: return h/2, W-w/2
                if corner == 2: return H-h/2, W-w/2
                if corner == 3: return H-h/2, w/2
            else:
                raise ValueError("Supported corner values are 0, 1, 2, 3")
        elif mode == "manual":
            center = kwargs.get("center", None)
            if isinstance(center, (tuple, list)) and len(center)==2:
                return center
            else:
                raise ValueError("For manual mode, provide a center as a tuple or list of two elements")
        elif mode == "center":
            return H/2, W/2
        elif mode == "adaptive":
            y = W/2 if w<W else w/2
            x = H/2 if h<H else h/2
            return x,y
        else:
            raise ValueError("Unsupported mode")
    
    def set_spots(self, profile:VisiumProfile, mode="adaptive", **kwargs) -> np.ndarray:
        """\
        Place the Visium spots onto the Visium HD grid and label bins according
        to the spots' coverage.

        Parameters
        ----------
        profile : VisiumProfile
        mode : str, optional
            The positioning mode for aligning the Visium profile on the HD grid. Supported modes are:
            - "center": Center the profile on the grid.
            - "corner": Align the profile at a specified corner.
            - "adaptive": (default) Adjust the placement adaptively based on the profile and grid sizes.
            - "manual": Use a user-specified center.\n
        **kwargs : dict
            Additional keyword arguments required for the chosen mode. For example:
            - For "corner" mode: provide a key ``corner`` with an integer value in {0, 1, 2, 3}.\n
                Define: 0-top-left, 1-top-right, 2-bottom-right, 3-bottom-left
            - For "manual" mode: provide a key ``center`` with a tuple or list of two numeric coordinates.

        Returns
        -------
        numpy.ndarray
            A 2D NumPy array of shape (row_range, col_range) representing the grid. Each element is:
            - 0 if the corresponding bin is not covered by any spot.
            - A positive integer (spot id + 1) if the bin is covered by a spot.

        Raises
        ------
        ValueError
            If an unsupported positioning mode is provided or if the required parameters for the
            specified mode are missing or invalid.
        """
        frame_center = self.__get_frame_center(profile.frame, mode, kwargs)
        x0 = frame_center[0]-profile.frame[1]/2
        y0 = frame_center[1]-profile.frame[0]/2
        spots = (
            (id, x+x0, y+y0, r)
            for id, _, (x, y, r) in profile.spots
        )

        # Lambda to determine the range of bin indices to check for a given coordinate
        bin_iter = lambda a: range(int((a-r)/self.bin_size),int((a+r)/self.bin_size)+2)
        d2 = lambda x,y,a,b: (x-a)*(x-a) + (y-b)*(y-b)

        spot_label_image = np.zeros((self.row_range,self.col_range))
        spot_label = np.zeros(len(self), dtype=int)
        uncovered = np.zeros(len(profile), dtype=int)
        covered = np.zeros(len(profile), dtype=int)
        
        # Iterate over all spots in the profile
        for id, x, y, r in spots:
            # Iterate over bins near the spot center
            for i, j in product(bin_iter(x), bin_iter(y)):
                bin_x, bin_y, _ = self[i,j]
                if d2(bin_x,bin_y,x,y) < r*r:
                    if i<0 or j<0 or i>self.col_range-1 or j>self.row_range-1:
                        uncovered[id] += 1
                    else:
                        spot_label_image[i,j] = id + 1
                        spot_label[i*profile.row_range+j] = id
                        covered[id] += 1
            if uncovered[id]:
                covered_rate = 100 * covered[id] / (uncovered[id]+covered[id])
                warnings.warn(f"Spot {profile.id2array(id)} cover rate: {covered_rate:0.2f}%, that {uncovered[id]:d} bins outside the grid.")

        self.tissue_positions["spot_label"] = spot_label
        profile.tissue_positions["num_bin_in_spot"] = covered
        profile.tissue_positions["num_bin_out_spot"] = uncovered
        
        return spot_label_image, frame_center

class VisiumData:
    def __init__(self, **kwargs):
        
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
    
    def load(self):
        pass

    def save(self, prefix:Path):
        (prefix/"spatial").mkdir(parents=True, exist_ok=True)
        self.locDF.to_csv(prefix/"spatial/tissue_positions.csv", index=False, header=False)
        write_10X_h5(self.adata, prefix/'filtered_feature_bc_matrix.h5', self.metadata)
        with open(prefix/"spatial/scalefactors_json.json", "w") as f:
            json.dump(self.scaleF, f, ensure_ascii=False, indent=4)

class VisiumHDData:

    def __init__(self, **kwargs):
        if kwargs.get("path", False):
            self.load(kwargs["path"], VisiumHDProfile())

    def load(self, path:Path, profile:VisiumHDProfile):
        self.path = path
        self.profile = profile

        self.locDF = pd.read_parquet(path/"spatial/tissue_positions.parquet")

        self.adata = sc.read_10x_h5(path/'filtered_feature_bc_matrix.h5')
        # self.adata.var_names_make_unique()
        with h5py.File(path/'filtered_feature_bc_matrix.h5', mode="r") as f:
            self.metadata = dict(f.attrs)
        
        with open(path/'spatial/scalefactors_json.json','r') as f:
            self.scaleF = json.load(f)
        self.bin_size = self.scaleF["bin_size_um"]

        if self.bin_size != 2.0:
            warnings.warn(f"Using data in bin size of {self.bin_size}, recommed 2 um.")
            self.profile.reset(self.bin_size)
        self.__match2profile()
    
    def set_image(self, file:Path, resolution:float):
        self.pixel_size = resolution
        self.image_raw = tifffile.imread(file)
    
    def __match2profile(self):
        order = self.profile.tissue_positions[["array_row", "array_col"]].values.tolist()
        raw_order = self.locDF[["array_row", "array_col"]].values.tolist()
        if not np.array_equal(order, raw_order):
            temp = self.locDF.set_index(["array_row", "array_col"])
            self.locDF = temp.loc[order]
            self.locDF.reset_index(inplace=True)
            # TODO match adata
    
    def __spot2image(self, profile:VisiumProfile, frame_center:Tuple[float, float]):

        x0 = frame_center[0]-self.profile.frame[1]/2
        y0 = frame_center[1]-self.profile.frame[0]/2
        binsOnFrame = self.profile.tissue_positions[["frame_row","frame_col"]].values + np.array([[x0,y0]])
        binsOnImage = self.locDF[["pxl_row_in_fullres","pxl_col_in_fullres"]].values
        transformer = AffineTransform(binsOnFrame, binsOnImage)

        spotOnFrame = profile.tissue_positions[["frame_row","frame_col"]].values
        spotsOnImage = transformer.transform_batch(spotOnFrame)
        profile.tissue_positions["barcode"] = profile.tissue_positions["id"].apply( hash_to_dna )
        profile.tissue_positions[["pixel_x", "pixel_y"]] = spotsOnImage

        return spotsOnImage

    def generate_Visium(self, profile:VisiumProfile):

        _, frame_center = self.profile.set_spots(profile)
        self.__spot2image(profile, frame_center)
        
        X_indptr = np.zeros(len(profile)+1)
        X_indices = np.zeros(0)
        X_data = np.zeros(0)
        spot_in_tissue = np.zeros(len(profile))
        
        mask_in_tissue = self.locDF["in_tissue"] == 1
        for id in range(len(profile)):
            mask_in_spot = self.profile.tissue_positions["spot_label"] == id + 1
            mask = mask_in_spot[mask_in_tissue].values
            if mask.any():
                bin_in_spot = self.adata.X[mask]
                spot_data = bin_in_spot.sum(axis=0).A1
                gene_index = np.where(spot_data>0)[0]
                X_indices = np.hstack((X_indices, gene_index))
                X_data = np.hstack((X_data, spot_data[gene_index]))
                X_indptr[id + 1] = len(X_indices) -1
                spot_in_tissue[id] = 1
            else:
                spot_in_tissue[id] = 0
            if id%100==0: print(id)
        
        tissue_positions = profile.tissue_positions[["barcode","array_row","array_col"]].copy()
        tissue_positions["x"] = np.round(profile.tissue_positions["pixel_x"].values)
        tissue_positions["y"] = np.round(profile.tissue_positions["pixel_y"].values)
        tissue_positions["in_tissue"] = spot_in_tissue
        
        X_sparse = csr_matrix((X_data, X_indices, X_indptr), shape=(len(profile), len(self.adata.var)))
        adata = AnnData(
            X=X_sparse,
            var=self.adata.var,
            obs=pd.DataFrame(index=tissue_positions["barcode"].values)
        )

        metadata = self.profile.metadata.copy()
        metadata["chemistry_description"] = self.metadata["chemistry_description"]
        scaleF = {
            "spot_diameter_fullres": (profile.spot_diameter+10)/self.scaleF["microns_per_pixel"],
            "tissue_hires_scalef": self.scaleF["tissue_hires_scalef"],
            "fiducial_diameter_fullres": self.scaleF["fiducial_diameter_fullres"],
            "tissue_lowres_scalef": self.scaleF["tissue_lowres_scalef"]
        }
        return VisiumData(
            tissue_positions = tissue_positions,
            feature_bc_matrix = adata,
            scalefactors = scaleF,
            metadata = metadata
        )

def vision_label(image):
    
    image[image>0]=255
    ii.imwrite("label.png",image.astype(np.uint8))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    parser.add_argument('--rawdata', type=str)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    save_prefix = Path(args.prefix)
    rawdata = VisiumHDData(path = Path(args.rawdata))
    result =  rawdata.generate_Visium(VisiumProfile())
    result.save(save_prefix)
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

    # a = VisiumHDProfile()
    # profile = VisiumProfile()
    # print(profile.frame,a.frame)
    # label, _ = a.set_spots(profile=profile, mode="center")
    # print(a.tissue_positions)
    # print(profile.tissue_positions)
    # # label = a.set_spots(profile=profile, mode="corner", corner=3)
    # # label = a.set_spots(profile=profile, mode="manual", center=(3000,2000))
    # vision_label(label)
    # a.set_spots()

if __name__ == "__main__":
    main()