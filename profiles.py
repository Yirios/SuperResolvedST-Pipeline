
import math
from itertools import product
from typing import List, Dict,Tuple

import numpy as np
import pandas as pd
import warnings

from utility import hash_to_dna, progress_bar


class Profile:
    LowresImage = 600
    HiresImage = [None,2000,None,None,2000,4000]
    RawColumns = ["barcode","in_tissue","array_row","array_col","pxl_row_in_fullres","pxl_col_in_fullres"]
    def __init__(self, tissue_positions:pd.DataFrame, frame:Tuple[float,float]):
        '''\
        The tissue_positions variable is a pandas DataFrame containing the following columns:
            ["id", "array_row", "array_col", "frame_row", "frame_col"]
        It is recommended that the "frame_row" and "frame_col" columns be of type float.
        '''
        self.tissue_positions = tissue_positions
        self.frame = frame

class VisiumProfile(Profile):
    SERIAL_SMALL = (1,4)
    SERIAL_LARGE = (5)
    SERIAL_CytAssist = (4,5)
    def __init__(self, slide_serial=1):
        '''\
        https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/inputs/image-slide-parameters
        '''
        self.spot_diameter = 55.0
        self.spot_step = 100.0
        if slide_serial in VisiumProfile.SERIAL_SMALL:
            # even numbers from 0 to 126 for even rows, and odd numbers from 1 to 127 for odd rows with each row (even or odd) resulting in 64 spots.
            self.row_range = 78
            self.col_range = 64
        elif slide_serial in VisiumProfile.SERIAL_LARGE:
            # even numbers from 0 to 222 for even rows, and odd numbers from 1 to 223 for odd rows with each row (even or odd) resulting in 111 spots.
            self.row_range = 128
            self.col_range = 111
        else:
            raise ValueError("Unsupported version")
        self.serial = slide_serial
        self.tissue_positions =  self.__get_dataframe()
        self.metadata = {
            'chemistry_description': f'Visium V{slide_serial} Slide',
            'filetype': 'matrix',
            'original_gem_groups': np.array([1]),
            'software_version': 'spaceranger-2.1.0',
            'version': 2}

    @property
    def frame(self):
        w = self.spot_step * (self.col_range-0.5) + self.spot_diameter
        h = self.spot_step * (self.row_range-1) * math.sqrt(3)/2 + self.spot_diameter
        return w, h
    
    @property
    def spots(self):
        id = 0
        if self.serial == 4:
            for i in range(self.row_range):
                for j in range(self.col_range):
                    yield id, (i,j), self[i,j]
                    id += 1
        elif self.serial == 1:
            for i_half in range(0,self.row_range,2):
                for j_dou in range(self.col_range*2):
                    i = i_half + j_dou%2
                    j = j_dou // 2
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
                (id, hash_to_dna(id),*self.id2array(id), x, y)
                for id, _, (x,y,_) in self.spots
            ],
            columns =["id","barcode","array_row","array_col","frame_row","frame_col"]
            )

    def reset(self, diameter, step):
        self.spot_diameter = diameter
        self.spot_step = step
        self.tissue_positions =  self.__get_dataframe()
    
    def id2array(self, id:int):
        if self.serial in VisiumProfile.SERIAL_CytAssist:
            array_row = id//self.col_range
            array_col = 2 * (id%self.col_range) + array_row%2
        elif self.serial == 1:
            array_row2, array_col = id//(self.col_range*2), id%(self.col_range*2)
            array_row = 2*array_row2 + array_col%2
        return array_row, array_col
    
    def set_bins(self, profile:"VisiumHDProfile", mode="adaptive", **kwargs) -> Tuple[np.ndarray,Tuple[float,float]]:
        frame_center = get_frame_center(self.frame, profile.frame, mode, kwargs)

class VisiumHDProfile(Profile):
    HiresImage = 6000
    def __init__(self, bin_size=2):
        self.bin_size = 2
        self.row_range = 3350
        self.col_range = 3350
        if bin_size != 2: 
            self.reset(bin_size)
        self.tissue_positions = self.__get_dataframe()
        self.metadata = {
            'chemistry_description': 'Visium HD v1',
            'filetype': 'matrix',
            'original_gem_groups': np.array([1]),
            'software_version': 'spaceranger-3.1.1',
            'version': 2}
    
    @property
    def frame(self):
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
        f = lambda i,j :"s_{:03}um_{:05}_{:05}-1".format(int(self.bin_size),i,j)
        return pd.DataFrame(
            [
                (id,f(i,j), i, j, x, y)
                for id, (i,j), (x,y,_) in self.bins
            ],
            columns=["id","barcode","array_row","array_col","frame_row","frame_col"],
            )

    def id2array(self, id:int):
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
        self.row_range = self.row_range*self.bin_size//bin_size
        self.col_range = self.col_range*self.bin_size//bin_size
        self.bin_size = bin_size
        self.tissue_positions = self.__get_dataframe()


def get_frame_center(frame_down, frame_up, mode, kwargs:Dict) -> Tuple:
    w, h = frame_up
    W, H = frame_down
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

def align_profile(HDprofile:VisiumHDProfile, profile:VisiumProfile, mode="center", quiet=False, **kwargs):
        """\
        Place the Visium spots onto the Visium HD grid and label bins according
        to the spots' coverage.

        Parameters
        ----------
        HDprofile : VisiumHDProfile
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
        quiet : Bool, optional

        Returns
        -------
        spot_label_image : np.ndarray
            A 2D numpy array of shape (self.row_range, self.col_range) representing the HD grid.
            Each element corresponds to a bin in the grid and is labeled with the associated spot ID (incremented by 1);
            a value of 0 indicates that the bin is not covered by any spot.
        frame_center : Tuple[float, float]
            A tuple (x, y) indicating the center coordinates of the frame used to align the Visium profile on the HD grid.

        Raises
        ------
        ValueError
            If an unsupported positioning mode is provided or if the required parameters for the
            specified mode are missing or invalid.
        """
        frame_center = get_frame_center(HDprofile.frame, profile.frame, mode, kwargs)
        x0 = frame_center[0]-profile.frame[1]/2
        y0 = frame_center[1]-profile.frame[0]/2

        # Lambda to determine the range of bin indices to check for a given coordinate
        bin_iter = lambda a: range(int((a-r)/HDprofile.bin_size),int((a+r)/HDprofile.bin_size)+2)
        d2 = lambda x,y,a,b: (x-a)*(x-a) + (y-b)*(y-b)

        spot_label_image = np.zeros((HDprofile.row_range,HDprofile.col_range))
        spot_label = np.zeros(len(HDprofile), dtype=int)
        uncovered = np.zeros(len(profile), dtype=int)
        covered = np.zeros(len(profile), dtype=int)
        
        # Iterate over all spots in the profile
        spots=(
            (id, x+x0, y+y0, r)
            for id, _, (x, y, r) in profile.spots
        )
        for id, x, y, r in spots:
            # Iterate over bins near the spot center
            for i, j in product(bin_iter(x), bin_iter(y)):
                bin_x, bin_y, _ = HDprofile[i,j]
                if d2(bin_x,bin_y,x,y) < r*r:
                    if i<0 or j<0 or i>HDprofile.row_range-1 or j>HDprofile.col_range-1:
                        uncovered[id] += 1
                    else:
                        spot_label_image[i,j] = id + 1
                        spot_label[i*HDprofile.col_range+j] = id
                        covered[id] += 1
            if not quiet and uncovered[id]:
                covered_rate = 100 * covered[id] / (uncovered[id]+covered[id])
                warnings.warn(f"Spot {profile.id2array(id)} cover rate: {covered_rate:0.2f}%, that {uncovered[id]:d} bins outside the grid.")

        HDprofile.tissue_positions["spot_label"] = spot_label
        profile.tissue_positions["num_bin_in_spot"] = covered
        profile.tissue_positions["num_bin_out_spot"] = uncovered
        
        return spot_label_image, frame_center