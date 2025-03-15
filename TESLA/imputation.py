import cv2
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.spatial.distance import cdist as sci_cdist
from scipy.sparse import issparse
from anndata import AnnData

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)
    return cnt_scaled

def extract_color(x_pixel, y_pixel, image, beta=49, RGB=True):
    x = np.asarray(x_pixel)
    y = np.asarray(y_pixel)
    
    beta_half = int(round(beta / 2))
    window_size = 2 * beta_half + 1
    
    dy, dx = np.mgrid[-beta_half:beta_half+1, -beta_half:beta_half+1]
    
    xx = x[:, None, None] + dx
    yy = y[:, None, None] + dy
    
    xx = np.clip(xx, 0, image.shape[0]-1)
    yy = np.clip(yy, 0, image.shape[1]-1)
    
    if RGB:
        neighborhoods = image[xx, yy, :]
    else:
        neighborhoods = image[xx, yy]
    return sum_patch(neighborhoods)


def one_barcode(pixels:np.ndarray):
    mean_colors = pixels.mean(axis=2)
    var = pixels.var(axis=2)
    weights = var / np.sum(var)
    return np.sum(mean_colors * weights)

def sum_patch(neighborhoods, index=None, RGB=True):
    if isinstance(neighborhoods, np.ndarray):
        if RGB:
            mean_colors = neighborhoods.mean(axis=(1, 2))
            var = np.var(neighborhoods, axis=(1,2))
            weights = var / var.sum(axis=1, keepdims=True)
            c3 = (mean_colors * weights).sum(axis=1)
        else:
            c3 = neighborhoods.mean(axis=(1, 2, 3))
    elif isinstance(neighborhoods, dict):
        if RGB:
            c3 = np.array(
                [one_barcode(neighborhoods[barcode]) for barcode in index]
            )
        else:
            c3 = np.array(
                [neighborhoods[barcode].mean() for barcode in index]
            )
    return c3

def imputation(
        img:np.ndarray,
        raw:AnnData,
        cnt,
        genes, 
        res=50, s=1, k=2, num_nbs=10
    ):
    binary=np.zeros((img.shape[0:2]), dtype=np.uint8)
    cv2.drawContours(binary, [cnt], -1, (1), thickness=-1)
    #Enlarged filter
    cnt_enlarged = scale_contour(cnt, 1.05)
    binary_enlarged = np.zeros(img.shape[0:2])
    cv2.drawContours(binary_enlarged, [cnt_enlarged], -1, (1), thickness=-1)
    x_max, y_max=img.shape[0], img.shape[1]
    x_list=list(range(int(res), x_max, int(res)))
    y_list=list(range(int(res), y_max, int(res)))
    x=np.repeat(x_list,len(y_list)).tolist()
    y=y_list*len(x_list)
    sudo=pd.DataFrame({"x":x, "y": y})
    sudo=sudo[sudo.index.isin([i for i in sudo.index if (binary_enlarged[sudo.x[i], sudo.y[i]]!=0)])]
    b=res
    sudo["color"]=extract_color(
        x_pixel=sudo.x.tolist(),
        y_pixel=sudo.y.tolist(),
        image=img, beta=b, RGB=True
    )
    z_scale=np.max([np.std(sudo.x), np.std(sudo.y)])*s
    sudo["z"]=(sudo["color"]-np.mean(sudo["color"]))/np.std(sudo["color"])*z_scale
    sudo=sudo.reset_index(drop=True)
    #------------------------------------Known points---------------------------------#
    known_adata = raw[:, raw.var.index.isin(genes)].copy()
    known_adata.obs["x"]=known_adata.obs["pixel_x"]
    known_adata.obs["y"]=known_adata.obs["pixel_y"]
    known_adata.obs["color"]=extract_color(
        x_pixel=known_adata.obs["pixel_x"].astype(int).tolist(),
        y_pixel=known_adata.obs["pixel_y"].astype(int).tolist(),
        image=img, beta=b, RGB=False
    )
    known_adata.obs["z"]=(known_adata.obs["color"]-np.mean(known_adata.obs["color"]))/np.std(known_adata.obs["color"])*z_scale
    imputation_sudo(sudo, known_adata, num_nbs, k)

def imputation_sudo(sudo, known_adata, num_nbs, k=None):
    #-----------------------Distance matrix between sudo and known points-------------#
    bin_cord = sudo[["x","y","z"]].values
    spot_cord = known_adata.obs[["x","y","z"]].values
    dis = sci_cdist(bin_cord, spot_cord, 'euclidean')
    dis=pd.DataFrame(dis, index=sudo.index, columns=known_adata.obs.index)
    #-------------------------Fill gene expression using nbs---------------------------#
    sudo_adata = AnnData(np.zeros((sudo.shape[0], known_adata.shape[1])))
    sudo_adata.obs = sudo
    sudo_adata.var = known_adata.var

    dis_threshold = np.quantile(dis.values.flatten(), num_nbs / known_adata.shape[0])

    n_spots = dis.shape[0]
    indices = np.argpartition(dis.values, num_nbs - 1, axis=1)[:, :num_nbs]
    rows = np.arange(n_spots)[:, None]
    distances = dis.values[rows, indices]

    mask = distances <= dis_threshold
    valid_mask = mask.any(axis=1)

    adjusted_dist = (distances + 0.1)/np.min(distances + 0.1, axis=1, keepdims=True)
    if isinstance(k, int):
        weights = 1 / (adjusted_dist ** k)
    else:
        weights = np.exp(-distances)

    # sum_weights = np.sum(weights * mask, axis=1, keepdims=True)
    sum_weights = np.sum(weights, axis=1, keepdims=True)
    sum_weights[sum_weights == 0] = 1
    # weights = (weights * mask) / sum_weights
    weights = weights / sum_weights

    # weights[~valid_mask, :] = 0
    if isinstance(known_adata.X, np.ndarray):
        # 处理密集矩阵
        neighbor_data = known_adata.X[indices]
        sudo_adata.X = np.einsum('ij,ijk->ik', weights, neighbor_data)
    else:
        print("Keep waiting.")
        from scipy.sparse import csr_matrix
        sudo_adata.X = csr_matrix((weights.shape[0], known_adata.X.shape[1]))
        for i in range(weights.shape[0]):
            if valid_mask[i]:
                neighbor_indices = indices[i]
                neighbor_data_i = known_adata.X[neighbor_indices]
                weighted_sum = weights[i] @ neighbor_data_i
                sudo_adata.X[i] = weighted_sum

    return sudo_adata

if __name__ == "__main__":
    from pathlib import Path

    prefix = Path("/home/yiriso/Research/Super-resolvedST/SuperResolvedST-Pipline/test/TESLA")
    res=27
    counts = sc.read(prefix/"data.h5ad")
    img=cv2.imread(str(prefix/"image.jpg"))
    mask=cv2.imread(str(prefix/"mask.png"))
    shape = [
        int(np.floor((img.shape[0]-res)/res)+1),
        int(np.floor((img.shape[1]-res)/res)+1)
        ]

    counts.var_names_make_unique()
    counts.raw=counts
    sc.pp.log1p(counts) # impute on log scale
    if issparse(counts.X):
        counts.X=counts.X.toarray()

    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    cnts, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    adata:AnnData = imputation(
        img = img,
        raw = counts,
        cnt=cnts[0], 
        genes=counts.var.index.tolist(), 
        res=res, 
        s=1, k=2, num_nbs=10
    )
    print(adata.to_df())
    print(adata.obs)