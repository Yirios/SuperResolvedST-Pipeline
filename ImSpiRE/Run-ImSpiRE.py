
import warnings
warnings.filterwarnings('ignore')
import argparse
from pathlib import Path
import os

import imspire_object as imspire
import pandas as pd
import numpy as np
import scanpy as sc
import cv2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str)
    args = parser.parse_args()
    return args

def get_patch_in_tissue(mask, step, scale_row = 1, scale_col = 1):
    PXL_ROW_MIN=int(min(np.where(mask)[0])/scale_row)
    PXL_ROW_MAX=int(max(np.where(mask)[0])/scale_row)

    PXL_COL_MIN=int(min(np.where(mask)[1])/scale_col)
    PXL_COL_MAX=int(max(np.where(mask)[1])/scale_col)

    row_list = range(PXL_ROW_MIN, PXL_ROW_MAX, step)
    col_list = range(PXL_COL_MIN, PXL_COL_MAX, step)

    len_row = len(row_list)
    len_col = len(col_list)

    patch_locations = pd.DataFrame(
        {"row": np.repeat(range(len_row), len_col, axis=0),
        "col": list(range(len_col)) * len_row,
        "pxl_row": np.repeat(row_list, len_col, axis=0),
        "pxl_col": list(col_list) * len_row,
        "in_tissue": np.repeat([0], len_row * len_col)})

    for index, row in patch_locations.iterrows():
        patch_pxl_row = row['pxl_row']
        patch_pxl_col = row['pxl_col']

        srowpos = int(patch_pxl_row * scale_row)
        scolpos = int(patch_pxl_col * scale_col)

        height, width = mask.shape[:2]
        # Determine which patches are located on the tissue
        if 0 <= srowpos < height and 0 <= scolpos < width:
            tissue = int(mask[srowpos, scolpos] == 1)
        else:
            tissue = 0

        patch_locations['in_tissue'][index] = tissue

    patch_in_tissue = patch_locations.loc[patch_locations["in_tissue"] == 1,]
    return patch_in_tissue

def run_imspire(imspire_param):
    imspire.create_folder(imspire_param.BasicParam_OutputDir,
                        imspire_param.BasicParam_OutputName,
                        imspire_param.BasicParam_Overwriting)


    imdata=imspire.ImSpiRE_Data()
    if imspire_param.BasicParam_PlatForm=="Visium":
        imdata.read_10x_visium(imspire_param.BasicParam_InputDir, count_file=imspire_param.BasicParam_InputCountFile)
    else:
        imdata.read_ST(imspire_param.BasicParam_InputDir, count_file=imspire_param.BasicParam_InputCountFile)


    im=imspire.ImSpiRE_HE_Image(imspire_param.BasicParam_InputImageFile,
                                imspire_param.BasicParam_PlatForm,
                                imspire_param.BasicParam_OutputDir,
                                imspire_param.BasicParam_OutputName,
                                imspire_param.FeatureParam_IterCount)



    spot_image_output_path=f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/ImageResults/SpotImage"
    im.segment_spot_image(pos_in_tissue_filter=imdata.pos_in_tissue_filter,
                        output_path=spot_image_output_path,
                        crop_size=imspire_param.ImageParam_CropSize)



    patch_image_output_path=f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/ImageResults/PatchImage"
    im.generate_patch_locations_2(pos_in_tissue=imdata.pos_in_tissue_filter,
                                dist=imspire_param.ImageParam_PatchDist)
    mask = np.load(imspire_param.BasicParam_InputDir/"mask.npy")
    im.segment_patch_image(patch_in_tissue=get_patch_in_tissue(mask, imspire_param.ImageParam_PatchDist), 
                        output_path=patch_image_output_path, 
                        crop_size=imspire_param.ImageParam_CropSize)

    feature_set=["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]

    if imspire_param.BasicParam_Verbose:
        print("Extracting features of spot images...")

    spot_ife = imspire.ImSpiRE_Image_Feature(imspire_param.BasicParam_InputImageFile,
                                        imspire_param.FeatureParam_ClipLimit,
                                        imspire_param.FeatureParam_ProcessNumber)
    spot_ife.image_preprocess()
    spot_ife.run_extract_features(processed_image=spot_ife.processed_image,
                                    feature_set=feature_set,
                                    img_meta=imdata.pos_in_tissue_filter[['pxl_row_in_fullres','pxl_col_in_fullres']],
                                    crop_size=imspire_param.ImageParam_CropSize)

    if imspire_param.BasicParam_Verbose:
        print("Extracting features of patch images...")

    patch_ife = imspire.ImSpiRE_Image_Feature(imspire_param.BasicParam_InputImageFile,
                                        imspire_param.FeatureParam_ClipLimit,
                                        imspire_param.FeatureParam_ProcessNumber)
    patch_ife.run_extract_features(processed_image=spot_ife.processed_image,
                                    feature_set=feature_set,
                                    img_meta=im.patch_in_tissue[['pxl_row','pxl_col']],
                                    crop_size=imspire_param.ImageParam_CropSize)

    spot_texture_features=spot_ife.texture_features.loc[imdata.pos_in_tissue_filter.index,]
    spot_intensity_features=spot_ife.intensity_features.loc[imdata.pos_in_tissue_filter.index,]

    patch_ife.texture_features.index=patch_ife.texture_features.index.astype('int')
    patch_ife.intensity_features.index=patch_ife.intensity_features.index.astype('int')
    patch_texture_features=patch_ife.texture_features.sort_index()
    patch_intensity_features=patch_ife.intensity_features.sort_index()
    patch_texture_features.index=list(range(patch_texture_features.shape[0]))
    patch_intensity_features.index=list(range(patch_intensity_features.shape[0]))

    spot_features=pd.concat([spot_texture_features,spot_intensity_features],axis=1)
    patch_features=pd.concat([patch_texture_features,patch_intensity_features],axis=1)

    spot_feature_output_path=f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/FeatureResults/SpotFeature"
    patch_feature_output_path=f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/FeatureResults/PatchFeature"

    spot_texture_features.to_csv(f"{spot_feature_output_path}/spot_texture_features.txt", sep = "\t")
    spot_intensity_features.to_csv(f"{spot_feature_output_path}/spot_intensity_features.txt", sep = "\t")
    spot_features.to_csv(f"{spot_feature_output_path}/spot_features.txt", sep = "\t")

    patch_texture_features.to_csv(f"{patch_feature_output_path}/patch_texture_features.txt", sep = "\t")
    patch_intensity_features.to_csv(f"{patch_feature_output_path}/patch_intensity_features.txt", sep = "\t")
    patch_features.to_csv(f"{patch_feature_output_path}/patch_features.txt", sep = "\t")


    spot_locations=np.array(imdata.pos_in_tissue_filter.loc[:,["pxl_row_in_fullres","pxl_col_in_fullres"]])
    patch_locations=np.array(im.patch_in_tissue.loc[:,["pxl_row","pxl_col"]])


    spot_texture=pd.read_csv(f"{spot_feature_output_path}/spot_texture_features.txt",sep="\t",index_col=0)
    spot_intensity=pd.read_csv(f"{spot_feature_output_path}/spot_intensity_features.txt",sep="\t",index_col=0)

    patch_texture=pd.read_csv(f"{patch_feature_output_path}/patch_texture_features.txt",sep="\t",index_col=0)
    patch_intensity=pd.read_csv(f"{patch_feature_output_path}/patch_intensity_features.txt",sep="\t",index_col=0)

    if imspire_param.FeatureParam_FeatureType==0:
        spot_feature=pd.concat([spot_texture,spot_intensity],axis=1)
        patch_feature=pd.concat([patch_texture,patch_intensity],axis=1)
    elif imspire_param.FeatureParam_FeatureType==1:
        spot_feature=spot_texture.copy()
        patch_feature=patch_texture.copy()
    elif imspire_param.FeatureParam_FeatureType==2:
        spot_feature=spot_intensity.copy()
        patch_feature=patch_intensity.copy()

    spot_feature=spot_feature.dropna(axis=1)
    patch_feature=patch_feature.dropna(axis=1)
    commom_feature=list(set(spot_feature.columns).intersection(set(patch_feature.columns)))
    spot_feature = spot_feature.loc[:,commom_feature]
    patch_feature = patch_feature.loc[:,commom_feature]

    spot_feature=np.array(spot_feature.loc[imdata.pos_in_tissue_filter.index,])
    patch_feature=np.array(patch_feature.sort_index())




    exp_data=imdata.adata.to_df()
    exp_data=exp_data.loc[imdata.pos_in_tissue_filter.index,]



    ot_solver=imspire.ImSpiRE_OT_Solver(spot_locations,patch_locations,
                                        spot_image_features=spot_feature,
                                        patch_image_features=patch_feature,
                                        spot_gene_expression=exp_data,
                                        random_state=imspire_param.BasicParam_RandomState)

    ot_solver.setup_cost_matrices(alpha=imspire_param.OptimalTransportParam_Alpha,
                                num_neighbors=imspire_param.OptimalTransportParam_NumNeighbors)
    ot_solver.solve_OT(beta=imspire_param.OptimalTransportParam_Beta, 
                    epsilon=imspire_param.OptimalTransportParam_Epsilon,
                    numItermax=imspire_param.OptimalTransportParam_NumIterMax,
                    verbose=imspire_param.BasicParam_Verbose)
    exp_data_hr=imspire.compute_high_resolution_expression_profiles(exp_data,ot_solver.T)

    im.patch_in_tissue.index=list(range(im.patch_in_tissue.shape[0]))
    im.patch_in_tissue.to_csv(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/{imspire_param.BasicParam_OutputName}_PatchLocations.txt", sep = "\t")

    adata_hr=sc.AnnData(exp_data_hr)
    adata_hr.write_h5ad(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/{imspire_param.BasicParam_OutputName}_ResolutionEnhancementResult.h5ad")

    np.save(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/SupplementaryResults/ot_M_alpha{imspire_param.OptimalTransportParam_Alpha}_beta{imspire_param.OptimalTransportParam_Beta}_epsilon{imspire_param.OptimalTransportParam_Epsilon}.npy",ot_solver.M)
    np.save(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/SupplementaryResults/ot_C1_alpha{imspire_param.OptimalTransportParam_Alpha}_beta{imspire_param.OptimalTransportParam_Beta}_epsilon{imspire_param.OptimalTransportParam_Epsilon}.npy",ot_solver.C1)
    np.save(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/SupplementaryResults/ot_C2_alpha{imspire_param.OptimalTransportParam_Alpha}_beta{imspire_param.OptimalTransportParam_Beta}_epsilon{imspire_param.OptimalTransportParam_Epsilon}.npy",ot_solver.C2)
    np.save(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/SupplementaryResults/ot_T_alpha{imspire_param.OptimalTransportParam_Alpha}_beta{imspire_param.OptimalTransportParam_Beta}_epsilon{imspire_param.OptimalTransportParam_Epsilon}.npy",ot_solver.T)

if __name__ == "__main__":
    args = get_args()
    prefix = Path(args.prefix)

    imspire_param=imspire.ImSpiRE_Parameters()
    imspire_param.BasicParam_InputCountFile="filtered_feature_bc_matrix.h5"
    imspire_param.BasicParam_InputDir=prefix
    imspire_param.BasicParam_InputImageFile=prefix/"image.tif"
    imspire_param.BasicParam_PlatForm="Visium"
    imspire_param.BasicParam_OutputDir=prefix
    imspire_param.BasicParam_OutputName="result"
    with open(prefix/"patch_size.txt", "r") as f:
        step = int(f.read().strip())
        imspire_param.ImageParam_CropSize=step
        imspire_param.ImageParam_PatchDist=step
    imspire_param.FeatureParam_ProcessNumber=int(os.cpu_count()/2)
    imspire_param.BasicParam_Overwriting=True
    imspire_param.BasicParam_Verbose=True
    run_imspire(imspire_param)