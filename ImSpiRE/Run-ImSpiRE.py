
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
    parser = argparse.ArgumentParser(
        usage="python %(prog)s [OPTIONS] /FULL/PATH/TO/WORKSPACE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
$ python %(prog)s \\
    --num_nbs 10 \\
    --color_scale 1 \\
    --dist_decay_exp 2 \\
    /path/to/workspace
"""
    )
    parser.add_argument('-m', '--Mode',
                        type=int,
                        choices=[1,2],
                        default=1,
                        help="Two types of extracted image features. When this parameter is set to 1, ImSpiRE will extract intensity and texture features of the image, which are the objective features of the image itself. When this parameter is set to 2, ImSpiRE will use CellProfiler to extract image features, which may be more biologically significant. DEFAULT: %(default)s."
                        )
    parser.add_argument('--FeatureParam_ProcessNumber',
                        type=int,
                        default=int(os.cpu_count()/2),
                        help="The number of worker processes to create when extracting texture and intensity features, which is used when `-m` is 1. DEFAULT: %(default)s."
                        )
    parser.add_argument('--FeatureParam_FeatureType',
                        type=int,
                        choices=[0,1,2],
                        default=0,
                        help="This determines which type of image features to use when `-m` is 1. 0 for both texture and intensity features, 1 for texture features only and 2 for intensity features only. DEFAULT: %(default)s."
                        )
    parser.add_argument('--FeatureParam_ClipLimit',
                        type=float,
                        default=0.01,
                        help="The clipping limit, which is used when `-m` is 1. It is normalized between 0 and 1, with higher values representing more contrast. DEFAULT: %(default)s."
                        )
    parser.add_argument('--FeatureParam_IterCount',
                        type=float,
                        default=50,
                        help="Number of iterations Grabcut image segmentation algorithm should make before returning the result. DEFAULT: %(default)s."
                        )
    parser.add_argument('--CellProfilerParam_Pipeline',
                        type=str,
                        default="Cellprofiler_Pipeline_HE.cppipe",
                        help="The path to the CellProfiler pipline. It would be better to use different piplines for H&E and IF samples. For H&E samples, `Cellprofiler_Pipeline_HE.cppipe` is recommended. For IF samples, `Cellprofiler_Pipeline_IF_C3/4.cppipe` is recommended based on the total number of channels. In the docker image, the pipelines are stored in `/root`. You can also download them from https://github.com/TongjiZhanglab/ImSpiRE to your own working directory. DEFAULT: %(default)s."
                        )

    parser.add_argument('--CellProfilerParam_KernelNumber',
                        type=int,
                        default=8,
                        help="This option specifies the number of kernel to use to run CellProfiler. DEFAULT: %(default)s."
                        )

    parser.add_argument('--OptimalTransportParam_Alpha',
                        type=float,
                        default=0.5,
                        help="The weighting parameter between physical distance network and image feature distance network, ranging from 0 to 1. For example, `--OptimalTransportParam_Alpha 0.5` means ImSpiRE will equally consider the weight of physical distance network and image feature distance network. DEFAULT: %(default)s."
                        )

    parser.add_argument('--OptimalTransportParam_Beta',
                        type=float,
                        default=0.5,
                        help="The constant interpolating parameter of Fused-gromov-Wasserstein transport ranging from 0 to 1. DEFAULT: %(default)s."
                        )

    parser.add_argument('--OptimalTransportParam_NumNeighbors',
                        type=int,
                        default=5,
                        help="The number of neighbors for nearest neighbors graph. DEFAULT: %(default)s."
                        )

    parser.add_argument('--OptimalTransportParam_Epsilon',
                        type=float,
                        default=0.001,
                        help="The entropic regularization term with value greater than 0. DEFAULT: %(default)s."
                        )

    parser.add_argument('--OptimalTransportParam_NumIterMax',
                        type=int,
                        default=10,
                        help="Max number of iterations when solving the OT problem. DEFAULT: %(default)s."
                        )
    parser.add_argument('-c', '--clean',
                        dest="clean", action="store_true",
                        help="Clean intermediate files after processing")
    parser.add_argument('workspace',
                        metavar="/FULL/PATH/TO/WORKSPACE",
                        type=str,
                        help="Required working directory containing input data generated from pipeline")

    return parser.parse_args()

def get_patch_locations(mask, step):

    rows, cols = np.indices(mask.shape)
    
    pxl_row = np.round((rows + 0.5) * step)
    pxl_col = np.round((cols + 0.5) * step)
    
    return pd.DataFrame({
        'row': rows.ravel(),
        'col': cols.ravel(),
        'pxl_row': pxl_row.ravel(),
        'pxl_col': pxl_col.ravel(),
        'in_tissue': mask.ravel().astype(int)
    })[['row', 'col', 'pxl_row', 'pxl_col', 'in_tissue']]

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
    
    mask = np.load(imspire_param.BasicParam_InputDir/"mask.npy")
    patch_locations = get_patch_locations(mask, imspire_param.ImageParam_PatchDist)
    im.patch_in_tissue = patch_locations.loc[patch_locations["in_tissue"] == 1,:]
    im.segment_patch_image(patch_in_tissue=im.patch_in_tissue, 
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

    if not imspire_param.BasicParam_Clean:
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

    if imspire_param.FeatureParam_FeatureType==0:
        spot_feature=pd.concat([spot_texture_features,spot_intensity_features],axis=1)
        patch_feature=pd.concat([patch_texture_features,patch_intensity_features],axis=1)
    elif imspire_param.FeatureParam_FeatureType==1:
        spot_feature=spot_texture_features.copy()
        patch_feature=patch_texture_features.copy()
    elif imspire_param.FeatureParam_FeatureType==2:
        spot_feature=spot_intensity_features.copy()
        patch_feature=patch_intensity_features.copy()

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

    if not imspire_param.BasicParam_Clean:
        np.save(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/SupplementaryResults/ot_M_alpha{imspire_param.OptimalTransportParam_Alpha}_beta{imspire_param.OptimalTransportParam_Beta}_epsilon{imspire_param.OptimalTransportParam_Epsilon}.npy",ot_solver.M)
        np.save(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/SupplementaryResults/ot_C1_alpha{imspire_param.OptimalTransportParam_Alpha}_beta{imspire_param.OptimalTransportParam_Beta}_epsilon{imspire_param.OptimalTransportParam_Epsilon}.npy",ot_solver.C1)
        np.save(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/SupplementaryResults/ot_C2_alpha{imspire_param.OptimalTransportParam_Alpha}_beta{imspire_param.OptimalTransportParam_Beta}_epsilon{imspire_param.OptimalTransportParam_Epsilon}.npy",ot_solver.C2)
        np.save(f"{imspire_param.BasicParam_OutputDir}/{imspire_param.BasicParam_OutputName}/SupplementaryResults/ot_T_alpha{imspire_param.OptimalTransportParam_Alpha}_beta{imspire_param.OptimalTransportParam_Beta}_epsilon{imspire_param.OptimalTransportParam_Epsilon}.npy",ot_solver.T)

if __name__ == "__main__":
    args = get_args()
    prefix = Path(args.workspace)

    imspire_param=imspire.ImSpiRE_Parameters()
    # optional parament
    imspire_param.BasicParam_Clean = args.clean
    ## Feature extraction
    imspire_param.FeatureParam_ProcessNumber=args.FeatureParam_ProcessNumber
    imspire_param.FeatureParam_FeatureType=args.FeatureParam_FeatureType
    imspire_param.FeatureParam_ClipLimit=args.FeatureParam_ClipLimit
    imspire_param.FeatureParam_IterCount=args.FeatureParam_IterCount
    ## CellProfiler Params
    imspire_param.CellProfilerParam_Pipeline=args.CellProfilerParam_Pipeline
    imspire_param.CellProfilerParam_KernelNumber=args.CellProfilerParam_KernelNumber
    ## OT Solver Params
    imspire_param.OptimalTransportParam_Alpha=args.OptimalTransportParam_Alpha
    imspire_param.OptimalTransportParam_Beta=args.OptimalTransportParam_Beta
    imspire_param.OptimalTransportParam_NumNeighbors=args.OptimalTransportParam_NumNeighbors
    imspire_param.OptimalTransportParam_Epsilon=args.OptimalTransportParam_Epsilon
    imspire_param.OptimalTransportParam_NumIterMax=args.OptimalTransportParam_NumIterMax

    # build in parament
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
    imspire_param.BasicParam_Overwriting=True
    imspire_param.BasicParam_Verbose=True

    print("Running ImSpiRE model...")
    run_imspire(imspire_param)

    if imspire_param.BasicParam_Clean :
        print("Cleaning intermediate files...")
        import shutil
        for directory  in ["FeatureResults", "ImageResults", "SupplementaryResults"]:
            try:
                shutil.rmtree(Path(prefix/f"result/{directory}"))
            except FileNotFoundError:
                continue
    print("Processing completed successfully!")