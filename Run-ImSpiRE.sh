
prefix=$1

export patch_size=$(cat ${prefix}patch_size.txt)
export cores=$(($(grep -c ^processor /proc/cpuinfo) / 2))

ImSpiRE -i ${prefix} \
    -c filtered_feature_bc_matrix.h5 \
    -s ${prefix}image.tif \
    --Switch_Preprocess OFF \
    --ImageParam_CropSize ${patch_size} \
    --ImageParam_PatchDist ${patch_size} \
    --FeatureParam_ProcessNumber ${cores} \
    -t 'H&E' \
    -p Visium \
    -m 1 \
    -o ${prefix} \
    -n result \
    -O --Verbose