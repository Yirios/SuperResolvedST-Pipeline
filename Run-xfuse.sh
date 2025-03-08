#!/bin/bash

prefix=$1
config_file="xfuse/my-config.toml"
device=0 # 选择 GPU [0,1,2,3] 

json_file="${prefix}/super_resolution_config.json"

mode=$(grep -oP '"mode":\s*"\K[^"]+' "$json_file")

cp ${config_file} ${prefix}config.toml
sed -i "/data = \"section1\/data.h5\"/s|data = \"section1/data.h5\"|data = \"${prefix}data/data.h5\"|" ${prefix}config.toml
sed -i "s|device = 0|device = $device|" ${prefix}config.toml

if [ "$mode" == "VisiumHD" ]; then
    echo "Mode is VisiumHD"
elif [ "$mode" == "Image" ]; then
    echo "Mode is Image"
    export scale=$(cat ${prefix}scale.txt)
    xfuse convert visium \
        --image ${prefix}image.png \
        --bc-matrix ${prefix}filtered_feature_bc_matrix.h5 \
        --tissue-positions ${prefix}tissue_positions_list.csv \
        --scale-factors ${prefix}scalefactors_json.json \
        --scale ${scale} \
        --no-rotate \
        --mask-file ${prefix}mask.png \
        --save-path ${prefix}data
else
    echo "Unknown mode: $mode"
fi

xfuse run --save-path ${prefix}result ${prefix}config.toml

