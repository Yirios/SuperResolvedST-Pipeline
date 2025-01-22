
prefix=$1
config_file="xfuse/my-config.toml"
pixel_size=8 # 超分辨后的像素边长（微米）
device=0 # 选择 GPU [0,1,2,3] 

# D=$(jq .spot_diameter_fullres ${prefix}/scalefactors_json.json)
# scale_factor=$(jq .tissue_hires_scalef ${prefix}/scalefactors_json.json)
# raw_size=$(echo "scale=0; 2000 / ${scale_factor}" | bc -l)
# # scale=$(echo "65 / ${pixel_size} / ${D}" | bc -l)
# scale=$(echo "4 * ${scale_factor} / ${pixel_size}" | bc -l)
# scale_size=$(echo "scale=0; ${raw_size} * ${scale}" | bc -l)

export scale=$(cat ${prefix}scale.txt)

cp ${config_file} ${prefix}config.toml
sed -i "/data = \"section1\/data.h5\"/s|data = \"section1/data.h5\"|data = \"${prefix}data/data.h5\"|" ${prefix}config.toml
sed -i "s|device = 0|device = $device|" ${prefix}config.toml

xfuse convert visium \
    --image ${prefix}image.png \
    --bc-matrix ${prefix}filtered_feature_bc_matrix.h5 \
    --tissue-positions ${prefix}tissue_positions_list.csv \
    --scale-factors ${prefix}scalefactors_json.json \
    --scale ${scale} \
    --no-rotate \
    --mask-file ${prefix}mask.png \
    --save-path ${prefix}data \

xfuse run --save-path ${prefix}result ${prefix}config.toml

