#!/bin/bash
set -e

config_file="$(dirname $(readlink -f "$0"))/my-config.toml"
patch_size=768
# Default parameters
cuda_id=0
batch_size=3
epochs=200000
learning_rate=0.0003

function show_help() {
    echo "Usage: $(basename "$0") [OPTIONS] /FULL/PATH/TO/WORKSPACE"
    echo
    echo "Arguments:"
    echo "  /FULL/PATH/TO/WORKSPACE        Required working directory containing input data generated from pipeline"
    echo
    echo "Options:"
    echo "  -g, --GPU_id ID          Select GPU device ID (default: ${cuda_id})"
    echo "  -e, --epochs NUM         Set number of training epochs (default: ${epochs})"
    echo "  -b, --batch_size SIZE    Set batch size (default: ${batch_size})"
    # echo "  -p, --patch_size SIZE    Set patch size (default: ${patch_size})"
    echo "  -r, --learning_rate LR   Set learning rate (default: ${learning_rate})"
    echo "  -c, --clean              Clean intermediate files after processing"
    echo "  -h, --help               Show this help message"
    echo
    echo "Examples:"
    echo "  $ ./$(basename "$0") \\"
    echo "      --epochs 200000 \\"
    echo "      --batch_size 3 \\"
    echo "      --learning_rate 0.00003 \\"
    # echo "      --patch_size 768 \\"
    echo "      --GPU_id 0 \\"
    echo "      --clean \\"
    echo "      /path/to/workspace"
    exit 0
}

# Parse command line arguments
params=$(getopt \
    -o g:e:b:r:ch \
    --long GPU_id:,epochs:,batch_size:,learning_rate:,clean,help \
    -n "$(basename "$0")" -- "$@") || { show_help; exit 1; }

eval set -- "$params"

while true; do
    case "$1" in
        -g|--GPU_id)
            cuda_id="$2"
            shift 2 ;;
        -e|--epochs)
            epochs="$2"
            shift 2 ;;
        -b|--batch_size)
            batch_size="$2"
            shift 2 ;;
        -r|--learning_rate)
            learning_rate="$2"
            shift 2 ;;
        -c|--clean)
            clean=true
            shift ;;
        -h|--help)
            show_help ;;
        --)
            shift
            break ;;
        *)
            echo "Parameter error! Use --help for usage information"
            exit 1 ;;
    esac
done

# Validate required argument
if [ $# -ne 1 ]; then
    echo "Error: Missing required workspace path"
    show_help
fi

prefix="$1"

cp ${config_file} ${prefix}/config.toml
sed -i "/data = \"section1\/data.h5\"/s|data = \"section1/data.h5\"|data = \"${prefix}/data/data.h5\"|" ${prefix}/config.toml
export CUDA_VISIBLE_DEVICES=$cuda_id
sed -i "s|batch_size =|batch_size = $batch_size|" ${prefix}/config.toml
sed -i "s|epochs =|epochs = $epochs|" ${prefix}/config.toml
sed -i "s|learning_rate =|learning_rate = $learning_rate|" ${prefix}/config.toml
sed -i "s|patch_size =|patch_size = $patch_size|" ${prefix}/config.toml

json_file="${prefix}/super_resolution_config.json"
mode=$(grep -oP '"mode":\s*"\K[^"]+' "$json_file")

if [ "$mode" == "VisiumHD" ]; then
    echo "Mode is VisiumHD"
elif [ "$mode" == "Image" ]; then
    echo "Mode is Image"
    echo "Starting xfuse build-in convert..."
    export scale=$(cat ${prefix}scale.txt)
    xfuse convert visium \
        --image ${prefix}/image.png \
        --bc-matrix ${prefix}/filtered_feature_bc_matrix.h5 \
        --tissue-positions ${prefix}/tissue_positions_list.csv \
        --scale-factors ${prefix}/scalefactors_json.json \
        --scale ${scale} \
        --no-rotate \
        --mask-file ${prefix}/mask.png \
        --save-path ${prefix}/data
else
    echo "Unknown mode: $mode"
fi

echo "Running xfuse model..."
xfuse run --save-path ${prefix}/result ${prefix}/config.toml

# Clean intermediate files if requested
if $clean; then
    echo "Cleaning intermediate files..."
    rm -rf "${prefix}/result/stats"
    rm -rf "${prefix}/result/checkpoints"
fi

echo "Processing completed successfully!"