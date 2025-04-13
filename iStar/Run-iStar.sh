#!/bin/bash
set -e

# Default parameters
num_jobs=2
num_states=5
device="cuda"
cuda_id=0
iStar_home="./istar-master/"
clean=false
epochs=400

function show_help() {
    echo "Usage: $(basename "$0") [OPTIONS] /FULL/PATH/TO/WORKSPACE"
    echo
    echo "Arguments:"
    echo "  /FULL/PATH/TO/WORKSPACE    Required working directory containing input data generated form pipeline"
    echo
    echo "Options:"
    echo "  -j, --num_jobs NUM     Number of parallel jobs (default: ${num_jobs})"
    echo "  -s, --num_states NUM   Number of repeat states (default: ${num_states})"
    echo "  -e, --epochs NUM       Number of epochs at each state (default: ${epochs})"
    echo "  -d, --device DEVICE    Computation device: cuda or cpu (default: ${device})"
    echo "  -g, --GPU_id ID        Select GPU device ID (default: ${cuda_id})"
    echo "  -i, --iStar_home DIR   iStar home directory (default: ${iStar_home})"
    echo "  -c, --clean            Clean intermediate files after processing"
    echo "  -h, --help             Show this help message"
    echo
    echo "Examples:"
    echo "  $ ./$(basename "$0") \\"
    echo "      --num_jobs 5 \\"
    echo "      --num_states 5 \\"
    echo "      --epochs 400 \\"
    echo "      --device cuda \\"
    echo "      --iStar_home ./istar-master \\"
    echo "      --clean \\"
    echo "      /path/to/workspace"
    exit 0
}


# Parse command line arguments
params=$(getopt \
    -o j:s:d:g:j:i:e:ch \
    --long num_jobs:,num_states:,device:,GPU_id:,iStar_home:,epochs:,clean,help \
    -n "$(basename "$0")" -- "$@") || { show_help; exit 1; }

eval set -- "$params"

while true; do
    case "$1" in
        -j|--num_jobs)
            num_jobs="$2"
            shift 2 ;;
        -s|--num_states)
            num_states="$2"
            shift 2 ;;
        -d|--device)
            device="$2"
            shift 2 ;;
        -g|--GPU_id)
            cuda_id="$2"
            shift 2 ;;
        -i|--iStar_home)
            iStar_home="$2"
            shift 2 ;;
        -e|--epochs)
            epochs="$2"
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

# Verify iStar home directory
if [ ! -d "${iStar_home}" ]; then
    echo "Error: iStar home directory not found at ${iStar_home}"
    exit 1
fi

# Main processing
cd "${iStar_home}" || { echo "Error: Cannot cd to directory ${iStar_home}"; exit 1; }
export CUDA_VISIBLE_DEVICES=$cuda_id

echo "Starting image feature extraction..."
python extract_features.py "${prefix}/" --device="${device}"

echo "Running iStar imputation module..."
python impute.py "${prefix}/" \
    --epochs="${epochs}" \
    --device="${device}" \
    --n-states="${num_states}" \
    --n-jobs="${num_jobs}"

# Clean intermediate files if requested
if $clean; then
    echo "Cleaning intermediate files..."
    rm -rf "${prefix}states"
fi

echo "Processing completed successfully!"
