#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/
num_jobs=2
num_states=5
device="cuda"  # "cuda" or "cpu"

cd ../istar/istar-master/

# extract histology features
python extract_features.py ${prefix} --device=${device}

# train gene expression prediction model and predict at super-resolution
python impute.py ${prefix} --epochs=400 --device=${device} --n-states=${num_states} --n-jobs=${num_jobs} # train model from scratch
rm -r ${prefix}states
