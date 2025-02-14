#!/bin/bash
set -e

prefix=$1  # e.g. data/demo/

device="cuda"  # "cuda" or "cpu"
n_genes=2000  # number of most variable genes to impute

cd ../istar/istar-master/
python rescale.py ${prefix} --image --mask
python preprocess.py ${prefix} --image --mask

# extract histology features
python extract_features.py ${prefix} --device=${device}

# predict super-resolution gene expression
# rescale coordinates and spot radius
python rescale.py ${prefix} --locs --radius

# train gene expression prediction model and predict at super-resolution
python impute.py ${prefix} --epochs=400 --device=${device}  # train model from scratch
rm -r ${prefix}states
