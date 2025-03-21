import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

import pandas as pd
import imageio

from datasets import rawData, VisiumData, VisiumHDData
from profiles import Profile, VisiumProfile, VisiumHDProfile
from models import SRtools, iStar, ImSpiRE, Xfuse, TESLA
from run_in_conda import run_command_in_conda_env


CONFIG = {
    'n_top_hvg': 2000,
    'min_counts': 10,
    'auto_mask': True
}

CONDA_ENV = {
    "iStar":  "iStar",
    "xfuse":  "xfuse-cuda11.7",
    "ImSpiRE":  "imspire",
    "TESLA":  "DataReader",
}
TEMP_DIR =  {
    "iStar":  "./iStar/temp",
    "xfuse":  "../xfuse/temp",
    "ImSpiRE":  "./ImSpiRE/temp",
    "TESLA":  "./TESLA/temp",
}
TOOL_SCRIPTS = {
    "iStar":  "./Run-iStar.sh",
    "xfuse":  "./Run-xfuse.sh",
    "ImSpiRE":  "./Run-ImSpiRE.sh",
    "TESLA":  "./Run-TESLA.py --prefix",
}

class Benchmark :
    def __init__(self, input_path, output_path, source_image_path):
        self.visiumHD_path = Path(input_path)
        self.output_path = Path(output_path)
        self.source_image_path = Path(source_image_path)

    def preprocess(self, slide_serial=4, n_top_genes=2000, min_counts=10):

        self.visium_profile = VisiumProfile(slide_serial=slide_serial)
        self.visiumHD_profile_small = VisiumHDProfile(bin_size=2)

        ####### merge pseudo Visium ########
        HDdata = VisiumHDData()
        HDdata.load(
            path=self.visiumHD_path,
            profile=self.visiumHD_profile_small,
            source_image_path=self.source_image_path
        )
        
        # select gene
        HDdata.select_HVG(n_top_genes=n_top_genes, min_counts=min_counts)
        
        # merging bin in spot
        emulate_visium = HDdata.HD2Visium(self.visium_profile)
        # emulate_visium.save(self.output_path/"Pseudo_Visium")
        # mask_image = HDdata.generate_tissue_mask_image()
        # imageio.imwrite(self.output_path/"mask.png", mask_image)

    
    def run(self, model_name, mask_image_path, bin_size):

        if model_name == 'iStar':
            model = iStar()
        elif model_name == 'xfuse':
            model = Xfuse()
        elif model_name == 'ImSpiRE':
            model = ImSpiRE()
        elif model_name == 'TESLA':
            model = TESLA()
        else:
            ValueError()
        
        self.visiumHD_profile_large = VisiumHDProfile(bin_size=bin_size)

        ####### super resolving ########
        model.load(
            path=self.output_path/"Pseudo_Visium",
            profile=self.visium_profile,
            source_image_path=Path(self.source_image_path)
        )
        
        # preprocessing
        ## build mask 
        model.tissue_mask(mask_image_path=mask_image_path)
        # match and build visiumHD struct
        center = [ i/2 for i in self.visiumHD_profile_small.frame]
        visiumHD = model.Visium2HD(HDprofile=self.visiumHD_profile_large, mode='manual',center=center)
        model.set_target_VisiumHD(visiumHD)

        # run super resolve model
        Model_temp = Path(TEMP_DIR[model_name])
        now = datetime.now()
        format_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        Model_dir = Model_temp/f"{format_time}_{bin_size:03}"
        model.save_input(Model_dir)
        run_time = run_command_in_conda_env(
            CONDA_ENV[model_name],
            f'{TOOL_SCRIPTS[model_name]} {Model_dir.resolve()}/',
        )
        model.update_params(run_time=run_time)
        model.load_output(Model_dir)

        ####### save ########
        model.to_VisiumHD()
        visiumHD.to_anndata().write_h5ad(self.output_path/f"{model_name}_superHD.h5ad")

def main():
    metadata = pd.read_csv("benchmark/metadata.tsv", sep="\t")
    for i in range(len(metadata)):
        id = metadata.loc[i,"Sample ID"]
        benchmarker = Benchmark(
            input_path=f"benchmark/data/{id}/binned_outputs/square_002um",
            output_path=f"benchmark/data/{id}",
            source_image_path=f"benchmark/data/{id}/{id}_tissue_image.tif"
        )
        benchmarker.preprocess(
            slide_serial=4,
            n_top_genes=2000,
            min_counts=10
        )
        for model_name in ["iStar", "ImSpiRE", "TESLA"]:
            benchmarker.run(
                model_name=model_name,
                mask_image_path=f"benchmark/data/{id}/mask.png",
                bin_size=16
            )
    

if __name__ == "__main__":
    main()