import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import os
import yaml
import json

import pandas as pd
import imageio

from datasets import rawData, VisiumData, VisiumHDData
from profiles import Profile, VisiumProfile, VisiumHDProfile
from models import SRtools, iStar, ImSpiRE, Xfuse, TESLA
from run_in_conda import run_command_in_conda_env


CONFIG_PATHS = [
    os.path.join(os.getcwd(), "config.yaml")
]

def load_config():
    """从配置文件中加载全局配置，支持 YAML 或 JSON"""
    for path in CONFIG_PATHS:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                if path.endswith((".yaml", ".yml")):
                    return yaml.safe_load(f)
                elif path.endswith(".json"):
                    return json.load(f)
    return {}

CONFIG = {
    'n_top_hvg': 2000,
    'min_counts': 10,
    'auto_mask': True
}

configs = load_config() or {}
global_configs:dict = configs.get("global", {})
CONDA_ENV:dict = global_configs.get(
    "conda_env_prefix",
    {
        "iStar":  "iStar",
        "xfuse":  "xfuse-cuda11.7",
        "ImSpiRE":  "imspire",
        "TESLA":  "DataReader"
    }
)
TOOL_SCRIPTS:dict = global_configs.get(
    "tool_scripts",
    {
        "iStar":  "./iStar/Run-iStar.sh",
        "xfuse":  "./xfuse/Run-xfuse.sh",
        "ImSpiRE":  "python ./ImSpiRE/Run-ImSpiRE.py --prefix",
        "TESLA":  "./TESLA/Run-TESLA.py --prefix"
    }
)

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
        Model_temp = Path(self.output_path/f"bin_{bin_size:03}um/{model_name}_workspace")
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
        visiumHD.to_anndata().write_h5ad(Model_temp/"superHD.h5ad")

def rebinning(prefix:Path, source_image_path:Path, output_path:Path, bin_size, n_top_genes=2000, min_counts=10):
    prefix = Path(prefix)
    source_image_path = Path(source_image_path)
    output_path = Path(output_path)
    visiumHD_profile_small = VisiumHDProfile(bin_size=2)
    visiumHD_profile_lagel = VisiumHDProfile(bin_size=bin_size)

    HDdata = VisiumHDData()
    HDdata.load(
        path = prefix/"square_002um",
        profile = visiumHD_profile_small,
        source_image_path = source_image_path
    )
    HDdata.select_HVG(n_top_genes=n_top_genes, min_counts=min_counts)
    if (prefix/f"square_{bin_size:03}um").exists():
        genes = HDdata.adata.var.index
        HDrebin = VisiumHDData()
        HDrebin.load(
            path = prefix/f"square_{bin_size:03}um",
            profile = visiumHD_profile_lagel,
            source_image_path = source_image_path
        )
        HDrebin.require_genes(genes=genes)
    else:
        HDrebin = HDdata.rebining(visiumHD_profile_lagel)
    outfile = output_path/f"bin_{bin_size:03}um/rawHD.h5ad"
    HDrebin.to_anndata().write_h5ad(outfile)

def main():
    metadata = pd.read_csv("benchmark/metadata.tsv", sep="\t")
    for i in range(len(metadata)):
        id = metadata.loc[i,"Sample ID"]
        rebinning(
            prefix = f"benchmark/data/{id}/binned_outputs",
            source_image_path = f"benchmark/data/{id}/{id}_tissue_image.tif",
            output_path = f"benchmark/data/{id}",
            bin_size=16
        )
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
        for model_name in ["TESLA", "iStar"]:
            benchmarker.run(
                model_name=model_name,
                mask_image_path=f"benchmark/data/{id}/mask.png",
                bin_size=16
            )

if __name__ == "__main__":
    main()