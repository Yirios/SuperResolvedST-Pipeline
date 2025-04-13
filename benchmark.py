import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from itertools import product
import os
import yaml
import json
import multiprocessing

import pandas as pd
import imageio

from datasets import rawData, VisiumData, VisiumHDData
from profiles import Profile, VisiumProfile, VisiumHDProfile
from models import SRtools, iStar, ImSpiRE, Xfuse, TESLA
from analyzer import Pipeline
from run_in_conda import run_command_in_conda_env

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")

def load_config(config_file):
    """从配置文件中加载全局配置，支持 YAML 或 JSON"""
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            if config_file.endswith((".yaml", ".yml")):
                return yaml.safe_load(f)
            elif config_file.endswith(".json"):
                return json.load(f)
    return {}

configs = load_config(DEFAULT_CONFIG_FILE)

def global_reset(configs:dict):
    global defaults, global_configs, SUPPORTED_TOOLS, SUPPORTED_OUTPUT_FORMATS
    defaults = configs.get("default", {})
    global_configs = configs.get("global", {})
    SUPPORTED_TOOLS = global_configs.get("supported_tools", [])
    SUPPORTED_OUTPUT_FORMATS = global_configs.get("supported_output_formats", [])

    global DEFAULT_FORMAT, DEFAULT_PREPROCESS, DEFAULT_SUPER_PIXEL_SIZE, DEFAULT_VISIUM_SERIAL
    DEFAULT_FORMAT = defaults.get("format", "h5ad")
    DEFAULT_PREPROCESS = defaults.get("preprocess", {'n_top_hvg': -1, 'min_counts': 10, 'auto_mask': False})
    DEFAULT_SUPER_PIXEL_SIZE = defaults.get("super_pixel_size", 8)
    DEFAULT_VISIUM_SERIAL = defaults.get("visium_serial", 4)

global_reset(configs)

class Benchmark:
    def __init__(self, target_bin_size, source_bin_size:int=2, visium_serial:int=4):
        self.target_bin_size = target_bin_size
        self.source_bin_size = source_bin_size
        self.visium_serial = visium_serial
        self.datasets = {}
        self.models = {}
    
    def set_datasets(self, **kvargs):
        for dataset_id, dataset_config in kvargs.items():
            dataset_config["source_profile"] = VisiumHDProfile(bin_size=self.source_bin_size)
            dataset_config["target_porfile"] = VisiumHDProfile(bin_size=self.target_bin_size)
            dataset_config["visium_profile"] = VisiumProfile(slide_serial=self.visium_serial)

            dataset_config["input_path"] = Path(dataset_config["input_path"])
            dataset_config["output_path"] = Path(dataset_config["output_path"])
            dataset_config["source_image_path"] = Path(dataset_config["source_image_path"])
            self.datasets[dataset_id] = dataset_config

    def set_models(self, **kvargs):
        for model_name, model_config in kvargs.items():
            self.models[model_name] = model_config

    def preprocessHD_single(dataset_config:dict, preprocess_config:dict):

        output_path = dataset_config["output_path"]
        
        ####### merge pseudo Visium ########
        HDdata = VisiumHDData()
        HDdata.load(
            path = dataset_config["input_path"],
            profile = dataset_config["source_profile"],
            source_image_path = dataset_config["source_image_path"]
        )
        Pipeline.preprocessing(HDdata, {
                'n_top_hvg': -1, 
                'min_counts': preprocess_config['min_counts'],
                "auto_mask": False
            }
        )
        # merging bin in spot
        emulate_visium = HDdata.HD2Visium(profile=dataset_config["visium_profile"])
        emulate_visium.save(output_path/"Pseudo_Visium", save_full_image=False)
        mask_image = HDdata.generate_tissue_mask_image()
        imageio.imwrite(output_path/"mask.png", mask_image)

    def preprocessHD(self, preprocess_config:dict, num_works:int):
        self.preprocess_config = preprocess_config
        tasks = []
        for _, dataset_config in self.datasets.items():
            tasks.append((dataset_config, preprocess_config))
        with multiprocessing.Pool(processes=num_works) as pool:
            pool.starmap(Benchmark.preprocessHD_single, tasks)
        
    def running(self, num_works=1):

        tasks = []
        for dataset, model in product(self.datasets.items(), self.models.items()):
            tasks.append((self.target_bin_size, dataset, self.preprocess_config, model))
        with multiprocessing.Pool(processes=num_works) as pool:
            pool.starmap(Benchmark.run_single, tasks)

    def run_single(target_bin_size, dataset, preprocess_config, model):
        model_name, model_config = model
        dataset_id, dataset_config = dataset

        output_path = dataset_config["output_path"]
        model_config["temp_dir"] = output_path/f"bin_{target_bin_size:03}um/{model_name}_workspace"
        Visium2HD_pipeline = Pipeline(model_name, model_config)
        Visium2HD_pipeline.SRmodel.load(
            path = output_path/"Pseudo_Visium",
            profile=dataset_config["visium_profile"],
            source_image_path=dataset_config["source_image_path"]
        )
        Visium2HD_pipeline.SRmodel.tissue_mask(mask_image_path=output_path/"mask.png")
        Pipeline.preprocessing(Visium2HD_pipeline.SRmodel, {
                'n_top_hvg': preprocess_config['n_top_hvg'], 
                'min_counts': preprocess_config['min_counts'],
                'mask_image_path': output_path/"mask.png"
            }
        )
        center = [ i/2 for i in dataset_config["source_profile"].frame]
        visiumHD = Visium2HD_pipeline.SRmodel.Visium2HD(
            HDprofile=dataset_config["target_porfile"],
            mode='manual',
            center=center
        )
        Visium2HD_pipeline.SRmodel.set_target_VisiumHD(visiumHD)
        Visium2HD_pipeline.running(super_pixel_size=target_bin_size)

        Visium2HD_pipeline.SRmodel.to_VisiumHD()
        Pipeline.saving(visiumHD, "h5ad", model_config["temp_dir"])

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
    dataset_configs = {}
    for i in range(len(metadata)):
        id = metadata.loc[i,"Sample ID"]
        dataset_configs[id] = {
            "id": id,
            "input_path": f"benchmark/data/{id}/binned_outputs/square_002um",
            "output_path": f"benchmark/data/{id}",
            "source_image_path": f"benchmark/data/{id}/{id}_tissue_image.tif"
        }

    benchmarker = Benchmark(target_bin_size=8)
    benchmarker.set_datasets(**dataset_configs)

    benchmark_models = ["iStar", "TESLA"]
    model_configs = {}
    for model in benchmark_models:
        model_configs[model] = global_configs.get(model, {})
    
    benchmarker.set_models(**model_configs)        
    benchmarker.preprocessHD(preprocess_config=DEFAULT_PREPROCESS, num_works=2)
    benchmarker.running(num_works=2)

    # for model in SUPPORTED_TOOLS:
    #     model_configs[model] = global_configs.get(model, {})

    #     rebinning(
    #         prefix = f"benchmark/data/{id}/binned_outputs",
    #         source_image_path = f"benchmark/data/{id}/{id}_tissue_image.tif",
    #         output_path = f"benchmark/data/{id}",
    #         bin_size=16
    #     )
    #     benchmarker = Benchmark(
    #         input_path=f"benchmark/data/{id}/binned_outputs/square_002um",
    #         output_path=f"benchmark/data/{id}",
    #         source_image_path=f"benchmark/data/{id}/{id}_tissue_image.tif"
    #     )
    #     benchmarker.preprocess(
    #         slide_serial=4,
    #         n_top_genes=2000,
    #         min_counts=10
    #     )
    #     for model_name in ["TESLA", "iStar"]:
    #         benchmarker.run(
    #             model_name=model_name,
    #             mask_image_path=f"benchmark/data/{id}/mask.png",
    #             bin_size=16
    #         )

if __name__ == "__main__":
    main()