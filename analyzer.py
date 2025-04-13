#! /mnt/TenTA-f702/user/zhangyichi/conda-envs/SRIO/bin/python

import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from datasets import rawData, VisiumData, VisiumHDData
from profiles import Profile, VisiumProfile, VisiumHDProfile
from models import SRtools, iStar, ImSpiRE, Xfuse, TESLA
from run_in_conda import run_command_in_conda_env

import imageio.v2 as ii

import argparse
import os
import yaml
import json
from warnings import warn

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

def view_params(params:dict, indent=4):
    return "\n"+"\n".join([" "*indent+f"{k}:\t{v}" for k,v in params.items()])

# 先加载全局配置文件，提取默认值（如果配置文件不存在则使用代码预设值）
configs = load_config(DEFAULT_CONFIG_FILE)

def global_reset(configs:dict):
    global defaults, global_configs, SUPPORTED_TOOLS, SUPPORTED_OUTPUT_FORMATS
    defaults = configs.get("default", {})
    global_configs = configs.get("global", {})
    SUPPORTED_TOOLS = global_configs.get("supported_tools", [])
    SUPPORTED_OUTPUT_FORMATS = global_configs.get("supported_output_formats", [])

    global DEFAULT_FORMAT, DEFAULT_MODEL, DEFAULT_PREPROCESS, DEFAULT_POSTPROCESS, DEFAULT_SUPER_PIXEL_SIZE, DEFAULT_VISIUM_SERIAL
    DEFAULT_FORMAT = defaults.get("format", "h5ad")
    DEFAULT_MODEL = defaults.get("model", "iStar")
    DEFAULT_PREPROCESS = defaults.get("preprocess", {'n_top_hvg': -1, 'min_counts': 10, 'auto_mask': True})
    # DEFAULT_POSTPROCESS = defaults.get("postprocess", {"normalize": False})
    DEFAULT_SUPER_PIXEL_SIZE = defaults.get("super_pixel_size", 16)
    DEFAULT_VISIUM_SERIAL = defaults.get("visium_serial", 1)

global_reset(configs)

class Pipeline:
    def __init__(self, model_name:str, model_config:dict):
        self.model_name = model_name

        if model_name == 'iStar':
            self.SRmodel = iStar()
        elif model_name == 'xfuse':
            self.SRmodel = Xfuse()
        elif model_name == 'ImSpiRE':
            self.SRmodel = ImSpiRE()
        elif model_name == 'TESLA':
            self.SRmodel = TESLA()
        else:
            raise ValueError("Unsupported model")
        
        if len(model_config)==0:
            raise ValueError(f"Check {model_name} global setting in config file")

        self.Model_temp = Path(model_config.get("temp_dir",f"/tmp/SRST_Pipeline_{model_name}"))

        self.Model_env = Path(model_config.get("conda_env_prefix", None))
        if not isinstance(self.Model_env, Path):
            raise ValueError(f"Check {model_name} conda_env_prefix setting in config file")
        
        self.Model_script = Path(model_config.get("tool_script", None))
        if not isinstance(self.Model_script, Path):
            raise ValueError(f"Check {model_name} tool_script setting in config file")
        
        self.Model_params:dict = model_config.get("model_params", {})
    
    def __get_commend(self, prefix:Path):
        params = " ".join(
            f"--{k} {v}" for k,v in self.Model_params.items()
        )
        return str(self.Model_script) + " " + params + " " + str(prefix.resolve())
    
    def running(self, super_pixel_size:int):
        now = datetime.now()
        format_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        Model_dir = self.Model_temp/f"{format_time}_{super_pixel_size:03}"
        self.SRmodel.save_input(Model_dir)
        run_time = run_command_in_conda_env(
            self.Model_env, self.__get_commend(Model_dir)
        )
        self.SRmodel.update_params(run_time=run_time)
        self.SRmodel.load_output(Model_dir)

    def preprocessing(dataset:rawData|SRtools, preprocess:dict):
        # preprocessing
        
        ## select gene
        if preprocess.get("require_genes", False):
            validate_file(preprocess["require_genes"])
            with open(preprocess["require_genes"], "r") as f:
                content = f.read()
                genes = content.rstrip().split()
            dataset.require_genes(genes=genes)
        else:
            n_top_genes = int(preprocess.get("n_top_hvg", -1))
            min_counts = int(preprocess.get("min_counts", 0))
            if n_top_genes > 0 or min_counts > 0:
                dataset.select_HVG(n_top_genes=n_top_genes, min_counts=min_counts)
            else:
                warn("Haven't select genes, running under all the gene.")

        if isinstance(dataset, SRtools) and dataset.mask is None:
            ## build mask 
            if preprocess.get("mask_image_path", False):
                dataset.tissue_mask(mask_image_path=Path(preprocess["mask_image_path"]))
            elif preprocess.get("auto_mask", False):
                dataset.tissue_mask(auto_mask=True)
            else:
                warn("Haven't submit tissue mask image, running under unmask.")
                from numpy import full_like, uint8
                dataset.tissue_mask(mask=full_like(dataset.image, 255, dtype=uint8))
        
        return None
    
    def postprocessing(dataset:rawData|SRtools, postprocess:dict):
        pass
    
    def saving(dataset:rawData, format:str, output_path:Path, filename:str="superHD.h5ad"):
        if format == "raw":
            dataset.save(output_path)
        elif format == "h5ad":
            dataset.to_anndata().write_h5ad(output_path/filename)
        else:
            raise ValueError("Unsupported format")
    
    def run_Visium2HD(self,
                      input_path, source_image_path, output_path,
                      super_pixel_size, format,
                      slide_serial=DEFAULT_VISIUM_SERIAL,
                      preprocess:Dict=DEFAULT_PREPROCESS,
                    #   postprocess:Dict=DEFAULT_POSTPROCESS,
                      ):
        visium_path = Path(input_path)
        output_path = Path(output_path)

        visium_profile = VisiumProfile(slide_serial=slide_serial)
        self.SRmodel.load(
            path=visium_path,
            profile=visium_profile,
            source_image_path=Path(source_image_path)
        )
        Pipeline.preprocessing(self.SRmodel, preprocess)

        # match and build visiumHD struct
        HD_profile = VisiumHDProfile(bin_size=super_pixel_size)
        visiumHD = self.SRmodel.Visium2HD(HDprofile=HD_profile, quiet=True)
        self.SRmodel.set_target_VisiumHD(visiumHD)

        # run super resolve model
        self.running(super_pixel_size=super_pixel_size)
        self.SRmodel.to_VisiumHD(superHD_demo=visiumHD)

        # Pipeline.postprocessing()
        # save as VisiumHD
        Pipeline.saving(visiumHD, format, output_path)
    
    def run_HD2Visium(input_path, source_image_path, output_path,
                      format, bin_size,
                      slide_serial=DEFAULT_VISIUM_SERIAL,
                      preprocess:Dict=DEFAULT_PREPROCESS,
                      ):
        visiumHD_path = Path(input_path)
        output_path = Path(output_path)

        visium_profile = VisiumProfile(slide_serial=slide_serial)
        visiumHD_profile = VisiumHDProfile(bin_size=bin_size)
        HDdata = VisiumHDData()
        HDdata.load(
            path=visiumHD_path,
            profile=visiumHD_profile,
            source_image_path=source_image_path
        )

        Pipeline.preprocessing(HDdata, preprocess)
        
        emulate_visium = HDdata.HD2Visium(visium_profile)

        Pipeline.saving(emulate_visium, format, output_path)

    def run_Benchmark(self,
                      input_path, source_image_path, output_path,
                      super_pixel_size, bin_size, format, rebin,
                      slide_serial=DEFAULT_VISIUM_SERIAL,
                      preprocess:Dict=DEFAULT_PREPROCESS,
                    #   postprocess:Dict=DEFAULT_POSTPROCESS,
                      ):
        visiumHD_path = Path(input_path)
        output_path = Path(output_path)

        visium_profile = VisiumProfile(slide_serial=slide_serial)
        visiumHD_profile_small = VisiumHDProfile(bin_size=bin_size)
        visiumHD_profile_large = VisiumHDProfile(bin_size=super_pixel_size)

        ####### merge pseudo Visium ########

        HDdata = VisiumHDData()
        HDdata.load(
            path=visiumHD_path,
            profile=visiumHD_profile_small,
            source_image_path=source_image_path
        )
        Pipeline.preprocessing(HDdata, {
                'n_top_hvg': -1, 
                'min_counts': preprocess['min_counts'],
                "auto_mask": False
            }
        )
        # merging bin in spot
        emulate_visium = HDdata.HD2Visium(visium_profile)
        emulate_visium.save(output_path/"Pseudo_Visium")
        mask_image = HDdata.generate_tissue_mask_image()
        ii.imwrite(output_path/"mask.png",mask_image)
        del emulate_visium

        ####### rebin VisiumHD ########
        if rebin:
            rebinHD= HDdata.rebining(visiumHD_profile_large)
            Pipeline.saving(rebinHD, format, output_path, filename="rebinHD.h5ad")

        ####### super resolving ########
        self.SRmodel.load(
            path=output_path/"Pseudo_Visium",
            profile=visium_profile,
            source_image_path=Path(source_image_path)
        )
        ## build mask 
        self.SRmodel.tissue_mask(mask=mask_image)
        Pipeline.preprocessing(self.SRmodel, {
                'n_top_hvg': preprocess['n_top_hvg'], 
                'min_counts': preprocess['min_counts'],
                'mask_image_path': output_path/"mask.png"
            }
        )

        # match and build visiumHD struct
        center = [ i/2 for i in visiumHD_profile_small.frame]
        visiumHD = self.SRmodel.Visium2HD(HDprofile=visiumHD_profile_large, mode='manual',center=center)
        self.SRmodel.set_target_VisiumHD(visiumHD)

        # run super resolve model
        self.running(super_pixel_size=super_pixel_size)

        ####### save ########
        self.SRmodel.to_VisiumHD()
        Pipeline.saving(visiumHD, format, output_path)

def parse_key_value_pairs(pair_list, default_dict):
    """Parse parameters in key=value format and merge them with default values."""
    parsed = default_dict.copy()
    if pair_list:
        for pair in pair_list:
            try:
                key, value = pair.split("=")
                parsed[key] = value
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid parameter format: '{pair}'. Expected key=value format."
                )
    return parsed

def validate_directory(path):
    """Validate that the input directory exists."""
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Error: Directory '{path}' does not exist")
    return path

def ensure_directory(path):
    """Ensure that the output directory exists; create it if it doesn't."""
    os.makedirs(path, exist_ok=True)
    return path

def validate_file(path):
    """Validate that the file exists."""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"Error: File '{path}' does not exist")
    return path

def common_args(parser:argparse.ArgumentParser):
    """Add common arguments for all subcommands."""
    parser.add_argument("-i", "--input", required=True, type=validate_directory,
                        help="Input directory path")
    parser.add_argument("-o", "--output", required=True, type=ensure_directory,
                        help="Output directory path")
    parser.add_argument("-f", "--format", choices=SUPPORTED_OUTPUT_FORMATS,
                        default=DEFAULT_FORMAT,
                        help="Output format")
    parser.add_argument("--source_image_path", required=True, type=validate_file,
                        help="Original microscopic image file, only supported .tif")
    parser.add_argument("--config", type=validate_file,
                        default=DEFAULT_CONFIG_FILE,
                        help="The config file of this pipeline, will replace all setting")
    parser.add_argument("--visium_serial", choices=[1, 4, 5],
                        type=int,
                        default=DEFAULT_VISIUM_SERIAL,
                        help="10X Visium slide serial number \
                            https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/inputs/image-slide-parameters#slide-serial-numbers")
    parser.add_argument("--preprocess", nargs="*", 
                        default=[],
                        help=f"Preprocessing parameters, format: key=value (default: {DEFAULT_PREPROCESS})")
    # parser.add_argument("--postprocess", nargs="*", 
    #                     default=[],
    #                     help=f"Postprocessing parameters, format: key=value (default: {DEFAULT_POSTPROCESS})")

def visium2hd(args):
    preprocess_params = parse_key_value_pairs(args.preprocess, DEFAULT_PREPROCESS)
    # postprocess_params = parse_key_value_pairs(args.postprocess, DEFAULT_POSTPROCESS)
    print("Running Visium2HD")
    print(f"  Input Dir: {args.input}")
    print(f"  Output Dir: {args.output}")
    print(f"  Format: {args.format}")
    print(f"  Super Pixel Size: {args.super_pixel_size}")
    print(f"  Model: {args.model}")
    print(f"  Preprocessing: {view_params(preprocess_params)}")
    # print(f"  Postprocessing: {postprocess_params}")
    pipeline = Pipeline(model_name=args.model, model_config=global_configs.get(args.model, dict()))
    pipeline.run_Visium2HD(
        input_path=args.input,
        source_image_path=args.source_image_path,
        output_path=args.output,
        super_pixel_size=args.super_pixel_size,
        format=args.format,
        slide_serial=args.visium_serial,
        preprocess=preprocess_params,
        # postprocess=postprocess_params
    )
    
def hd2visium(args):
    preprocess_params = parse_key_value_pairs(args.preprocess, DEFAULT_PREPROCESS)
    # postprocess_params = parse_key_value_pairs(args.postprocess, DEFAULT_POSTPROCESS)
    print("Running HD2Visium")
    print(f"  Input Dir: {args.input}")
    print(f"  Output Dir: {args.output}")
    print(f"  Format: {args.format}")
    print(f"  Preprocessing: {view_params(preprocess_params)}")
    # print(f"  Postprocessing: {postprocess_params}")
    Pipeline.run_HD2Visium(
        input_path=args.input,
        source_image_path=args.source_image_path,
        output_path=args.output,
        bin_size=args.bin_size,
        format=args.format,
        slide_serial=args.visium_serial,
        preprocess=preprocess_params,
    )

def benchmark(args):
    preprocess_params = parse_key_value_pairs(args.preprocess, DEFAULT_PREPROCESS)
    # postprocess_params = parse_key_value_pairs(args.postprocess, DEFAULT_POSTPROCESS)
    print("Running Benchmark")
    print(f"  Input Dir: {args.input}")
    print(f"  Output Dir: {args.output}")
    print(f"  Super Pixel Size: {args.super_pixel_size}")
    print(f"  Model: {args.model}")
    print(f"  Rebin: {args.rebin}")
    print(f"  Preprocessing: {view_params(preprocess_params)}")
    # print(f"  Postprocessing: {postprocess_params}")
    pipeline = Pipeline(model_name=args.model, model_config=global_configs.get(args.model, dict()))
    pipeline.run_Benchmark(
        input_path=args.input,
        source_image_path=args.source_image_path,
        output_path=args.output,
        bin_size=args.bin_size,
        super_pixel_size=args.super_pixel_size,
        format=args.format,
        rebin=args.rebin,
        slide_serial=args.visium_serial,
        preprocess=preprocess_params,
        # postprocess=postprocess_params
    )

def parse_args():
    #### read user define config file
    initial_parser = argparse.ArgumentParser(add_help=False)
    initial_parser.add_argument("--config", type=validate_file, default=None)
    initial_args, _ = initial_parser.parse_known_args()

    if isinstance(initial_args.config, str):
        user_configs = load_config(initial_args.config)
        global_reset(user_configs)
    
    #### main argparse
    parser = argparse.ArgumentParser(
        description="This Pipeline is designed to integrate multiple super-resolution tools into spatial transcriptomics (ST) data analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # HD2Visium subcommand (model option not needed)
    parser_h2v = subparsers.add_parser("HD2Visium", help="Convert VisiumHD to Visium")
    common_args(parser_h2v)
    parser_h2v.add_argument("--bin_size", type=int, default=2,
                           help="Bin size for merging spots (default: 2)")
    parser_h2v.set_defaults(func=hd2visium)

    # Visium2HD subcommand: add model option and super pixel size
    parser_v2hd = subparsers.add_parser("Visium2HD", help="Super-resolution Visium to VisiumHD")
    common_args(parser_v2hd)
    parser_v2hd.add_argument("--super_pixel_size", type=int,
                             default=DEFAULT_SUPER_PIXEL_SIZE,
                             help=f"Super pixel size (default: {DEFAULT_SUPER_PIXEL_SIZE})")
    parser_v2hd.add_argument("--model", choices=SUPPORTED_TOOLS,
                             default=DEFAULT_MODEL,
                             help="Super-resolution model option")
    parser_v2hd.set_defaults(func=visium2hd)

    # Benchmark subcommand: add method, model option, and super pixel size
    parser_bm = subparsers.add_parser("Benchmark", help="Benchmark different super-resolution tools")
    common_args(parser_bm)
    parser_bm.add_argument("--bin_size", type=int, default=2,
                           help="Bin size for merging spots (default: 2)")
    parser_bm.add_argument("--no-rebin", dest="rebin", action="store_false",
                        help="Do not merge bins into super pixel size (default: True)")
    parser_bm.add_argument("--super_pixel_size", type=int,
                           default=DEFAULT_SUPER_PIXEL_SIZE,
                           help=f"Super pixel size (default: {DEFAULT_SUPER_PIXEL_SIZE})")
    parser_bm.add_argument("--model", choices=SUPPORTED_TOOLS,
                           default=DEFAULT_MODEL,
                           help="Super-resolution model option")
    parser_bm.set_defaults(func=benchmark)

    return parser.parse_args()

def main():
    args = parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

