
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime

from anndata import AnnData

from datasets import rawData, VisiumData, VisiumHDData
from profiles import Profile, VisiumProfile, VisiumHDProfile
from models import iStar, ImSpiRE, Xfuse, TESLA
from run_in_conda import run_command_in_conda_env


import argparse
import os
import yaml
import json

# 配置文件可能的路径
CONFIG_PATHS = [
    os.path.join(os.getcwd(), "config.yaml"),        # 当前目录下
]
OUTPUT_FORMAT = ["raw", "h5ad"]

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

# 先加载全局配置文件，提取默认值（如果配置文件不存在则使用代码预设值）
configs = load_config() or {}
defaults = configs.get("default", {})
global_configs = configs.get("global", {})
SUPPORTED_TOOLS = global_configs.get("supported_tools", [])
CONDA_ENV = global_configs.get("conda_env_prefix", {})
TEMP_DIR = global_configs.get("temp_dir", {})
TOOL_SCRIPTS = global_configs.get("tool_scripts", {})

# 根据配置文件设置各参数的默认值
DEFAULT_FORMAT = defaults.get("format", "h5ad")
DEFAULT_MODEL = defaults.get("model", "iStar")
DEFAULT_PREPROCESS = defaults.get("preprocess", {'n_top_hvg': 2000, 'min_counts': 10, 'auto_mask': True})
DEFAULT_POSTPROCESS = defaults.get("postprocess", {"normalize": False})
DEFAULT_SUPER_PIXEL_SIZE = defaults.get("super_pixel_size", 16)
DEFAULT_VISIUM_SERIAL = defaults.get("visium_serial", 1)

class Pipeline:
    def __init__(self, model_name):
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
        
        self.Model_temp = Path(TEMP_DIR.get(model_name,f"/tmp/SRST_Pipeline_{model_name}"))

    def run_Visium2HD(self,
                      input_path, source_image_path, output_path,
                      super_pixel_size, format,
                      slide_serial=DEFAULT_VISIUM_SERIAL,
                      preprocess:Dict=DEFAULT_PREPROCESS,
                      postprocess:Dict=DEFAULT_POSTPROCESS,
                      ):
        visium_path = Path(input_path)
        output_path = Path(output_path)

        visium_profile = VisiumProfile(slide_serial=slide_serial)
        self.SRmodel.load(
            path=visium_path,
            profile=visium_profile,
            source_image_path=Path(source_image_path)
        )
        
        # preprocessing
        ## build mask 
        if preprocess.get("mask_image", False):
            self.SRmodel.tissue_mask(mask_image_path=preprocess["mask_image"])
        elif preprocess.get("auto_mask", False):
            mask_image = self.SRmodel.tissue_mask(auto_mask=True)
            import imageio
            imageio.imwrite(output_path/"auto_mask.png", mask_image)
        ## select gene
        if preprocess.get("require_genes", False):
            validate_file(preprocess["require_genes"])
            with open(preprocess["require_genes"], "r") as f:
                content = f.read()
                genes = content.rstrip().split()
                print(genes)
            self.SRmodel.require_genes(genes=genes)
        elif preprocess.get("n_top_hvg", False):
            self.SRmodel.select_HVG(n_top_genes=preprocess["n_top_hvg"])

        # match and build visiumHD struct
        HD_profile = VisiumHDProfile(bin_size=super_pixel_size)
        visiumHD = self.SRmodel.Visium2HD(HDprofile=HD_profile, quiet=True)
        self.SRmodel.set_target_VisiumHD(visiumHD)

        # run super resolve model
        now = datetime.now()
        format_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        Model_dir = self.Model_temp/f"{format_time}_{super_pixel_size}"
        self.SRmodel.save_input(Model_dir)
        run_command_in_conda_env(
            CONDA_ENV[self.model_name],
            f'{TOOL_SCRIPTS[self.model_name]} {Model_dir.resolve()}/',
            f'{Model_dir.resolve()}/{self.model_name}.log'
        )
        self.SRmodel.load_output(Model_dir)
        
        # save as VisiumHD
        self.SRmodel.to_VisiumHD()
        if format == "raw":
            visiumHD.save(output_path)
        elif format == "csv":
            self.SRmodel.to_csv(output_path/"superHD.csv")
        elif format == "h5ad":
            visiumHD.to_anndata().write_h5ad(output_path/"superHD.h5ad")
        else :
            raise ValueError("Unsupported format")
    
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
        
        # preprocessing
        ## select gene
        if preprocess.get("require_genes", False):
            validate_file(preprocess["require_genes"])
            with open(preprocess["require_genes"], "r") as f:
                content = f.read()
                genes = content.rstrip().split()
                print(genes)
            HDdata.require_genes(genes=genes)
        elif preprocess.get("n_top_hvg", False):
            HDdata.select_HVG(n_top_genes=preprocess["n_top_hvg"])
        
        # merging bin in spot
        emulate_visium = HDdata.HD2Visium(visium_profile)

        # save as Visium
        if format == "raw":
            emulate_visium.save(output_path)
        elif format == "h5ad":
            emulate_visium.to_anndata().write_h5ad(output_path/"emulate_visium.h5ad")
        else :
            raise ValueError("Unsupported format")

    def run_Benchmark(self,
                      input_path, source_image_path, output_path,
                      super_pixel_size, bin_size, format, rebin,
                      slide_serial=DEFAULT_VISIUM_SERIAL,
                      preprocess:Dict=DEFAULT_PREPROCESS,
                      postprocess:Dict=DEFAULT_POSTPROCESS,
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
        
        # select gene
        if preprocess.get("require_genes", False):
            validate_file(preprocess["require_genes"])
            with open(preprocess["require_genes"], "r") as f:
                content = f.read()
                genes = content.rstrip().split()
                print(genes)
            HDdata.require_genes(genes=genes)
        elif preprocess.get("n_top_hvg", False):
            HDdata.select_HVG(n_top_genes=preprocess["n_top_hvg"])
        
        # merging bin in spot
        emulate_visium = HDdata.HD2Visium(visium_profile)
        emulate_visium.save(output_path/"Pseudo_Visium")
        del emulate_visium

        ####### rebin VisiumHD ########
        if rebin:
            rebinHD= HDdata.rebining(visiumHD_profile_large)

        ####### super resolving ########
        self.SRmodel.load(
            path=output_path/"Pseudo_Visium",
            profile=visium_profile,
            source_image_path=Path(source_image_path)
        )
        
        # preprocessing
        ## build mask 
        if preprocess.get("mask_image", False):
            self.SRmodel.tissue_mask(mask_image_path=preprocess["mask_image"])
        elif preprocess.get("auto_mask", False):
            mask_image = self.SRmodel.tissue_mask(auto_mask=True)
            import imageio
            imageio.imwrite(output_path/"auto_mask.png", mask_image)

        # match and build visiumHD struct
        center = [ i/2 for i in visiumHD_profile_small.frame]
        visiumHD = self.SRmodel.Visium2HD(HDprofile=visiumHD_profile_large, mode='manual',center=center)
        self.SRmodel.set_target_VisiumHD(visiumHD)

        # run super resolve model
        now = datetime.now()
        format_time = now.strftime("%Y-%m-%d_%H-%M-%S")
        Model_dir = self.Model_temp/f"{format_time}_{super_pixel_size}"
        self.SRmodel.save_input(Model_dir)
        run_command_in_conda_env(
            CONDA_ENV[self.model_name],
            f'{TOOL_SCRIPTS[self.model_name]} {Model_dir.resolve()}/',
            f'{Model_dir.resolve()}/{self.model_name}.log'
        )
        self.SRmodel.load_output(Model_dir)

        ####### save ########
        self.SRmodel.to_VisiumHD()
        if format == "raw":
            visiumHD.save(output_path/"superHD")
            if rebin:
                rebinHD.save(output_path/"rebinHD")
        elif format == "h5ad":
            visiumHD.to_anndata().write_h5ad(output_path/"superHD.h5ad")
            if rebin:
                rebinHD.to_anndata().write_h5ad(output_path/"rebinHD.h5ad")
        else :
            raise ValueError("Unsupported format")

def parse_key_value_pairs(pair_list, default_dict):
    """解析 key=value 格式的参数，并将其与默认值合并"""
    parsed = default_dict.copy()
    if pair_list:
        for pair in pair_list:
            try:
                key, value = pair.split("=")
                parsed[key] = value
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"参数格式错误: '{pair}'，应为 key=value 格式"
                )
    return parsed

def validate_directory(path):
    """验证输入目录是否存在"""
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"错误: 目录 '{path}' 不存在")
    return path

def ensure_directory(path):
    """确保输出目录存在，不存在则自动创建"""
    os.makedirs(path, exist_ok=True)
    return path

def validate_file(path):
    """验证文件是否存在"""
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"错误: 文件 '{path}' 不存在")
    return path

def common_args(parser):
    """为所有子命令添加通用参数"""
    parser.add_argument("-i", "--input", required=True, type=validate_directory,
                        help="输入文件夹路径")
    parser.add_argument("-o", "--output", required=True, type=ensure_directory,
                        help="输出文件夹路径")
    parser.add_argument("-f", "--format", choices=OUTPUT_FORMAT,
                        default=DEFAULT_FORMAT,
                        help="保存格式")
    parser.add_argument("--source_image_path", required=True, type=validate_file,
                        help="原始显微图像文件")
    parser.add_argument("--visium_serial", choices=[1, 4, 5],
                        default=DEFAULT_VISIUM_SERIAL,
                        help="10X Visium slide serial number \
                            https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/inputs/image-slide-parameters#slide-serial-numbers")
    parser.add_argument("--preprocess", nargs="*", 
                        default=[],
                        help=f"预处理参数（默认: {DEFAULT_PREPROCESS}，格式: key=value）")
    parser.add_argument("--postprocess", nargs="*", 
                        default=[],
                        help=f"后处理参数（默认: {DEFAULT_POSTPROCESS}，格式: key=value）")

def visium2hd(args):
    preprocess_params = parse_key_value_pairs(args.preprocess, DEFAULT_PREPROCESS)
    postprocess_params = parse_key_value_pairs(args.postprocess, DEFAULT_POSTPROCESS)
    print("Running Visium2HD")
    print(f"  Input Dir: {args.input}")
    print(f"  Output Dir: {args.output}")
    print(f"  Format: {args.format}")
    print(f"  Super Pixel Size: {args.super_pixel_size}")
    print(f"  Model: {args.model}")
    print(f"  Preprocessing: {preprocess_params}")
    print(f"  Postprocessing: {postprocess_params}")
    pipeline = Pipeline(model_name=args.model)
    pipeline.run_Visium2HD(
        input_path=args.input,
        source_image_path=args.source_image_path,
        output_path=args.output,
        super_pixel_size=args.super_pixel_size,
        format=args.format,
        slide_serial=args.visium_serial,
        preprocess=preprocess_params,
        postprocess=postprocess_params
    )
    
def hd2visium(args):
    preprocess_params = parse_key_value_pairs(args.preprocess, DEFAULT_PREPROCESS)
    postprocess_params = parse_key_value_pairs(args.postprocess, DEFAULT_POSTPROCESS)
    print("Running HD2Visium")
    print(f"  Input Dir: {args.input}")
    print(f"  Output Dir: {args.output}")
    print(f"  Format: {args.format}")
    print(f"  Preprocessing: {preprocess_params}")
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
    postprocess_params = parse_key_value_pairs(args.postprocess, DEFAULT_POSTPROCESS)
    print("Running Benchmark")
    print(f"  Input Dir: {args.input}")
    print(f"  Output Dir: {args.output}")
    print(f"  Super Pixel Size: {args.super_pixel_size}")
    print(f"  Model: {args.model}")
    print(f"  Rebin: {args.rebin}")
    print(f"  Preprocessing: {preprocess_params}")
    print(f"  Postprocessing: {postprocess_params}")
    pipeline = Pipeline(model_name=args.model)
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
        postprocess=postprocess_params
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tool for processing Visium and VisiumHD data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # HD2Visium 子命令（不需要模型选项）
    parser_h2v = subparsers.add_parser("HD2Visium", help="Convert VisiumHD to Visium")
    common_args(parser_h2v)
    parser_h2v.add_argument("--bin_size", type=int, default=2,
                           help=f"用于合并的分箱大小（默认: 2）")
    parser_h2v.set_defaults(func=hd2visium)

    # Visium2HD 子命令：添加模型选项和超像素大小
    parser_v2hd = subparsers.add_parser("Visium2HD", help="Convert Visium to VisiumHD")
    common_args(parser_v2hd)
    parser_v2hd.add_argument("--super_pixel_size", type=int,
                             default=DEFAULT_SUPER_PIXEL_SIZE,
                             help=f"超像素大小（默认: {DEFAULT_SUPER_PIXEL_SIZE}）")
    parser_v2hd.add_argument("--model", choices=SUPPORTED_TOOLS,
                             default=DEFAULT_MODEL,
                             help=f"超分模型选项")
    parser_v2hd.set_defaults(func=visium2hd)

    # Benchmark 子命令：添加方法、模型选项和超像素大小
    parser_bm = subparsers.add_parser("Benchmark", help="Benchmark different processing methods")
    common_args(parser_bm)
    parser_bm.add_argument("--bin_size", type=int, default=2,
                           help=f"用于合并 spot 的 bin 大小（默认: 2）")
    parser_bm.add_argument("--no-rebin", dest="rebin", action="store_false",
                        help="不将 bin 合并成超像素大小（默认: True）")
    parser_bm.add_argument("--super_pixel_size", type=int,
                           default=DEFAULT_SUPER_PIXEL_SIZE,
                           help=f"超像素大小（默认: {DEFAULT_SUPER_PIXEL_SIZE}）")
    parser_bm.add_argument("--model", choices=SUPPORTED_TOOLS,
                           default=DEFAULT_MODEL,
                           help=f"超分模型选项")
    parser_bm.set_defaults(func=benchmark)

    return parser.parse_args()

def main():
    args = parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
