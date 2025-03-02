## SuperResolvedST-Pipeline

**Read this in other languages: [English](README.md), [中文](README.zh.md).**

SuperResolvedST-Pipeline 旨在将多种超分辨率工具整合到空间转录组学（ST）数据分析中。本流程设计了两种工作模式：一是输出图像形式的超分结果，二是输出 VisiumHD 格式的超分结果。前者适用于对图像完整性高度敏感的研究，后者会对图像进行裁减和重组，但是有很高的空间准确性，可以将误差控制在像素大小的5%，适用于对空间位置敏感的研究。

已经通过测试的 GPU 环境
1. NVIDIA GeForce RTX 4090 D
    - Driver Version: 550.100   CUDA Version: 12.4
2. NVIDIA GeForce RTX 4070 Ti SUPER
    - Driver Version: 550.120   CUDA Version: 12.4

### Install
1. 建议使用 conda 隔离每个工具的运行环境，本流程对一些工具进行了小修改。具体安装方法请参照根目录下对应工具的安装指导。
2. 安装本流程所需的依赖。
3. 配置本流程所需的环境变量，编辑[config.yaml](config.yaml)文件，尽量使用完整路径。

### Quick Start

将 VisiumHD 数据合并成 Visium 数据
```
python analyzer.py HD2Visium \
    -i /data/datasets/Visium_HD_Mouse_Brain_Fresh_Frozen/binned_outputs/square_002um \
    --source_image_path /data/datasets/Visium_HD_Mouse_Brain_Fresh_Frozen/Visium_HD_Mouse_Brain_Fresh_Frozen_tissue_image.tif \
    -o test_HD2Visium \
    -f raw
```
将 Visium 超分 
```
python analyzer.py Visium2HD \
    -i /home/yiriso/Research/Super-resolvedST/data/DLPFC/sample_151673 \
    --source_image_path /home/yiriso/Research/Super-resolvedST/data/DLPFC/sample_151673/151673_full_image.tif \
    -o test_Visium2HD \
    -f h5ad \
    --model iStar \
    --preprocess mask_image=/home/yiriso/Research/Super-resolvedST/data/DLPFC/sample_151673/mask.png 
```
执行 Benchmark，这将从 VisiumHD 数据构建 Pseudo_Visium，然后再超分到 VisiumHD 的分辨率。
```
python analyzer.py Benchmark \
    -i /data/datasets/Visium_HD_Mouse_Brain_Fresh_Frozen/binned_outputs/square_002um \
    --source_image_path /data/datasets/Visium_HD_Mouse_Brain_Fresh_Frozen/Visium_HD_Mouse_Brain_Fresh_Frozen_tissue_image.tif \
    -o test_Benchmark \
    -f h5ad \
    --super_pixel_size 16 \
    --rebin False \
    --model iStar
```
### Benchmark

### Tutorials and Analyses Pipeline in NoteBook

- [tutorials.ipynb](tutorials.ipynb) 中给出了几个基础超分流程。
- [analyses](analyses) 中给出了具体几个分析示例，和 Benchmark 方法。