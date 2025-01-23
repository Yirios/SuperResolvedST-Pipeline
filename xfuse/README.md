# xfuse

## Install

1. 选择合适的 torch 的 cuda 版本
```shell
conda create -n xfuse python=3.8
conda activate xfuse
## for 40 series Nvidia GPU
conda install conda-forge::cudatoolkit=11.7.1
conda install cudnn=8.2.1 -c conda-forge
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```
2. Download xfuse. Replace `pyproject.toml` and `xfuse/convert/utility.py`
```shell
pip install .
```