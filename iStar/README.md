## iStar

### Install

1. 
```shell
conda create -n iStar python=3.9
conda activate iStar
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
```
2. Download istar. Replace `requirements.txt`
```shell
pip install -r requirements.txt
```