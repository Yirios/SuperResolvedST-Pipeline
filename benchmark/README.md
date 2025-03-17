

### Data Download

benchmark 使用的数据均来自 10X 官网，[metadata](metadata.tsv) 中记录更详细的信息。 

由于数据集较大，考虑到不同的网络环境，建议分步手动下载原始数据，确保数据完整。
```shell
# 生成数据下载脚本
python DataDownload.py
# 进入数据集目录，如 Human Kidney 数据集
cd Visium_HD_Human_Kidney_FFPE
bash download.sh
tar -xzvf Visium_HD_Human_Kidney_FFPE_binned_outputs.tar.gz
tar -xzvf Visium_HD_Human_Kidney_FFPE_spatial.tar.gz
```

### 
