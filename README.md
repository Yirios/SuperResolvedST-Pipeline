## SuperResolvedST-Pipeline

**Read this in other languages: [English](README.md), [中文](README.zh.md).**

SuperResolvedST-Pipeline is designed to integrate multiple super-resolution tools into spatial transcriptomics (ST) data analysis. This pipeline is designed to work in two modes: one is to output super-resolved results in image form, and the other is to output super-resolved results in VisiumHD format. The former is suitable for studies that are highly sensitive to image integrity, while the latter crops and reorganizes the image but maintains high spatial accuracy, controlling the error to within 5% of the pixel size, making it suitable for studies sensitive to spatial location.

GPU environments that have passed the test
1. NVIDIA GeForce RTX 4090 D
    - Driver Version: 550.100 CUDA Version: 12.4
2. NVIDIA GeForce RTX 4070 Ti SUPER
    - Driver Version: 550.120 CUDA Version: 12.4

### Install
1. It is recommended to use conda to isolate the running environment of each tool, this procedure has made minor modifications to some tools. Please refer to the installation instructions of the corresponding tools in the root directory for specific installation methods.
2. Install the dependencies required by this pipeline.
3. Configure the environment variables required by this pipeline. Edit the [config.yaml](config.yaml) file, use the full path if possible.

### Quick Start
```
python analyzer.py Visium2HD \
    -i /path/to/Visium_rawdata \
    --source_image_path /path/to/image \
    -o test_Visium2HD \
    -f h5ad \
    --model iStar
```
### Benchmark
|dataset                 |xfuse           |iStar           |TESLA           |ImSpiRE         |
|------------------------|----------------|----------------|----------------|----------------|
|Mouse Brain Fresh Frozen|88.4            |88.1            |76.2            |21.4            |
|Mouse Brain Fixed Frozen|92.5            |92.3            |84.0            |20.0            |
|Mouse Brain             |93.7            |93.5            |87.1            |19.0            |
|Mouse Embryo            |89.3            |89.2            |76.8            |17.5            |
|Mouse Small Intestine   |91.8            |91.5            |83.6            |20.1            |
|Mouse Kidney            |90.7            |90.7            |79.9            |-               |
|Human Kidney FFPE       |95.7            |95.6            |91.5            |-               |
|Human Colon Cancer P5   |90.8            |90.8            |79.2            |-               |
|Human Colon Normal P5   |92.9            |92.7            |86.6            |17.0            |

### Tutorials and Analyses Pipeline

- A few basic super resolved pipelines are given in [tutorials.ipynb](tutorials.ipynb).
- The [analyses](analyses) gives a few specific examples of analyses, and the Benchmark method.