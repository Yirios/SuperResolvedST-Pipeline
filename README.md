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

### Quite Start

### Benchmark

### Tutorials and Analyses Pipeline

- A few basic super resolved pipelines are given in [tutorials.ipynb](tutorials.ipynb).
- The [analyses](analyses) gives a few specific examples of analyses, and the Benchmark method.