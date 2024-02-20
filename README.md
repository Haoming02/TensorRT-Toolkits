# TensorRT Toolkits
Inference models blazingly fast thanks to TensorRT acceleration!

## Prerequisites
- nVIDIA GPU
- CUDA Toolkit
- cuDNN

## Getting Started
0. Clone the Repo
1. Install the necessary packages
    ```bash
    pip install -r requirements.txt
    ```
    > For `tensorrt`, download it from: https://developer.nvidia.com/tensorrt-download *(account required)*

2. Download the converted model(s) from [Releases](https://github.com/Haoming02/TensorRT-Toolkits/releases) and place it into the Repo folder

## Features

#### Caption
Caption images using Booru tags
- Run the `tag.py` script and enter a path

<details>
<summary>Benchmark</summary>

Caption 240 images within a folder using `RTX 3060`
- Took **~16 sec**
</details>

#### Upscale
Upscale the images by 4x
- Run the `upscale.py` script and enter a path

<details>
<summary>Benchmarks</summary>

Upscale a 1024x1024 Image on a `RTX 3060`
- Running on [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) in (`fp32`) PyTorch (`.pth`) format: **~54 sec**
- Running (`fp32`) `4xNomos8kDAT` in TensorRT format: **~36 sec**
- Running (`fp16`) `4xNomos8kSCHAT-S` in TensorRT format: **~26 sec**
</details>

## Convert Your Own Models
- Prepare a `.onnx` model
- Use the `trtexec` executable from the TensorRT package and run the command
    ```bash
    trtexec --onnx=model.onnx --saveEngine=model.trt
    ```

<hr> 

<sup>This repo serves as a documentation and practice for converting **Onnx** models to **TensorRT** format. </sup>
