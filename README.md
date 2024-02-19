# TensorRT Toolkits
Execute models blazingly fast thanks to the TensorRT acceleration!

## Requirements
0. TensorRT capable Nvidia GPU
1. Install the necessary packages
    ```bash
    pip install -r requirements.txt
    ```
    > For `tensorrt`, download it from: https://developer.nvidia.com/tensorrt-download *(account required)*

2. Download the converted model from [Release](https://github.com/Haoming02/TensorRT-Toolkits/releases) and place it into this folder

### Caption
Caption the images with Booru tags using a [Tagger](https://github.com/SmilingWolf/SW-CV-ModelZoo) model
- Run the `tag.py` script and enter a path

### Upscale
Upscale the images by 4x with a [DAT](https://github.com/zhengchen1999/DAT) model
- Run the `upscale.py` script and enter a path

<details>
<summary>Benchmark</summary>

Upscale a 1024x1024 Image on a RTX 3060
- Running on [Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) with PyTorch (`.pth`) format: **58s**
- Running with TensorRT (`.trt`) format: **26s**
</details>

### Convert Your Own Models
0. Prepare a `.onnx` model
1. Obtain the `trtexec` program from the TensorRT package mentioned above
2. Run the command *([Guide](https://docs.nvidia.com/deeplearning/tensorrt/quick-start-guide/index.html))*
    ```bash
    trtexec --onnx=model.onnx --saveEngine=model.trt
    ```

<hr> 

<sup>This repo serves as a documentation and practice for converting **Onnx** models to **TensorRT** format. </sup>
