<h1 align="center">TensorRT Toolkits</h1>
<p align="center">Blazingly fast deep learning inference via TensorRT acceleration</p>

## Prerequisites
- Nvidia GPU
- CUDA Toolkit
- cuDNN
- TensorRT

> [!IMPORTANT]
> This repo was built with TensorRT **10.0.1** on Python **3.10**; older TensorRT is **not** compatible

## Getting Started
0. Clone the Repo
    ```bash
    git clone https://github.com/Haoming02/TensorRT-Toolkits
    cd TensorRT-Toolkits
    ```
1. Create a virtual environment
    ```bash
    python -m venv venv
    venv\scripts\activate
    ```
2. Install the necessary packages
    ```bash
    (venv) pip install -r requirements.txt
    ```
3. Download **TensorRT** package from:
    - https://developer.nvidia.com/tensorrt/download
    - *(a developer account is required)*
4. Extract and install the pre-built wheel in the `python` folder of the downloaded `.zip`
    ```bash
    (venv) pip install tensorrt-10.0.1-cp310-none-win_amd64.whl
    ```
5. [Download](https://github.com/Haoming02/TensorRT-Toolkits/releases) / [Convert](#how-to-convert-models) a Model

## Features
> This repo comes with a script to load a TensorRT engine; and two example scripts for running inferences

#### Caption
Caption images using Booru tags
- **Script:** `tag.py`
- **Benchmark:** Tagging **850** images on a RTX 3060 took **~150s**

#### Upscale
Upscale images using ESRGAN
- **Script:** `upscale.py`
- **Benchmark:** Upscaling a **1024x1024** image on a RTX 3060 took:
    - **~5.6s** via PyTorch `.pth`
    - **~3.6s** via TensorRT `.trt`

## How to Convert Models
1. Prepare a `.onnx` model
2. Extract the `trtexec.exe` program from the `bin` folder of the downloaded `.zip`
3. Run the conversion command
    ```bash
    trtexec --onnx=model.onnx --saveEngine=model.trt
    ```

> [!TIP]
> See [here](https://github.com/Haoming02/TensorRT-Cpp#trtexec) for more details

<hr>

- **See Also:** [TensorRT-Cpp](https://github.com/Haoming02/TensorRT-Cpp) for **C++** implementation
