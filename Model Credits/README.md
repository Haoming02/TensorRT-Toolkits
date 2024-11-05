## WD SwinV2 Tagger v3
- **License:** `apache-2.0`
- **Link:** https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3
- **Conversion:**

    ```bash
    trtexec --onnx=model.onnx --saveEngine=WD14.trt --fp16
    ```

## 2xNomosUni_esrgan_multijpg
- **License:** `CC-BY-4.0`
- **Link:** https://github.com/Phhofm/models/releases/tag/2xNomosUni_esrgan_multijpg
- **Conversion:**

    ```bash
    trtexec --onnx=2xNomosUni_esrgan_multijpg_fp32_opset17.onnx --saveEngine=2xNomosUni_multijpg.trt --shapes=input:1x3x256x256 --fp16
    ```
