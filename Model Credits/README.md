## WD 1.4 SwinV2 Tagger V2
- **Link:** https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2
- **License:** `apache-2.0`
- **Conversion:**
    ```bash
    trtexec --onnx=model.onnx --saveEngine=WD14.trt --fp16 --noTF32
    ```

## 4xNomos8kDAT
- **Link:** https://openmodeldb.info/models/4x-Nomos8kDAT
- **License:** `CC-BY-4.0`
- **Conversion:**
    ```bash
    trtexec --onnx=4xNomos8kDAT.onnx --saveEngine=4xNomos8kDAT.trt --shapes="input:1x3x192x192" --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw
    ```

## 4xNomos8kSCHAT-S
- **Link:** https://openmodeldb.info/models/4x-Nomos8kSCHAT-S
- **License:** `CC-BY-4.0`
- **Conversion:**
    ```bash
    trtexec --onnx=4xNomos8kSCHAT-S.onnx --saveEngine=4xNomos8kSCHAT-S.trt --shapes=input:1x3x256x256 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
    ```
