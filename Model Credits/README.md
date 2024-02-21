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
    trtexec --onnx=4xNomos8kDAT.onnx --saveEngine=4xNomos8kDAT.trt --shapes=input:1x3x192x192 --inputIOFormats=fp32:chw --outputIOFormats=fp32:chw
    ```

## 4xNomos8kSCHAT-S
- **Link:** https://openmodeldb.info/models/4x-Nomos8kSCHAT-S
- **License:** `CC-BY-4.0`
- **Conversion:**
    ```bash
    trtexec --onnx=4xNomos8kSCHAT-S.onnx --saveEngine=4xNomos8kSCHAT-S.trt --shapes=input:1x3x256x256 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw
    ```

## 2xHFA2kAVCSRFormer_light
- **Link:** https://openmodeldb.info/models/2x-HFA2kAVCSRFormer-light
- **License:** `CC-BY-4.0`
- **Conversion:**
    ```bash
    trtexec --onnx=2xHFA2kAVCSRFormer_light_64_onnxsim_fp16.onnx --saveEngine=2xHFA2kAVCSRFormer.trt --shapes=input:1x3x256x256 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --precisionConstraints=obey --layerPrecisions=/patch_embed/norm/ReduceMean:fp32,/patch_embed/norm/Sub:fp32,/patch_embed/norm/Pow:fp32,/patch_embed/norm/ReduceMean_1:fp32,/patch_embed/norm/Add:fp32,/patch_embed/norm/Sqrt:fp32,/patch_embed/norm/Div:fp32,/patch_embed/norm/Mul:fp32,/patch_embed/norm/Add_1:fp32,/norm/ReduceMean:fp32,/norm/Sub:fp32,/norm/Pow:fp32,/norm/ReduceMean_1:fp32,/norm/Add:fp32,/norm/Sqrt:fp32,/norm/Div:fp32,/norm/Mul:fp32,/norm/Add_1:fp32 --fp16
    ```
