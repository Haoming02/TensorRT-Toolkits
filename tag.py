import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

from PIL import Image, UnidentifiedImageError
from pathlib import Path

import pandas as pd
import numpy as np
import cv2
import os

# =============================================================== #
# TRT Inference Code Credit: https://stackoverflow.com/a/67492525 #
# =============================================================== #

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return f"Host:\n{str(self.host)}\nDevice:\n{str(self.device)}"

    def __repr__(self):
        return self.__str__()


class TrtModel:
    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = (trt.volume(self.engine.get_tensor_shape(binding)) * self.max_batch_size)

            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray, batch_size = 1):

        x = x.astype(self.dtype)

        np.copyto(self.inputs[0].host, x.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        return [out.host.reshape(batch_size, -1) for out in self.outputs]


# =========================================================== #
# Model Used:                                                 #
# https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2 #
# =========================================================== #

SHAPE = (1, 448, 448, 3)

def preprocess_image(image:Image):
    image = image.convert("RGBA")

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)

    image = new_image.convert("RGB")
    image = np.asarray(image)

    image = image[:, :, ::-1]

    image = cv2.resize(image, (SHAPE[1], SHAPE[2]), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    image = np.expand_dims(image, 0)

    return image

def process(path:str, general_threshold:float = 0.25, character_threshold:float = 1.0, escape:bool = True, replace_underscore:bool = False):
    path = path.strip('"').strip()
    if not os.path.exists(path):
        print('Invalid Path')
        return

    model = TrtModel("WD14.trt")
    CSV = pd.read_csv("selected_tags.csv")

    tag_names = CSV["name"].tolist()
    general_indexes = list(np.where(CSV["category"] == 0)[0])
    character_indexes = list(np.where(CSV["category"] == 4)[0])

    for FILE in os.listdir(path):
        IMAGE = os.path.join(path, FILE)
        try:
            data = preprocess_image(Image.open(IMAGE))
        except UnidentifiedImageError:
            continue

        result = model(data, 1)[0]

        labels = list(zip(tag_names, result[0].astype(float)))

        general_names = [labels[i] for i in general_indexes]
        general_res = [x for x in general_names if x[1] > general_threshold]
        general_res = dict(general_res)

        character_names = [labels[i] for i in character_indexes]
        character_res = [x for x in character_names if x[1] > character_threshold]
        character_res = dict(character_res)

        b = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
        c = ", ".join(list(b.keys()))

        if escape:
            c = c.replace("(", "\(").replace(")", "\)")
        if replace_underscore:
            c = c.replace("_", " ")

        info = Path(IMAGE).with_suffix('.txt')
        with open(info, 'w') as OUTPUT:
            OUTPUT.write(c)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        process(*sys.argv[1:])

    else:
        PATH = input('Path to Images: ')
        process(PATH)
