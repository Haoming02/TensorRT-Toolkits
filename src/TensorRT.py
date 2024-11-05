import pycuda.driver as cuda
import pycuda.autoinit  # noqa F401
import tensorrt as trt
import numpy as np


class TrtRuntime:

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()

        (
            self.inputs,
            self.outputs,
            self.bindings,
            self.stream,
        ) = self._allocate_buffers(self.engine)

    class HostDevice:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def _load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            return self.runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i in range(engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDevice(host_mem, device_mem))
            else:
                outputs.append(self.HostDevice(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def get_input_shape(self) -> tuple[int]:
        layer: str = self.engine.get_tensor_name(0)
        return self.engine.get_tensor_shape(layer)

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(
                self.engine.get_tensor_name(i), self.bindings[i]
            )

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        cuda.memcpy_dtoh_async(
            self.outputs[0].host, self.outputs[0].device, self.stream
        )

        self.stream.synchronize()

        return self.outputs[0].host
