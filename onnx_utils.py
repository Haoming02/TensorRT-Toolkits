import onnx


def debug(path: str):
    """Print out the I/O Shape and Name of an Onnx model"""
    path = path.strip('"').strip()
    assert path.endswith(".onnx")
    model = onnx.load(path)

    print(model.graph.input)
    print(model.graph.output)

    for node in model.graph.node:
        print(node.name)


def convert(path: str):
    """Attempt to convert the model to fp16 precision"""
    from onnxconverter_common import float16
    import os

    path = path.strip('"').strip()
    assert path.endswith(".onnx")
    model = onnx.load(path)

    model_fp16 = float16.convert_float_to_float16(
        model,
        min_positive_val=1e-7,
        max_finite_val=1e4,
        keep_io_types=False,
        disable_shape_infer=False,
        op_block_list=None,
        node_block_list=None,
    )

    onnx.save(model_fp16, f"{os.path.splitext(path)[0]}-fp16.onnx")


if __name__ == "__main__":
    PATH = input("Path to ONNX Model: ")
    MODE = input("Mode [io/16]: ")

    if MODE.strip() == 'io':
        debug(PATH)
    if MODE.strip() == '16':
        convert(PATH)
    else:
        raise SystemExit
