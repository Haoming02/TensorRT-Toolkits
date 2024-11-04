import onnx


def debug(path: str):
    """Print out the I/O Shape and Name of an Onnx model"""
    path = path.strip('"').strip()
    assert path.endswith(".onnx")
    model = onnx.load(path)

    print("")
    print(model.graph.input)
    print("\n")
    print(model.graph.output)
    print("")


if __name__ == "__main__":
    PATH = input("Path to ONNX Model: ")
    debug(PATH)
