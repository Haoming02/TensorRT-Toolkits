import onnx

def check(path: str):
    """Print out the I/O Shape and Name of an Onnx model"""
    path = path.strip('"').strip()
    assert path.endswith(".onnx")
    model = onnx.load(path)

    print(model.graph.input)
    print(model.graph.output)


if __name__ == "__main__":
    PATH = input("Path to ONNX Model: ")
    check(PATH)
