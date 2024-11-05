from PIL import Image, UnidentifiedImageError
from typing import Generator
import onnx
import os


def parse_path(path: str) -> Generator[Image.Image, None, None]:
    """Given a folder path, return all Images within it"""

    if os.path.isfile(path):
        try:
            yield Image.open(path)
        except UnidentifiedImageError:
            pass
        return

    for file in [os.path.join(path, f) for f in os.listdir(path)]:
        yield from parse_path(file)


def debug(path: str):
    """Given an Onnx model, print out the Name and Shape of its input and output layer"""

    path = path.strip('"').strip()
    assert path.endswith(".onnx")
    model = onnx.load(path)

    print("")
    print(model.graph.input)
    print("\n")
    print(model.graph.output)
    print("")


if __name__ == "__main__":
    path = input("Path to Onnx Model: ")
    debug(path)
