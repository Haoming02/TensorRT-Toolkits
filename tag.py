from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os

from TensorRT import TrtModel
from common import parse_path

Benchmark = False


def preprocess_image(image: Image) -> np.array:
    """Given an Image, preprocess to the specified format"""

    image = np.asarray(image.convert("RGB"))

    # (H, W, 3) -> (448, 448, 3)
    image = cv2.resize(image, (448, 448), interpolation=cv2.INTER_AREA)
    # RGB -> BGR
    image = image[:, :, ::-1]
    # (448, 448, 3) -> (1, 448, 448, 3)
    image = np.expand_dims(image, 0)

    return image


def process(
    path: str,
    general_threshold: float = 0.36,
    character_threshold: float = 1.0,
    escape: bool = True,
    replace_underscore: bool = True,
):
    """Main Function"""
    path = path.strip('"').strip()
    if not os.path.exists(path):
        print("Invalid Path...")
        return

    IMAGES = parse_path(path)
    if len(IMAGES) == 0:
        print("No Valid Images...")
        return

    if Benchmark:
        import time
        start_time = time.time()

    model = TrtModel("WD14.trt", dtype=np.float16)
    CSV = pd.read_csv("selected_tags.csv")

    tag_names = CSV["name"].tolist()
    general_indexes = list(np.where(CSV["category"] == 0)[0])
    character_indexes = list(np.where(CSV["category"] == 4)[0])

    for IMAGE in IMAGES:
        data = preprocess_image(IMAGE)
        result = model(data)[0]

        labels = list(zip(tag_names, result[0]))

        general_names = [labels[i] for i in general_indexes]
        general_res = [x for x in general_names if x[1] > general_threshold]
        general_res = dict(general_res)

        character_names = [labels[i] for i in character_indexes]
        character_res = [x for x in character_names if x[1] > character_threshold]
        character_res = dict(character_res)

        tags = dict(sorted(general_res.items(), key=lambda item: item[1], reverse=True))
        caption = ", ".join(list(tags.keys()))

        if escape:
            caption = caption.replace("(", "\(").replace(")", "\)")
        if replace_underscore:
            caption = caption.replace("_", " ")

        with open(f"{os.path.splitext(IMAGE.filename)[0]}.txt", "w") as OUTPUT:
            OUTPUT.write(caption)

    if Benchmark:
        print(f"Took: {round(time.time() - start_time, 2)}s")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        process(*sys.argv[1:])

    else:
        path = input("Path to Image/Folder: ")
        process(path)
