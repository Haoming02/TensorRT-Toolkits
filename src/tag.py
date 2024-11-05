from TensorRT import TrtRuntime
from util import parse_path
from PIL import Image

import pandas as pd
import numpy as np
import cv2
import os


def preprocess_image(image: Image.Image, shape: tuple[int]) -> np.ndarray:
    """Given an Image, preprocess to the specified format"""
    image = np.asarray(image.convert("RGB"), dtype=np.uint8)

    # (H, W, 3) -> (448, 448, 3)
    image = cv2.resize(image, shape[1:3], interpolation=cv2.INTER_AREA)
    # RGB -> BGR
    image = image[:, :, ::-1]
    # (448, 448, 3) -> (1, 448, 448, 3)
    image = np.expand_dims(image, 0)

    return image


def _get_tags() -> pd.DataFrame:
    if os.path.isfile("selected_tags.csv"):
        return pd.read_csv("selected_tags.csv")
    else:
        return pd.read_csv(input("Path to CSV: "))


def process(
    model_path: str,
    image_path: str,
    general_threshold: float = 0.4,
    character_threshold: float = 999.0,
    escape: bool = True,
    replace_underscore: bool = True,
):
    """Main Function"""
    trt_model = TrtRuntime(model_path)
    shape: tuple[int] = trt_model.get_input_shape()
    dataset = _get_tags()

    tag_names: list[str] = dataset["name"].tolist()
    general_indexes: list[int] = np.nonzero(dataset["category"] == 0)[0]
    character_indexes: list[int] = np.nonzero(dataset["category"] == 4)[0]

    for image in parse_path(image_path):
        input_data = preprocess_image(image, shape)
        result: tuple[float] = trt_model.infer(input_data)

        tags: dict[str, float] = {}

        for i, (tag, score) in enumerate(zip(tag_names, result)):
            if i in general_indexes and score > general_threshold:
                tags[tag] = score
            if i in character_indexes and score > character_threshold:
                tags[tag] = score

        sorted_tags: list[str] = sorted(
            tags.keys(),
            key=lambda k: tags[k],
            reverse=True,
        )

        caption: str = ", ".join(sorted_tags)

        if escape:
            caption = caption.replace("(", "\(").replace(")", "\)")
        if replace_underscore:
            caption = caption.replace("_", " ")

        with open(f"{os.path.splitext(image.filename)[0]}.txt", "w+") as file:
            file.write(caption)


if __name__ == "__main__":
    model_path = input("Path to Engine: ")
    image_path = input("Path to Image: ")
    process(model_path, image_path)
