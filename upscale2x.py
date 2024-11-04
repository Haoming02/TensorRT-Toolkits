from PIL import Image
import numpy as np
import cv2
import os

from TensorRT import TrtModel
from common import parse_path


def merge_image(
    chunks: list,
    vertical_tile: int,
    horizontal_tile: int,
    chunk_size: int,
    overlap: int,
) -> np.ndarray:
    """Merge the chunks back into a full image"""
    merged_image = np.zeros(
        (vertical_tile * chunk_size, horizontal_tile * chunk_size, 3), dtype=np.float16
    )

    for y in range(vertical_tile):
        for x in range(horizontal_tile):
            chunk = chunks[x + y * horizontal_tile]

            if x > 0:
                for i in range(overlap):
                    chunk[:, i] *= i / (overlap - 1)

            if y > 0:
                for i in range(overlap):
                    chunk[i, :] *= i / (overlap - 1)

            if x < horizontal_tile - 1:
                for i in range(overlap):
                    chunk[:, chunk_size - i - 1] *= i / (overlap - 1)

            if y < vertical_tile - 1:
                for i in range(overlap):
                    chunk[chunk_size - i - 1, :] *= i / (overlap - 1)

            merged_image[
                y * (chunk_size - overlap) : y * (chunk_size - overlap) + chunk_size,
                x * (chunk_size - overlap) : x * (chunk_size - overlap) + chunk_size,
            ] += chunk

    return np.clip(merged_image * 255.0, 0, 255).astype(np.uint8)


def preprocess_image(image: Image.Image, chunk_size: int, overlap: int) -> tuple:
    """Slice the input image into chunks with overlaps"""

    image = np.asarray(image.convert("RGB"), dtype=np.uint8)

    slices = []
    height, width = image.shape[0:2]

    vertical_tiles = int((height - 1) / (chunk_size - overlap)) + 1
    horizontal_tiles = int((width - 1) / (chunk_size - overlap)) + 1

    padded_height = vertical_tiles * chunk_size
    padded_width = horizontal_tiles * chunk_size

    # aaa|abc|ccc
    padded_image = cv2.copyMakeBorder(
        image, 0, padded_height - height, 0, padded_width - width, cv2.BORDER_REPLICATE
    )

    stride = chunk_size - overlap
    for y in range(0, padded_height - chunk_size + 1, stride):
        for x in range(0, padded_width - chunk_size + 1, stride):
            chunk = padded_image[y : y + chunk_size, x : x + chunk_size]

            # RGB -> BGR
            chunk = chunk[:, :, ::-1].astype(np.float16)
            # 0 ~ 255 -> 0.0 ~ 1.0
            chunk = np.clip(chunk / 255.0, 0.0, 1.0)
            # (128, 128, 3) -> (3, 128, 128)
            chunk = np.transpose(chunk, (2, 0, 1))
            # (3, 128, 128) -> (1, 3, 128, 128)
            chunk = np.expand_dims(chunk, 0)

            slices.append(chunk)

    return slices, vertical_tiles, horizontal_tiles


def process(path: str):
    """Main Function"""
    path = path.strip('"').strip()
    if not os.path.exists(path):
        print("Invalid Path...")
        return

    IMAGES = parse_path(path)
    if len(IMAGES) == 0:
        print("No Valid Images...")
        return

    model = TrtModel("2xHFA2kAVCSRFormer.trt", dtype=np.float16)
    SHAPE = (1, 3, 256, 256)

    for IMAGE in IMAGES:
        OUTPUT = []
        data, vertical_tile, horizontal_tile = preprocess_image(
            IMAGE, int(SHAPE[2]), int(SHAPE[2] / 16)
        )

        for img in data:
            results = model(img)[0]
            result = results.reshape((3, int(SHAPE[2] * 2), int(SHAPE[3] * 2)))
            img = np.clip(result.transpose((1, 2, 0)), 0, 1)
            OUTPUT.append(img[:, :, ::-1])

        result = merge_image(
            OUTPUT,
            vertical_tile,
            horizontal_tile,
            int(SHAPE[2] * 2),
            int(SHAPE[2] / 8),
        )

        w, h = IMAGE.size
        img = Image.fromarray(result[0 : h * 2, 0 : w * 2])
        img.save(f"{os.path.splitext(IMAGE.filename)[0]}_2x.png")


if __name__ == "__main__":
    path = input("Path to Image/Folder: ")
    process(path)
