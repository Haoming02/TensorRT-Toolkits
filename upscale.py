from PIL import Image
import numpy as np
import cv2
import os

from TensorRT import TrtModel
from common import parse_path

# ============================================= #
# Model Used:                                   #
# https://openmodeldb.info/models/4x-Nomos8kDAT #
# ============================================= #

SHAPE = {
    "B": 1,
    "C": 3,
    "H": 128,
    "W": 128
}


def merge_image(chunks, vertical_tile, horizontal_tile, chunk_size=512, overlap=16):
    """Merge the chunks back into a full image"""
    merged_image = np.zeros(
        (vertical_tile * chunk_size, horizontal_tile * chunk_size, 3), dtype=np.float32
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


def preprocess_image(image: Image, chunk_size=128, overlap=4) -> list:
    """Slice the input image into chunks with overlaps"""
    image = image.convert("RGBA")

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, mask=image)

    image = new_image.convert("RGB")
    image = np.asarray(image)

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
            chunk = chunk[:, :, ::-1].astype(np.float32)
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

    model = TrtModel("4xNomos8kDAT.trt")

    for IMAGE in IMAGES:
        OUTPUT = []
        data, vertical_tile, horizontal_tile = preprocess_image(IMAGE)

        for img in data:
            results = model(img)[0]
            # (1, 786432) -> (1, 3, 512, 512)
            result = results.reshape((1, 3, 512, 512))
            # (1, 3, 512, 512) -> (512, 512, 3)
            img = np.clip(result[0].transpose((1, 2, 0)), 0, 1)
            # BGR -> RGB
            OUTPUT.append(img[:, :, ::-1])

        result = merge_image(OUTPUT, vertical_tile, horizontal_tile)

        w, h = IMAGE.size
        img = Image.fromarray(result[0 : h * 4, 0 : w * 4])
        img.save(f"{os.path.splitext(IMAGE.filename)[0]}_4x.png")


if __name__ == "__main__":
    PATH = input("Path to Images Folder: ")
    process(PATH)
