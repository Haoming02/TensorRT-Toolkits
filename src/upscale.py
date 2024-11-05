from TensorRT import TrtRuntime
from util import parse_path
from PIL import Image

import numpy as np
import cv2
import os
import re


def merge_image(
    chunks: list[np.ndarray],
    v_tile_count: int,
    h_tile_count: int,
    chunk_size: int,
    overlap: int,
) -> np.ndarray:
    """Merge the chunks back into a full image"""

    merged_image = np.zeros(
        (v_tile_count * chunk_size, h_tile_count * chunk_size, 3),
        dtype=np.float32,
    )

    for y in range(v_tile_count):
        for x in range(h_tile_count):
            chunk = chunks[x + y * h_tile_count]

            if x > 0:
                for i in range(overlap):
                    chunk[:, i] *= i / (overlap - 1)

            if y > 0:
                for i in range(overlap):
                    chunk[i, :] *= i / (overlap - 1)

            if x < h_tile_count - 1:
                for i in range(overlap):
                    chunk[:, chunk_size - i - 1] *= i / (overlap - 1)

            if y < v_tile_count - 1:
                for i in range(overlap):
                    chunk[chunk_size - i - 1, :] *= i / (overlap - 1)

            merged_image[
                y * (chunk_size - overlap) : y * (chunk_size - overlap) + chunk_size,
                x * (chunk_size - overlap) : x * (chunk_size - overlap) + chunk_size,
            ] += chunk

    return np.clip(merged_image * 255.0, 0, 255).astype(np.uint8)


def preprocess_image(
    image: Image.Image, chunk_size: int, overlap: int
) -> tuple[list[np.ndarray], int, int]:
    """Slice the input image into chunks with overlaps"""

    image = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, _ = image.shape

    v_tile_count = int((height - 1) / (chunk_size - overlap)) + 1
    h_tile_count = int((width - 1) / (chunk_size - overlap)) + 1

    padded_height = v_tile_count * chunk_size
    padded_width = h_tile_count * chunk_size

    # aaa|abc|ccc
    padded_image = cv2.copyMakeBorder(
        image,
        0,
        padded_height - height,
        0,
        padded_width - width,
        cv2.BORDER_REPLICATE,
    )

    slices: list[np.ndarray] = []
    stride: int = chunk_size - overlap

    for y in range(0, padded_height - chunk_size + 1, stride):
        for x in range(0, padded_width - chunk_size + 1, stride):
            slice = padded_image[y : y + chunk_size, x : x + chunk_size]

            # RGB -> BGR
            chunk = slice[:, :, ::-1].astype(np.float32)
            # 0 ~ 255 -> 0.0 ~ 1.0
            chunk = np.clip(chunk / 255.0, 0.0, 1.0)
            # (h, w, 3) -> (3, h, w)
            chunk = np.transpose(chunk, (2, 0, 1))
            # (3, h, w) -> (1, 3, h, w)
            chunk = np.expand_dims(chunk, 0)

            slices.append(chunk)

    return slices, v_tile_count, h_tile_count


def _get_scale(path: str) -> int:
    """Automatically detect the upscale ratio from the model name"""
    match = re.search(r"(\d)[xX]|[xX](\d)", path)
    if match:
        return int(match.group(1) or match.group(2))
    else:
        return int(input("Scale: "))


def process(model_path: str, image_path: str, compress: bool = False):
    """Main Function"""
    trt_model = TrtRuntime(model_path)
    scale: int = _get_scale(model_path)
    shape: tuple[int] = trt_model.get_input_shape()

    for image in parse_path(image_path):
        (
            chunks,
            v_tile_count,
            h_tile_count,
        ) = preprocess_image(image, shape[2], shape[2] // 8)

        outputs: list[np.ndarray] = []
        for chunk in chunks:
            result = (
                trt_model.infer(chunk)
                .reshape(
                    (
                        3,
                        int(shape[2] * scale),
                        int(shape[3] * scale),
                    )
                )
                .transpose((1, 2, 0))
            )
            outputs.append(result[:, :, ::-1].copy())

        upscaled = merge_image(
            outputs,
            v_tile_count,
            h_tile_count,
            shape[2] * scale,
            shape[2] // 8 * scale,
        )

        og_w, og_h = image.size
        chunk = Image.fromarray(upscaled[0 : og_h * scale, 0 : og_w * scale])
        chunk.save(
            f"{os.path.splitext(image.filename)[0]}_{scale}x.png",
            optimize=compress,
            quality=100,
        )


if __name__ == "__main__":
    model_path = input("Path to Engine: ")
    image_path = input("Path to Image: ")
    process(model_path, image_path)
