from PIL import Image, UnidentifiedImageError
import os


def parse_path(path: str) -> list:
    """Given a path, return a list of all Images within it"""
    if os.path.isdir(path):
        images = []

        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                try:
                    img = Image.open(os.path.join(path, file))
                    images.append(img)
                except UnidentifiedImageError:
                    continue
            else:
                images += parse_path(os.path.join(path, file))

        return images

    else:
        try:
            img = Image.open(path)
            return [img]
        except UnidentifiedImageError:
            return []
