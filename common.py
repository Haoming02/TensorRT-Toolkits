from PIL import Image, UnidentifiedImageError
import os

def parse_path(path: str) -> list:
    """Given a path, return a list of all Images within it"""
    if os.path.isdir(path):
        IMAGES = []

        for FILE in os.listdir(path):
            if os.path.isfile(os.path.join(path, FILE)):
                try:
                    img = Image.open(os.path.join(path, FILE))
                    IMAGES.append(img)
                except UnidentifiedImageError:
                    continue
            else:
                IMAGES += parse_path(os.path.join(path, FILE))

        return IMAGES

    else:
        try:
            img = Image.open(path)
            return [img]
        except UnidentifiedImageError:
            return []
