import numpy as np
from skimage import exposure
from PIL import Image
from io import BytesIO
import base64


def base64_encode_image(img):
    """Base64 encode an RGB or grayscale 8-bit image

    Args:
        img: Image array as 3 or 4 channel RGB or grayscale; will be converted to uint8 if not already of that type
    Returns:
        Base64 string
    """
    if img.dtype != np.uint8:
        img = exposure.rescale_intensity(img, out_range=np.uint8).astype(np.uint8)
    im = Image.fromarray(img)
    bio = BytesIO()
    im.save(bio, format='PNG')
    return base64.b64encode(bio.getvalue()).decode()


def base64_decode_image(img):
    """Decode base64 image string

    Args:
        img: base64 encoded image data
    Returns:
        3 channel RGB 8-bit image
    """
    return np.array(Image.open(BytesIO(base64.b64decode(img))))
