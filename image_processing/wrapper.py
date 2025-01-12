from functools import wraps
from typing import Callable
import importlib
import numpy as np

m = importlib.import_module("image_processing.image_processing_native")

def validate_input(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(image: np.ndarray, kernel: np.ndarray, *args, **kwargs) -> np.ndarray:
        
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy ndarray.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a 3-channel array with shape (H, W, C).")
        if image.dtype != np.uint8:
            raise TypeError("Image dtype must be np.uint8 (as returned by cv2.imread).")

        if not isinstance(kernel, np.ndarray):
            raise TypeError("Kernel must be a numpy ndarray.")
        if kernel.dtype != np.float32:
            raise TypeError("Kernel dtype must be np.float32.")

        return func(image, kernel, *args, **kwargs)
    
    return wrapper

@ validate_input
def convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    conved = m.apply_convolution(image, kernel)
    return np.clip(conved, 0, 255).astype(np.uint8)