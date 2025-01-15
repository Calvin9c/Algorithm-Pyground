# Image Processing
This package implements a `convolution` function using `CUDA` for high-performance image processing.

## Installation
```shell
# Navigate to the image_processing directory
mkdir build && cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make
```

## Quick Start
```python
from image_processing import convolution
import cv2
import numpy as np

# Read an image as a numpy array with dtype uint8
img = cv2.imread('path/to/img')

# Define a convolution kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]], dtype=np.float32)

# Apply the convolution
img_conved = convolution(img, kernel)

# Save the processed image
cv2.imwrite('conved.png', img_conved)
```