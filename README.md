# Algorithm-Pyground
This repository implements various classic algorithms using `C++` and wraps them with `Pybind11` for `Python` integration. The algorithms are tested with `Pytest`. 

Algorithm-Pyground serves as a learning notebook for programming languages `(Python, C++, CUDA, etc.)`, algorithms, and related tools. Note that the features provided by each package are not exhaustive.

## Table of Contents
- [Sorting](#sorting)
- [Dynamic Programming](#dynamic-programming)
- [Image Processing](#image-processing)

## Sorting
The `sortlib` package includes implementations of the following sorting algorithms:
- Bubble Sort
- Insertion Sort
- Selection Sort
- Merge Sort
- Quick Sort

### Installation
```shell
# Navigate to the sortlib directory
mkdir build && cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make
```

### Quick Start
```python
from sortlib import bubble_sort, insertion_sort, selection_sort, merge_sort, quick_sort

x = [5, 5, 7, 1, 3] # List to be sorted
x = bubble_sort(x) # Out-of-space sorting
print(x) # [1, 3, 5, 5, 7]
```

### Testing
- Test the C++ implementation:
  ```shell
  # Navigate to sortlib/cpp
  make run
  ```

- Test the Python package using Pytest:
  ```shell
  # Navigate to the root of Algorithm-Pyground
  python -m pytest
  ```

## Dynamic Programming
 - TODO

## Image Processing
This package implements a `convolution` function using `CUDA` for high-performance image processing.

### Installation
```shell
# Navigate to the image_processing directory
mkdir build && cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make
```

### Quick Start
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