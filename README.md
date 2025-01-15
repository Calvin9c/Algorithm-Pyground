# Algorithm-Pyground
This repository implements various classic algorithms using `C++` and wraps them with `Pybind11` for `Python` integration. The algorithms are tested with `Pytest`. 

Algorithm-Pyground serves as a learning notebook for programming languages `(Python, C++, CUDA, etc.)`, algorithms, and related tools. Note that the features provided by each package are not exhaustive.

## Table of Contents
- [DataStructure](#data-structure)
- [Sorting](#sorting)
- [Dynamic Programming](#dynamic-programming)
- [Image Processing](#image-processing)

## Data Structure
The `data_structure` package provides implementations of classic data structures. Currently, it includes:
- LinkedList: A doubly linked list supporting insertion, deletion, and more.

### Installation
```shell
# Navigate to data_structure / linked_list
mkdir build && cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make
```

### Quick Start

```python
from data_structure import LinkedList

# Create a LinkedList for integers or floats
dtype = "float"  # Change to "int" for integer-based LinkedList
ll = LinkedList(dtype=dtype)

# Insert elements
ll.insert(10.5)          # Insert at the end
ll.insert(0, 5.0)        # Insert at index 0
ll.insert(1, 7.5)        # Insert at index 1

# Access elements
print(ll[0])             # Output: 5.0
print(ll[1])             # Output: 7.5

# Modify elements
ll[1] = 6.5              # Update value at index 1
print(ll[1])             # Output: 6.5

# Remove elements
value = ll.pop()         # Remove and return the last element
print(value)             # Output: 10.5

# Reverse the list
ll.reverse()
print(ll[0])             # Output: 6.5 (now the last inserted element is first)

# Get list properties
print(len(ll))           # Output: 2 (size of the list)
print(ll.empty())        # Output: False
```

### Testing
- Test the Python package using Pytest:
  ```shell
  # Navigate to the root of Algorithm-Pyground
  python -m pytest tests/test_linked_list.py
  ```

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
  python -m pytest tests/test_sortlib.py
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