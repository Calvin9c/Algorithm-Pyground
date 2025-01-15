# Sorting
The `sortlib` package includes implementations of the following sorting algorithms:
- Bubble Sort
- Insertion Sort
- Selection Sort
- Merge Sort
- Quick Sort

## Installation
```shell
# Navigate to the sortlib directory
mkdir build && cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make
```

## Quick Start
```python
from sortlib import bubble_sort, insertion_sort, selection_sort, merge_sort, quick_sort

x = [5, 5, 7, 1, 3] # List to be sorted
x = bubble_sort(x) # Out-of-space sorting
print(x) # [1, 3, 5, 5, 7]
```

## Testing
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