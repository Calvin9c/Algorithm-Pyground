# Data Structure
The `data_structure` package provides implementations of classic data structures. Currently, it includes:
- LinkedList: A doubly linked list supporting insertion, deletion, and more.

## Installation
```shell
# Navigate to data_structure / linked_list
mkdir build && cd build
cmake -DPython3_EXECUTABLE=$(which python) ..
make
```

## Quick Start

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

## Testing
- Test the Python package using Pytest:
  ```shell
  # Navigate to the root of Algorithm-Pyground
  python -m pytest tests/test_linked_list.py
  ```