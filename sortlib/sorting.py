from functools import wraps
from typing import Callable
import importlib

m = importlib.import_module("sortlib.sortlib_native")

def validate_input(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(data):
        if not isinstance(data, list):
            raise TypeError(f"Input data must be a list, got {type(data).__name__} instead.")
        if not all(isinstance(item, (int, float)) for item in data):
            raise TypeError("All elements in the list must be int or float.")
        if len(data) == 0:
            return []
        return func(data)
    return wrapper

@validate_input
def bubble_sort(data: list[int|float]):
    return m.bubble_sort(data)

@validate_input
def insertion_sort(data: list[int|float]):
    return m.insertion_sort(data)

@validate_input
def selection_sort(data: list[int|float]):
    return m.selection_sort(data)

@validate_input
def merge_sort(data: list[int|float]):
    return m.merge_sort(data)

@validate_input
def quick_sort(data: list[int|float]):
    return m.quick_sort(data)