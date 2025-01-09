import pytest
from sortlib import bubble_sort, insertion_sort, selection_sort, merge_sort, quick_sort

sorting_algorithms = [
    bubble_sort,
    insertion_sort,
    selection_sort,
    merge_sort,
    quick_sort,
]

@pytest.mark.parametrize("sort_function", sorting_algorithms)
@pytest.mark.parametrize(
    "input_list, expected_output",
    [
        ([], []), 
        ([1], [1]),
        ([3, 2, 1], [1, 2, 3]),
        ([1, 3, 2], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 3]),
        ([5, -1, 0, 3], [-1, 0, 3, 5]),
        ([1.1, 2.2, 3.3], [1.1, 2.2, 3.3]),
        ([3.1, 2.1, 1.1], [1.1, 2.1, 3.1]),
        ([1, 2, 2, 3], [1, 2, 2, 3]),
        ([1000, 500, 100, 50], [50, 100, 500, 1000]),
    ],
)
def test_sorting_correctness(sort_function, input_list, expected_output):
    assert sort_function(input_list) == expected_output

@pytest.mark.parametrize("sort_function", sorting_algorithms)
def test_sorting_large_input(sort_function):
    input_list = list(range(1000, 0, -1))
    expected_output = list(range(1, 1001))
    assert sort_function(input_list) == expected_output

@pytest.mark.parametrize("sort_function", sorting_algorithms)
def test_sorting_invalid_input(sort_function):
    with pytest.raises(TypeError):
        sort_function(None)
    with pytest.raises(TypeError):
        sort_function("not a list")
    with pytest.raises(TypeError):
        sort_function([1, "two", 3])