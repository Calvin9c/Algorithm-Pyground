import pytest
from data_structure import LinkedList

dtypes = ["int", "float"]

@pytest.mark.parametrize("dtype", dtypes)
def test_constructor(dtype):
    linked_list = LinkedList(dtype=dtype)
    assert linked_list.size() == 0
    assert linked_list.empty() is True

@pytest.mark.parametrize("dtype", dtypes)
def test_insert_at_tail(dtype):
    linked_list = LinkedList(dtype=dtype)
    linked_list.insert(10)
    linked_list.insert(20)
    linked_list.insert(30)
    assert linked_list.size() == 3
    assert linked_list[0] == 10
    assert linked_list[1] == 20
    assert linked_list[2] == 30

@pytest.mark.parametrize("dtype", dtypes)
def test_insert_at_index(dtype):
    linked_list = LinkedList(dtype=dtype)
    linked_list.insert(10)
    linked_list.insert(20)
    linked_list.insert(0, 5)
    linked_list.insert(2, 15)
    assert linked_list.size() == 4
    assert linked_list[0] == 5
    assert linked_list[1] == 10
    assert linked_list[2] == 15
    assert linked_list[3] == 20

@pytest.mark.parametrize("dtype", dtypes)
def test_pop(dtype):
    linked_list = LinkedList(dtype=dtype)
    linked_list.insert(10)
    linked_list.insert(20)
    linked_list.insert(30)
    assert linked_list.pop() == 30
    assert linked_list.pop(0) == 10
    assert linked_list.size() == 1
    assert linked_list[0] == 20

@pytest.mark.parametrize("dtype", dtypes)
def test_reverse(dtype):
    linked_list = LinkedList(dtype=dtype)
    linked_list.insert(10)
    linked_list.insert(20)
    linked_list.insert(30)
    linked_list.reverse()
    assert linked_list.size() == 3
    assert linked_list[0] == 30
    assert linked_list[1] == 20
    assert linked_list[2] == 10

@pytest.mark.parametrize("dtype", dtypes)
def test_empty_and_size(dtype):
    linked_list = LinkedList(dtype=dtype)
    assert linked_list.empty() is True
    linked_list.insert(10)
    assert linked_list.empty() is False
    assert linked_list.size() == 1

@pytest.mark.parametrize("dtype", dtypes)
def test_index_out_of_range(dtype):
    linked_list = LinkedList(dtype=dtype)
    linked_list.insert(10)
    with pytest.raises(IndexError):
        linked_list[1]
    with pytest.raises(IndexError):
        linked_list[-1]
    with pytest.raises(IndexError):
        linked_list.pop(1)

@pytest.mark.parametrize("dtype", dtypes)
def test_set_item(dtype):
    linked_list = LinkedList(dtype=dtype)
    linked_list.insert(10)
    linked_list.insert(20)
    linked_list[1] = 25
    assert linked_list[1] == 25

@pytest.mark.parametrize("dtype", dtypes)
def test_len_function(dtype):
    linked_list = LinkedList(dtype=dtype)
    assert len(linked_list) == 0
    linked_list.insert(10)
    linked_list.insert(20)
    assert len(linked_list) == 2
