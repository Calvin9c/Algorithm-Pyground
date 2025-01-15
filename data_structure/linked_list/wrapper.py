import importlib

m = importlib.import_module("data_structure.linked_list.linked_list_native")

class LinkedList:
    def __init__(self, dtype: str='float'):
        if dtype == 'int':
            self.arch = m.LinkedListInt()
        elif dtype == 'float':
            self.arch = m.LinkedListFloat()
        else:
            raise NotImplementedError(f'Get invalid value for dtype, expected "float" or "int" but get {dtype}')

    def insert(self, *args) -> None:
        
        if len(args) == 2:
            index, value = args
            if not isinstance(index, int):
                raise TypeError(f"Index must be an integer, got {type(index)}")
        elif len(args) == 1:
            value = args[0]
            index = None
        else:
            raise ValueError("Invalid number of arguments. Expected 1 or 2 arguments.")
        
        # Check value type
        if not isinstance(value, (int, float)):
            raise TypeError(f"Value must be an int or float, got {type(value)}")

        if index in [-1, None]:
            self.arch.insert(value) # Insert at tail
        else:
            self.arch.insert(index, value) # Insert at specific index
        
    def pop(self, index: int=-1) -> int | float:
        if index in [-1, None]:
            return self.arch.pop()
        else:
            return self.arch.pop(index)            
    
    def size(self) -> int:
        return self.arch.size()

    def empty(self) -> bool:
        return self.arch.empty() 
    
    def reverse(self) -> None:
        self.arch.reverse()
    
    def __getitem__(self, index: int) -> int | float:
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)}")
        if index < 0 or self.size() <= index:
            raise IndexError("Index out of range")
        return self.arch[index]

    def __setitem__(self, index: int, value: int | float) -> None:
        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)}")
        if index < 0 or index >= self.size():
            raise IndexError("Index out of range")
        self.arch[index] = value

    def __len__(self):
        return self.size()