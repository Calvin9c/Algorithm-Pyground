#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "linked_list.h"

namespace py = pybind11;

template <typename T>
void bind_linked_list(py::module_& m, const std::string& name) {
    py::class_<LinkedList<T>>(m, name.c_str()) // m: module,
                                               // name: class name in the module

        .def(py::init<>())                     // Bind the default constructor to Python.
                                               // This allows users to create an instance in Python like:
                                               //     linked_list = LinkedList()

        // Since there are multiple overloaded versions of the insert function,
        // we need to use py::overload_cast to specify which version to bind.
        // This binds the version of insert that allows inserting at a specific index.
        .def("insert", py::overload_cast<size_t, const T&>(&LinkedList<T>::insert), "Insert at index")
        // This binds the version of insert that inserts a value at the end of the list.
        .def("insert", py::overload_cast<const T&>(&LinkedList<T>::insert), "Insert at tail")

        // for the same reason, we need to specifiy which version of pop to bind.
        .def("pop", py::overload_cast<size_t>(&LinkedList<T>::pop), "Pop element at index")
        .def("pop", py::overload_cast<>(&LinkedList<T>::pop), "Pop element at tail")

        .def("size", &LinkedList<T>::size, "Get size of the list")
        .def("empty", &LinkedList<T>::empty, "Check if the list is empty")
        .def("reverse", &LinkedList<T>::reverse, "Reverse the list")

        // Bind the __getitem__ operator to Python.
        // This enables indexing, allowing users to access elements like:
        //     value = linked_list[index]
        .def("__getitem__", [](LinkedList<T>& self, size_t index) {
            return self[index];
        })
        // Bind the __setitem__ operator to Python.
        // This enables indexing with assignment, allowing users to modify elements like:
        //     linked_list[index] = value
        .def("__setitem__", [](LinkedList<T>& self, size_t index, const T& value) {
            self[index] = value;
        })

        // Bind the __len__ operator to Python.
        // This allows users to get the size of the list using len():
        //     size = len(linked_list)
        .def("__len__", &LinkedList<T>::size);
}

PYBIND11_MODULE(linked_list_native, m) {
    m.doc() = "LinkedList module exposed using Pybind11";
    bind_linked_list<int>(m, "LinkedListInt");
    bind_linked_list<float>(m, "LinkedListFloat");
}