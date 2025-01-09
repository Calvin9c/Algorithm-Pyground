#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/sorting.h"

namespace py = pybind11;

PYBIND11_MODULE(sortlib_native, m) {
    m.doc() = "Sorting algorithms implemented in C++ with Pybind11";

    /*bubble sort*/
    m.def("bubble_sort", [](std::vector<int> data){
        bubble_sort(data);
        return data;
    }, "Bubble Sort for a list of int");
    m.def("bubble_sort", [](std::vector<double> data){
        bubble_sort(data);
        return data;
    }, "Bubble Sort for a list of float");

    /*insertion sort*/
    m.def("insertion_sort", [](std::vector<int> data){
        insertion_sort(data);
        return data;
    }, "Insertion Sort for a list of int");
    m.def("insertion_sort", [](std::vector<double> data){
        insertion_sort(data);
        return data;
    }, "Insertion Sort for a list of float");

    /*selection sort*/
    m.def("selection_sort", [](std::vector<int> data){
        selection_sort(data);
        return data;
    }, "Selection Sort for a list of int");
    m.def("selection_sort", [](std::vector<double> data){
        selection_sort(data);
        return data;
    }, "Selection Sort for a list of float");

    /*merge sort*/
    m.def("merge_sort", &merge_sort<int>, "Merge Sort for a list of int");
    m.def("merge_sort", &merge_sort<double>, "Merge Sort for a list of float");
    
    /*quick sort*/
    m.def("quick_sort", [](std::vector<int> data){
        quick_sort(data);
        return data;
    }, "Quick Sort for a list of int");
    m.def("quick_sort", [](std::vector<double> data){
        quick_sort(data);
        return data;
    }, "Quick Sort for a list of float");
}