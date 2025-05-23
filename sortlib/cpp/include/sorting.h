#pragma once
#include <vector>

template <typename T>
void bubble_sort(std::vector<T>& data);

template <typename T>
void insertion_sort(std::vector<T>& data);

template <typename T>
void selection_sort(std::vector<T>& data);

/*merge sort*/
template <typename T>
std::vector<T> merge_sort(const std::vector<T>& data);

/*quick sort*/
template <typename T>
void quick_sort(std::vector<T>& data /*, int impl=0*/);

#include "sorting.tpp"