#ifndef SORTING_H
#define SORTING_H

#include <vector>

template <typename T>
void bubble_sort(std::vector<T>& data);

template <typename T>
void insertion_sort(std::vector<T>& data);

template <typename T>
void selection_sort(std::vector<T>& data);

/*merge sort*/
template <typename T>
std::vector<T> _merge(const std::vector<T>& l, const std::vector<T>& r);
template <typename T>
std::vector<T> merge_sort(const std::vector<T>& data);

/*quick sort*/
template <typename T>
size_t partition(std::vector<T>& data, size_t low, size_t high);
template <typename T>
void _quick_sort_helper_impl_0(std::vector<T>& data, size_t low, size_t high);
template<typename T>
void _quick_sort_helper_impl_1(std::vector<T>& data, size_t l, size_t r);

template <typename T>
void quick_sort(std::vector<T>& data /*, int impl=0*/);

#include "sorting.tpp"

#endif