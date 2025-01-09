#ifndef SORTING_TPP
#define SORTING_TPP

#include <iostream>
#include <limits>
#include <vector>

template <typename T>
void bubble_sort(std::vector<T>& data){
    size_t n = data.size();
    for (size_t i=0; i<n-1; ++i){
        bool swap = false;
        for(size_t j=0; j<n-i-1; ++j){
            if(data[j] > data[j+1]){
                swap = true;
                std::swap(data[j], data[j+1]);
            }
        }
        if(!swap){break;}
    }
}

template <typename T>
void insertion_sort(std::vector<T>& data){
    size_t n = data.size();
    for(size_t i=1; i<n; ++i){
        T k = data[i];
        size_t j = i;
        while (j > 0 && data[j - 1] > k) {
            data[j] = data[j - 1];
            --j;
        }
        data[j] = k;
    }
}

template <typename T>
void selection_sort(std::vector<T>& data){
    size_t n = data.size();
    for(size_t i=0; i<n; ++i){
        size_t min_idx = i;
        for(size_t j=i+1; j<n; ++j){
            if(data[min_idx]>data[j]){
                min_idx = j;
            }
        }
        std::swap(data[i], data[min_idx]);
    }
}

template <typename T>
std::vector<T> _merge(const std::vector<T>& l, const std::vector<T>& r){

    std::vector<T> res;
    size_t i=0, j=0;
    
    while(i<l.size() && j<r.size()){
        if(l[i] < r[j]){
            res.emplace_back(l[i++]);
        }
        else{
            res.emplace_back(r[j++]);
        }
    }

    while (i < l.size()) {
        res.emplace_back(l[i++]);
    }
    while (j < r.size()) {
        res.emplace_back(r[j++]);
    }

    return res;
}

template <typename T>
std::vector<T> merge_sort(const std::vector<T>& data){
    size_t n = data.size();
    if(n<=1){return data;}
    std::vector<T> l(data.begin(), data.begin() + n / 2);
    std::vector<T> r(data.begin() + n / 2, data.end());
    return _merge(merge_sort(l), merge_sort(r));
}

/*quick sort*/
/*impl with the usage of function: partition*/
template <typename T>
size_t partition(std::vector<T>& data, size_t low, size_t high) {
    T pivot = data[high];
    size_t i = low;

    for (size_t j = low; j < high; ++j) {
        if (data[j] < pivot) {
            std::swap(data[i], data[j]);
            ++i;
        }
    }
    std::swap(data[i], data[high]); // put pivot into its position
    return i; // return the pivot position
}
template <typename T>
void _quick_sort_helper_impl_0(std::vector<T>& data, size_t low, size_t high) {
    if (low < high) {

        size_t pivot_index = partition(data, low, high);

        if (pivot_index > 0) {
            _quick_sort_helper_impl_0(data, low, pivot_index - 1);
        }
        _quick_sort_helper_impl_0(data, pivot_index + 1, high);
    }
}

/*impl without using function: partition*/
template<typename T>
void _quick_sort_helper_impl_1(std::vector<T>& data, size_t l, size_t r){
    if(l<r){
        T pivot = data[l];
        size_t i = l+1, j = r;
        
        while(i<j){
            while(i<=r && data[i]<=pivot){
                ++i;
            }
            while(j>l && data[j]>pivot){
                --j;
            }
            if(i<j){
                std::swap(data[i], data[j]);
            }
        }
        std::swap(data[l], data[j]);
        if (j > 0) _quick_sort_helper_impl_1(data, l, j-1);
        _quick_sort_helper_impl_1(data, j+1, r);
    }
}

template <typename T>
void quick_sort(std::vector<T>& data /*, int impl*/) {
    if (!data.empty()) {
        _quick_sort_helper_impl_0(data, 0, data.size()-1);
        // if(impl==0){
        //     _quick_sort_helper_impl_0(data, 0, data.size()-1);
        // }
        // else{
        //     _quick_sort_helper_impl_1(data, 0, data.size()-1);
        // }
    }
}

#endif // SORTING_TPP