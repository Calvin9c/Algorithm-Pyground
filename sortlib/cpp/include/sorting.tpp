#ifndef SORTING_TPP
#define SORTING_TPP

#include <iostream>
#include <limits>
#include <vector>
#include <functional>

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
std::vector<T> merge_sort(const std::vector<T> &data){

    const size_t N = data.size();
    if (N<=1) return data;
    std::vector<T> l=merge_sort(std::vector<T>(data.begin(), data.begin()+N/2)),
                   r=merge_sort(std::vector<T>(data.begin()+N/2, data.end()));

    std::vector<T> res;
    int i=0, j=0;
    while (i<l.size() && j<r.size()) {
        if (l[i]<r[j]) {
            res.emplace_back(l[i++]);
        } else {
            res.emplace_back(r[j++]);
        }
    }

    while (i<l.size()) res.emplace_back(l[i++]); 
    while (j<r.size()) res.emplace_back(r[j++]);
    return res;
}

template <typename T>
void quick_sort(std::vector<T>& data) {
    std::function<void(const int&, const int&)> partition = 
    [&](const int &L, const int &R){
        if (L>=R) return;
        const T PIVOT = data[R];
        int i=L;
        for (int j=L; j<R; ++j) {
            if (data[j]<=PIVOT) {
                std::swap(data[j], data[i++]);
            }
        }
        std::swap(data[i], data[R]);
        partition(L, i-1); partition(i+1, R);
    };
    partition(0, static_cast<int>(data.size())-1);  
}

#endif // SORTING_TPP