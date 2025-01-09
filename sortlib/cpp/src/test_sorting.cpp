#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include "../include/sorting.h"

std::vector<int> _testing_data(size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);

    std::vector<int> data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dis(gen);
    }

    std::cout << "Generated testing data: ";
    for (const auto& val : data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return data;
}

template <typename Func>
void test_sort(
    const std::string& name,
    size_t n,
    Func sort_func
){
    std::cout << "########## ########## ##########\n";
    std::cout << "Test " << name << " ...\n";

    auto testing_data = _testing_data(n);
    sort_func(testing_data);

    std::cout << "Result: ";
    for (const auto& val : testing_data) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    if (std::is_sorted(testing_data.begin(), testing_data.end())) {
        std::cout << "Test passed.\n";
    } else {
        std::cout << "Test failed!\n";
    }
}

int main() {

    size_t n = 6;
    
    std::cout << "Testing elementary sorting algorithms ...\n";
    test_sort("bubble_sort", n, bubble_sort<int>);
    test_sort("insertion_sort", n, insertion_sort<int>);
    test_sort("selection_sort", n, selection_sort<int>);

    std::cout << "Testing advanced sorting algorithms ...\n";
    test_sort("merge_sort", n, [](std::vector<int>& data) { data = merge_sort(data); });
    test_sort("quick_sort", n, quick_sort<int>);

    return 0;
}