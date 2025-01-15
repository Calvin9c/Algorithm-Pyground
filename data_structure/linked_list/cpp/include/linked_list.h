#ifndef LINKED_LIST_H
#define LINKED_LIST_H

#include <stdexcept>
#include <iostream>
#include <utility>

template <typename T>
struct Node{
    T value;
    Node* prev;
    Node* next;

    Node(T val, Node* prev_node = nullptr, Node* next_node = nullptr)
        : value(val), prev(prev_node), next(next_node) {}
};

template <typename T>
class LinkedList{
private:
    Node<T>* head;
    Node<T>* tail;
    size_t list_size;

public:
    // default constructor
    LinkedList() : head(nullptr), tail(nullptr), list_size(0) {};
    // copy constructor
    LinkedList(const LinkedList& other);
    // move constructor
    LinkedList(LinkedList&& other) noexcept;
    // destructor
    ~LinkedList();

    void insert(size_t index, const T& value);
    void insert(const T& value);
    T pop(size_t index);
    T pop();

    T& operator[](size_t index);
    const T& operator[](size_t index) const;

    size_t size() const {
        return list_size;
    };
    bool empty() const {
        return list_size == 0;
    };

    void reverse();
};

#include "linked_list.tpp" 

#endif