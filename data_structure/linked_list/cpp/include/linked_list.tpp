#include "linked_list.h"
template <typename T>
void LinkedList<T>::insert(size_t index, const T& value){
    if(index > list_size){
        throw std::out_of_range("index out of range");
    }

    // create new_node in the heap
    Node<T>* new_node = new Node<T>(value); 

    if (index==0) {
        new_node->next = head; // equivalent to (*new_node).next
        if (head) {
            head->prev = new_node;
        }
        head = new_node;
        if (list_size == 0) {
            tail = new_node;
        }
    }
    else if (index == list_size) {
        new_node->prev = tail;
        if (tail) {
            tail->next = new_node;
        }
        tail = new_node;
    }
    else{
        Node<T>* current = head;
        for (size_t i=0; i<index; ++i) {
            current = current->next;
        }
        new_node->prev = current->prev;
        new_node->next = current;
        if (current->prev) {
            current->prev->next = new_node;
        } 
        current->prev = new_node;
    }
    ++list_size;
}

template <typename T>
void LinkedList<T>::insert(const T& value){
    insert(list_size, value);
}

// copy constructor
template <typename T>
LinkedList<T>::LinkedList(const LinkedList& other): head(nullptr), tail(nullptr), list_size(0){
    Node<T>* current = other.head;
    while(current){
        this->insert(list_size, current->value);
        current = current->next;
    }
}

// move constructor
template <typename T>
LinkedList<T>::LinkedList(LinkedList&& other) noexcept
: head(other.head), tail(other.tail), list_size(other.list_size){
    other.head = nullptr;
    other.tail = nullptr;
    other.list_size = 0;
}

// destructor
template <typename T>
LinkedList<T>::~LinkedList(){
    while (head) {
        Node<T>* tmp = head;
        head = head->next;
        delete tmp;
    }
    tail = nullptr;
    list_size = 0;
}

template <typename T>
T LinkedList<T>::pop(size_t index){

    if (index>= list_size) {
        throw std::out_of_range("index out of range");
    }

    T res;
    
    if (index == 0) {
        Node<T>* target = this->head;
        res = target->value;
        head = head->next;
        if (head) {
            head->prev = nullptr;
        }
        else{
            tail = nullptr;
        }
        delete target;
    }
    else if (index == list_size - 1) {
        Node<T>* target = this->tail;
        res = target->value;
        tail = tail->prev;
        if (tail) {
            tail->next = nullptr;
        }
        else{
            head = nullptr;
        }
        delete target;
    }
    else {
        Node<T>* target = this->head;
        for (size_t i = 0; i<index; ++i) {
            target = target->next;
        }
        res = target->value;
        target->prev->next = target->next;
        target->next->prev = target->prev;
        delete target;
    }
    
    --list_size;
    return res;
}

template <typename T>
T LinkedList<T>::pop(){
    return pop(list_size-1);
}

template <typename T>
T& LinkedList<T>::operator[](size_t index){
    if (index>= list_size) {
        throw std::out_of_range("Index out of range");
    }

    Node<T>* current = this->head;
    for (size_t i=0; i<index; ++i) {
        current = current->next;
    }
    return current->value;
}

template <typename T>
const T& LinkedList<T>::operator[](size_t index) const {
    if (index>= list_size) {
        throw std::out_of_range("index out of range");
    }

    Node<T>* current = this->head;
    for (size_t i=0; i<index; ++i) {
        current = current->next;
    }
    return current->value;
}

template <typename T>
void LinkedList<T>::reverse(){

    if (!head || !head->next) {
        return;
    }

    Node<T>* current = this->head;
    Node<T>* l=nullptr, *r=nullptr;
    while (current) {
        l = current->prev;
        r = current->next;
        current->next = l;
        current->prev = r;
        current = r;
    }
    std::swap(this->head, this->tail);
}