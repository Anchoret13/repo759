#include "msort.h"
#include <algorithm>

void merge(int* arr, int* temp, std::size_t left, std::size_t mid, std::size_t right) {
    std::size_t i = left;
    std::size_t j = mid;
    std::size_t k = left;
    
    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }
    
    while (i < mid) {
        temp[k++] = arr[i++];
    }
    
    while (j < right) {
        temp[k++] = arr[j++];
    }
    
    for (i = left; i < right; i++) {
        arr[i] = temp[i];
    }
}

void merge_sort_serial(int* arr, int* temp, std::size_t left, std::size_t right) {
    if (right - left <= 1) {
        return;
    }
    
    std::size_t mid = left + (right - left) / 2;
    
    merge_sort_serial(arr, temp, left, mid);
    merge_sort_serial(arr, temp, mid, right);
    
    merge(arr, temp, left, mid, right);
}

void merge_sort_parallel(int* arr, int* temp, std::size_t left, std::size_t right, std::size_t threshold) {
    if (right - left <= 1) {
        return;
    }
    
    if (right - left <= threshold) {
        merge_sort_serial(arr, temp, left, right);
        return;
    }
    
    std::size_t mid = left + (right - left) / 2;
    
    #pragma omp task
    merge_sort_parallel(arr, temp, left, mid, threshold);
    
    #pragma omp task
    merge_sort_parallel(arr, temp, mid, right, threshold);
    
    #pragma omp taskwait
    
    merge(arr, temp, left, mid, right);
}

void msort(int* arr, const std::size_t n, const std::size_t threshold) {
    int* temp = new int[n];
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            merge_sort_parallel(arr, temp, 0, n, threshold);
        }
    }
    
    delete[] temp;
}