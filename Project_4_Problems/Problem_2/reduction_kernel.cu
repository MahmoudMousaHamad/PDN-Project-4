#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <driver_types.h>
#include <curand.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cstdio>
#include <cuda.h>

#define BLOCK_SIZE 1024
#define MAX     123123123

__global__
void reduction_kernel(unsigned int* hash_array, unsigned int* nonce_array, unsigned int array_size, unsigned int* out_hash, unsigned int* out_nonce) {
    int index = 2 * blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float hash_reduction[BLOCK_SIZE];
    __shared__ float nonce_reduction[BLOCK_SIZE];

    if (index < array_size) {
        hash_reduction[threadIdx.x] = hash_array[index];
        nonce_reduction[threadIdx.x] = nonce_array[index];
    } else {
        hash_reduction[threadIdx.x] = MAX;
        nonce_reduction[threadIdx.x] = MAX;
    }

    if ((index + BLOCK_SIZE) < array_size) {
        if (hash_array[index + BLOCK_SIZE] < hash_reduction[threadIdx.x]) {
            hash_reduction[threadIdx.x] = hash_array[index + BLOCK_SIZE];
            nonce_reduction[threadIdx.x] = nonce_array[index + BLOCK_SIZE];
        }
    }

    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
        __syncthreads();
        if (threadIdx.x % (2*stride) == 0) {
            hash_reduction[threadIdx.x] = hash_reduction[threadIdx.x + stride];
            nonce_reduction[threadIdx.x] = nonce_reduction[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0) {
        *out_hash = hash_reduction[0];
        *out_nonce = nonce_reduction[0];
    }
}
