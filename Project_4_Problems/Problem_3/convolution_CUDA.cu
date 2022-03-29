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

#include "kernel.cu"
#include "support.h"

// to activate debug statements
#define DEBUG 1

// program constants
#define BLOCK_SIZE 1024

#define BILLION  1000000000.0
#define MAX_LINE_LENGTH 25000

#define BLUR_SIZE 2

void err_check(cudaError_t ret, char* msg, int exit_code);

int main (int argc, char *argv[])
{
    // Check console errors
    if( argc != 6)
    {
        printf("USE LIKE THIS: convolution_CUDA n_row n_col mat_input.csv mat_output.csv time.csv\n");
        printf("DEBUG: argc: %d\n", argc);
        return EXIT_FAILURE;
    }

    // Get dims
    int n_row = strtol(argv[1], NULL, 10);
    int n_col = strtol(argv[2], NULL, 10);

    // Get files to read/write 
    FILE* inputFile1 = fopen(argv[3], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s\n",argv[3]);
        return EXIT_FAILURE;
    }
    FILE* outputFile = fopen(argv[4], "w");
    FILE* timeFile  = fopen(argv[5], "w");

    size_t size = n_row * n_col * sizeof(int);
    // Matrices to use
    int* K = (int*)malloc(5 * 5 * sizeof(int));
    int* A = (int*) malloc(size);
    int* B = (int*) malloc(size);

    // read the data from the file
    int row_count = 0;
    char line[MAX_LINE_LENGTH] = {0};
    while (fgets(line, MAX_LINE_LENGTH, inputFile1)) {
        if (line[strlen(line) - 1] != '\n') printf("\n");
        char *token;
        const char s[2] = ",";
        token = strtok(line, s);
        int i_col = 0;
        while (token != NULL) {
            A[row_count*n_col + i_col] = strtol(token, NULL,10 );
            i_col++;
            token = strtok (NULL, s);
        }
        row_count++;
    }

    for(int i = 0; i< 5; i++)
        for(int j = 0; j< 5; j++)
            K[i*5+j]=0;

    K[0*5+0] = 1;
    K[1*5+1] = 1;
    K[2*5+2] = 1;
    K[3*5+3] = 1;
    K[4*5+4] = 1;
    
    K[4*5+0] = 1;
    K[3*5+1] = 1;
    K[1*5+3] = 1;
    K[0*5+4] = 1;

    fclose(inputFile1); 

    struct timespec start, end;    
    cudaError_t cuda_ret;

    int num_blocks = ceil((float) (n_row * n_col) / (float)BLOCK_SIZE);
    dim3 dimGrid(num_blocks, 1, 1);
    dim3 dimBlock(BLOCK_SIZE, 1, 1);

    
    // 1. Transfer the input image (the A matrix) to the device memory 
    clock_gettime(CLOCK_REALTIME, &start);

    int* A_d;
    cuda_ret = cudaMalloc((void**)&A_d, size);
    err_check(cuda_ret, (char*)"Unable to allocate A to device memory!", 1);
    cuda_ret = cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to transfer A from Host to Device!", 3);

    // 2. Transfer the convolution filter (the K matrix) to the device memory 
    int* K_d;
    cuda_ret = cudaMalloc((void**)&K_d, 5 * 5 * sizeof(int));
    err_check(cuda_ret, (char*)"Unable to allocate K to device memory!", 1);
    cuda_ret = cudaMemcpy(K_d, K, 5 * 5 * sizeof(int), cudaMemcpyHostToDevice);
    err_check(cuda_ret, (char*)"Unable to transfer K from Host to Device!", 3);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_spent = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;

    printf("Time to transfer matrices A and K to device: %f\n", time_spent);

    // 3. Launch the convolution kernel to compute the filter map (the B matrix) by applying the 
    // convolution to every pixel in the input image. 

    clock_gettime(CLOCK_REALTIME, &start);

    int* B_d;
    cuda_ret = cudaMalloc((void**)&B_d, size);
    err_check(cuda_ret, (char*)"Unable to allocate B to device memory!", 1);

    convolution_kernel<<< dimGrid, dimBlock >>>(A_d, K_d, B_d, n_col, n_row);
    cuda_ret = cudaDeviceSynchronize();
    err_check(cuda_ret, (char*)"Unable to launch convolution kernel!", 2);

    clock_gettime(CLOCK_REALTIME, &end);
    time_spent = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;
    printf("Time to launch convolution kernel on device: %f\n", time_spent);

    // 4. Transfer the filter map (the B matrix) from the device memory to the system memory. 
    clock_gettime(CLOCK_REALTIME, &start);

    cuda_ret = cudaMemcpy(B, B_d, size, cudaMemcpyDeviceToHost);
    err_check(cuda_ret, (char*)"Unable to read B from device memory!", 3);

    clock_gettime(CLOCK_REALTIME, &end);
    time_spent = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / BILLION;
    printf("Time to transfer B from device memory: %f\n", time_spent);

	// Save output matrix as csv file
    for (int i = 0; i<n_row; i++)
    {
        for (int j = 0; j<n_col; j++)
        {
            fprintf(outputFile, "%d", B[i*n_col +j]);
            if (j != n_col -1)
                fprintf(outputFile, ",");
            else if ( i < n_row-1)
                fprintf(outputFile, "\n");
        }
    }

    // Print time
    fprintf(timeFile, "%.20f", time_spent);

    // Cleanup
    fclose (outputFile);
    fclose (timeFile);

    free(A);
    free(B);
    free(K);

    return 0;
}

/* Error Check ----------------- //
*   Exits if there is a CUDA error.
*/
void err_check(cudaError_t ret, char* msg, int exit_code) {
    if (ret != cudaSuccess)
        fprintf(stderr, "%s \"%s\".\n", msg, cudaGetErrorString(ret)),
        exit(exit_code);
} // End Error Check ----------- //
