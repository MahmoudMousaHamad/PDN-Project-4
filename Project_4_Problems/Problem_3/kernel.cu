// program constants
#define BLOCK_SIZE 1024

#define BILLION  1000000000.0
#define MAX_LINE_LENGTH 25000

#define BLUR_SIZE 2

__global__
void convolution_kernel(int* A, int* K, int* out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        int pixVal = 0;
        // int pixels = 0;

        for (int r = -BLUR_SIZE; r < BLUR_SIZE + 1; ++r) {
            for (int c = -BLUR_SIZE; c < BLUR_SIZE + 1; ++c) {
                int current_row = Row + r;
                int current_col = Col + c;

                int i_row = r + BLUR_SIZE;
                int i_col = c + BLUR_SIZE;

                if (current_row > -1 && current_row < h && current_col > -1 && current_col < w) {
                    pixVal += A[current_row * w + current_col] * K[i_row * 5 + i_col];
                }
            }
        }
        out[Row * w + Col] = pixVal;
    }
}