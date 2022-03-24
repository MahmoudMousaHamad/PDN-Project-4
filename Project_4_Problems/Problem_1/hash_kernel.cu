#define MAX     123123123

/* Generate Hash ----------------------------------------- //
*   Generates a hash value from a nonce and transaction list.
*/
__global__
unsigned int my_generate_hash(unsigned int nonce, unsigned int index, unsigned int* transactions, unsigned int n_transactions) {

    

} // End Generate Hash ---------- //

/* Hash Kernel --------------------------------------
*       Generates an array of hash values from nonces.
*/
__global__
void hash_kernel(unsigned int* hash_array, unsigned int* nonce, unsigned int array_size, unsigned int* transactions, unsigned int n_transactions, unsigned int mod) {

    // Calculate thread index
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < array_size) {
        unsigned int hash = (nonce + transactions[0] * (index + 1)) % mod;
        for(int j = 1; j < n_transactions; j++){
            hash = (hash + transactions[j] * (index + 1)) % mod;
        }
        hash_array[index] = hash;
    }

} // End Hash Kernel //
