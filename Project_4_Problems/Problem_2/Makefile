NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include  
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE			= gpu_mining_problem2
OBJ			= gpu_mining_problem2.o support.o

default: $(EXE)

gpu_mining_problem2.o: gpu_mining_problem2.cu nonce_kernel.cu hash_kernel.cu reduction_kernel.cu support.h
	$(NVCC) -c -o $@ gpu_mining_problem2.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
