#include <stdio.h>
#include <cuda.h>

__global__ void vecAddKernel(float *A, float *B, float *C, int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    C[i] = A[i] + B[i];
}

int main(){
    // define the variables
    // host variables
    int N=10000;
    int size = N*sizeof(float);
    float A[N], B[N], C[N];

    // device variables
    float *d_A, *d_B, *d_C;

    // loop over the arrays in the host memory
    for(int i=0; i<10000; i++){
        A[i] = i*1.1f;
        B[i] = i*1.2f;
    }

    // allocate memory in the device memory and use device pointers to assign them
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // copy the arrays from the host memory to the device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // call the kernel func to compute the vector addition
    vecAddKernel<<<ceil(N/1000.0), 1000>>>(d_A, d_B, d_C, N);
    
    // copy the array from device back to the host memory
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    // print the arrays
    for(int i=0; i<5; i++){
        printf("%f, %f, %f\n", A[i], B[i], C[i]);
    }

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    return 0;
}