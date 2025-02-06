#include <stdio.h>
#include <cuda.h>

__global__ void vecaddkernel(float* A, float* B, float* C, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i<N) C[i] = A[i] + B[i];
}


void vec_add(float* A, float* B, float* C, int N)
{
    int size = N*sizeof(float);
    float *d_A, *d_B, *d_C;

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    vecaddkernel<<<4, 256>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


int main()
{
    int N=1000;
    float A[N];
    float B[N];
    float C[N];

    for(int i=0; i<N; i++){
        A[i] = i*0.1f;
        B[i] = i*0.2f;
    }

    vec_add(A, B, C, N);

    for(int i=0; i<5; i++){
        printf("%.2f, %.2f, %.2f\n", A[i], B[i], C[i]);
    }
    
    return 0;
}