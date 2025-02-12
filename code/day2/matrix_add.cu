#include <stdio.h>
#include <cuda.h>

#define N 100

__global__ void kernelMatrixAddElementWise(float *A, float *B, float *C){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx<N*N) C[idx] = A[idx] + B[idx];
}


void matrixAdd(float A[N][N], float B[N][N], float C[N][N]){
    // initialize device memory pointers
    float *d_A, *d_B, *d_C;
    int size = N*N*sizeof(float);

    // allocate memory in device memory
    cudaMalloc((void **)&d_A, size); 
    cudaMalloc((void **)&d_B, size); 
    cudaMalloc((void **)&d_C, size); 

    // copy the matrices from the host to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    
    // call the kernel function to perform matrix addition
    kernelMatrixAddElementWise<<<100, 100>>>(d_A,d_B,d_C);

    // copy back the result matrix from device to host memory
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    // free cuda memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}


int main(){
    float A[N][N], B[N][N], C[N][N];
    
    // initialize the matrices
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            A[i][j] = (i+j)*0.1f;
            B[i][j] = (i+j)*0.2f;
        }
    }

    matrixAdd(A, B, C);
    
    // print the first few elements
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            printf("%f, %f, %f\t", A[i][j], B[i][j], C[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}