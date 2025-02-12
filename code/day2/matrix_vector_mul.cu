#include <stdio.h>
#include <cuda.h>

#define N 100

__global__ void kernelMatVecMul(float *A, float *B, float *C){
    // get the thread number
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i<N){
        C[i] = 0;
        
        for(int j=0; j<N; j++){
            C[i] += A[i*N+j] * B[j];
        }
    }
}


void matVecMul(float A[N][N], float B[N], float C[N]){
    // define the variables for the device memory
    float *dA, *dB, *dC;
    int mat_size = N*N*sizeof(float);
    int vec_size = N*sizeof(float);
    

    // allocate memory in device for the matrices and vector
    cudaMalloc((void **)&dA, mat_size);
    cudaMalloc((void **)&dB, vec_size);
    cudaMalloc((void **)&dC, vec_size);

    // copy from host to device
    cudaMemcpy(dA, A, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, vec_size, cudaMemcpyHostToDevice);

    // calling the kernel to perform matrix to vector multiplication
    int num_threads = 256;
    int num_blocks = ceil(N/256.0);
    
    kernelMatVecMul<<<num_blocks, num_threads>>>(dA, dB, dC);

    // copy from device to host
    cudaMemcpy(C, dC, vec_size, cudaMemcpyDeviceToHost);

    // free device variables
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}


int main(){
    // define a matrix and a vector
    float A[N][N], B[N], C[N];

    // initialize the matrix and vector
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            A[i][j] = (i+j)*0.1f;
        }
        B[i] = i*0.3f;
    }

    // host stub function
    matVecMul(A, B, C);

    // print the matrix and vector
    printf("A Matrix:\n");
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            printf("%f\t",A[i][j]);
        }
        printf("\n");
    }
    printf("\nB Matrix:\n");
    for(int i=0; i<5; i++){
        printf("%f\t",B[i]);
    }
    printf("\n\nC Matrix:\n");
    for(int i=0; i<5; i++){
        printf("%f\t",C[i]);
    }
    printf("\n");

    return 0;
}
