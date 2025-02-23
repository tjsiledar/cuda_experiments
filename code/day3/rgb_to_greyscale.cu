#include <iostream>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CHANNELS 3

using namespace std;

__global__ void kRgbToGreyScale(unsigned char *dInput, unsigned char *dOutput, int width, int height){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x<width && y<height){
        int rgb_idx = (y*width + x)*CHANNELS; //multiplied by 3 as it is rgb image
        int grey_idx = y*width + x;

        //get the rgb values
        unsigned char r = dInput[rgb_idx];
        unsigned char g = dInput[rgb_idx+1];
        unsigned char b = dInput[rgb_idx+2];

        //convert rgb to grey scale
        dOutput[grey_idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    
}

int main(){
    int width, height, channels;

    //read the image using stbi load and assign it to a pointer
    unsigned char *hInput = stbi_load("test_image.png", &width, &height, &channels, CHANNELS);
    cout << width << " " << height << " " << channels << endl;

    //compute the rgb and grey scale image size
    int rgb_size = width*height*CHANNELS;
    int grey_size = width*height;

    //allocate memory in host for the output image
    unsigned char *hOutput = new unsigned char[grey_size];

    //define device variables for input and output images
    unsigned char *dInput, *dOutput;
    cudaMalloc((void **)&dInput, rgb_size);
    cudaMalloc((void **)&dOutput, grey_size);

    //copy the input image from the host to device
    cudaMemcpy(dInput, hInput, rgb_size, cudaMemcpyHostToDevice);

    //call the kernel function
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(ceil(width/16.0), ceil(height/16.0), 1);
    kRgbToGreyScale<<<gridSize, blockSize>>>(dInput, dOutput, width, height);

    //copy the output image from the device to the host
    cudaMemcpy(hOutput, dOutput, grey_size, cudaMemcpyDeviceToHost);

    //save output image
    stbi_write_png("output.png", width, height, 1, hOutput, width);

    cudaFree(dInput); cudaFree(dOutput);
    stbi_image_free(hInput);
    delete[] hOutput;
    
    return 0;
}