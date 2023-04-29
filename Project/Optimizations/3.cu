#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
__constant__ float constant_mask[10000];
__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int w_unrolled_rows = Map_out;
    int w_unrolled_cols = K * K * Channel;
    int X_unrolled_rows = w_unrolled_cols;
    int X_unrolled_cols = Width_out * Height_out;
    int Y_unrolled_rows = Map_out;
    int Y_unrolled_cols = X_unrolled_cols;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    __shared__ float subTilew[TILE_WIDTH][TILE_WIDTH];
    __shared__ float subTilex[TILE_WIDTH][TILE_WIDTH];

    // #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // #define mask_4d(i3, i2, i1, i0) constant_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // #define shared_2d(i1, i0) shared_mem[(i1) * (shared_width) + i0]
    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
	int by = blockIdx.y;
	int b = blockIdx.z;
    int Row = by*TILE_WIDTH+ty;
	int Col = bx*TILE_WIDTH+tx;


    float acc = 0.0f;
    for(int i = 0; i < (ceil(w_unrolled_cols*1.0/TILE_WIDTH)); i ++){
        if((Row < w_unrolled_rows) && ((i * TILE_WIDTH + tx) < w_unrolled_cols)){
            subTilew[ty][tx] = constant_mask[Row * (Channel * K * K) + i * TILE_WIDTH + tx];
        } else {
            subTilew[ty][tx] = 0;
        }
        if((Col < X_unrolled_cols) && ((i * TILE_WIDTH + ty) < X_unrolled_rows)){
            int h = Col / Width_out;
            int w = Col % Width_out;
            int p = (i * TILE_WIDTH + ty) % ( K * K) / K;
            int q = (i * TILE_WIDTH + ty) % ( K * K) % K;
            subTilex[ty][tx] = in_4d(b, (i * TILE_WIDTH + ty)/(K * K), h + p, w + q);
        } else {
            subTilex[ty][tx] = 0;
        }
        __syncthreads();
        if ((Row < Y_unrolled_rows) && (Col < Y_unrolled_cols)){
            for (int j = 0; j < TILE_WIDTH; j++){
                acc += subTilew[ty][j] * subTilex[j][tx];
            }
        }
        __syncthreads();
    }
    if ((Row < Y_unrolled_rows) && (Col < Y_unrolled_cols)){
        output[b * (Map_out * Height_out * Width_out) + Row * Y_unrolled_cols + Col] = acc;
    }

    #undef in_4d
}

	

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void **)device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **)device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));
    // cudaMalloc((void **)device_mask_ptr, K * K * Map_out * Channel * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, K * K * Channel *Map_out * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(constant_mask, host_mask, K * K * Channel *Map_out * sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int Y_unrolled_rows = Map_out;
    int Y_unrolled_cols = Width_out * Height_out;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(ceil(Y_unrolled_cols*1.0/TILE_WIDTH), ceil(Y_unrolled_rows * 1.0 / TILE_WIDTH), Batch);

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    // cudaFree(device_mask);

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
