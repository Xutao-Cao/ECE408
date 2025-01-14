//113ms
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
#define TILE_WIDTH 16
__constant__ __half constant_mask[10000];

__global__ void float_to_half(__half * output, const float* input, const int length){
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l < length) {
        output[l] = __float2half(input[l]);
    }
}
__global__ void half_to_float(float * output, __half* input, const int length){
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (l < length) {
        output[l] = __half2float(input[l]);
    }
}
__global__ void conv_forward_kernel(__half *  output, const __half *  input, const __half *  mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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
    // extern __shared__ __half shared_mem[];
    // int shared_width = TILE_WIDTH + K - 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // #define shared_2d(i1, i0) shared_mem[(i1) * (shared_width) + i0]
    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int W_grid = ceil(1.0 * Width_out / TILE_WIDTH);
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int tiled_h = (blockIdx.y / W_grid) * TILE_WIDTH;
    int tiled_w = (blockIdx.y % W_grid) * TILE_WIDTH;
    int b = blockIdx.z;

    __half acc = 0;
    for (int c = 0; c < Channel; c++){
        // for(int i = ty; i < shared_width; i += TILE_WIDTH ){
        //     for(int j = tx; j < shared_width; j += TILE_WIDTH){
        //         if(tiled_h + i < Height && tiled_w + j < Width){
        //             shared_2d(i, j) = in_4d(b, c, tiled_h + i, tiled_w + j);
        //         }
        //     }
        // }
        // __syncthreads();
        for(int p = 0; p < K; p++){
            for (int q = 0; q < K; q ++){
                // if (h + p < Height && w + q < Width)
                acc = __hadd(acc, __hmul(in_4d(b, c, h + p, w + q), mask_4d(m, c, p, q)));
            }
        }
        __syncthreads();
    }

    if (h >= Height_out || w >= Width_out){
        return;
    }

    out_4d(b, m, h, w) = acc;

    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef shared_2d
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
    cudaMalloc((void **)device_mask_ptr, K * K * Map_out * Channel * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, K * K * Channel *Map_out * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(constant_mask, host_mask, K * K * Channel *Map_out * sizeof(float));

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    __half* device_input_half;
    __half* device_output_half;
    __half* device_mask_half;
    
    
    // Set the kernel dimensions and call the kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    int input_length = Batch * Channel * Height * Width;
    int mask_length =  K * K * Map_out * Channel;

    cudaMalloc((void **)&device_input_half, input_length * sizeof(__half));
    cudaMalloc((void **)&device_output_half, Batch * Map_out * Height_out * Width_out * sizeof(__half));
    cudaMalloc((void **)&device_mask_half, mask_length * sizeof(__half));

    // __half* host_mask_half = (__half*)malloc(mask_length * sizeof(__half));
    float_to_half<<<ceil(1.0*input_length/256), 256>>>(device_input_half, device_input, input_length);
    cudaDeviceSynchronize();
    float_to_half<<<ceil(1.0*mask_length/256), 256>>>(device_mask_half, device_mask, mask_length);
    cudaDeviceSynchronize();
    // cudaMemcpy(host_mask_half, device_mask_half, mask_length * sizeof(__half), cudaMemcpyDeviceToHost);
    // cudaMemcpyToSymbol(constant_mask, host_mask_half, mask_length * sizeof(__half));

    int W_grid = ceil(1.0*Width_out/TILE_WIDTH);
    int H_grid = ceil(1.0*Height_out/TILE_WIDTH);
    int Y = H_grid * W_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(Map_out, Y, Batch);

    conv_forward_kernel<<<gridDim, blockDim>>>(device_output_half, device_input_half, device_mask_half, Batch, Map_out, Channel, Height, Width, K);
    cudaDeviceSynchronize();

    int output_length = Batch * Map_out * Height_out * Width_out;
    half_to_float<<<ceil(1.0 * output_length/256), 256>>>(device_output, device_output_half, output_length);
    // cudaDeviceSynchronize();
    // cudaFree(device_input_half);
    // cudaFree(device_output_half);
    // cudaFree(device_mask_half);
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
