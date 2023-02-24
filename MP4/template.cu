#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_WIDTH 8
#define KERNEL_WIDTH 3
#define INPUTTILE_WIDTH 10
//@@ Define constant memory for device kernel here
__constant__ float Kernel[KERNEL_WIDTH][KERNEL_WIDTH][KERNEL_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
   __shared__ float inputTile[INPUTTILE_WIDTH][INPUTTILE_WIDTH][INPUTTILE_WIDTH];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int x = blockIdx.x * TILE_WIDTH + tx;
  int y = blockIdx.y * TILE_WIDTH + ty;
  int z = blockIdx.z * TILE_WIDTH + tz;

  int radius = KERNEL_WIDTH / 2;
  int start_x = x - radius;
  int start_y = y - radius;
  int start_z = z - radius;

  float Pvalue = 0.0f;
  if(start_x >= 0 && start_x < x_size && start_y >= 0 && start_y < y_size && start_z >= 0 && start_z < z_size)
    inputTile[tz][ty][tx] = input[start_z * y_size * x_size + start_y * x_size + start_x];
  else
    inputTile[tz][ty][tx] = 0.0f;

  __syncthreads();


  if(tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH){
    for(int i = 0; i < KERNEL_WIDTH; i++)
      for(int j = 0; j < KERNEL_WIDTH; j++)
        for(int k = 0; k < KERNEL_WIDTH; k++){
          Pvalue += inputTile[tz + i][ty + j][tx + k] * Kernel[i][j][k];
        }
    if(x < x_size && y < y_size && z < z_size)
      output[z * y_size * x_size + y * x_size + x] = Pvalue;
  }

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength - 3) * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Kernel, hostKernel, kernelLength * sizeof(float), 0, cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size/(1.0*TILE_WIDTH)), ceil(y_size/(1.0*TILE_WIDTH)), ceil(z_size/(1.0*TILE_WIDTH)));
  dim3 dimBlock(INPUTTILE_WIDTH, INPUTTILE_WIDTH, INPUTTILE_WIDTH);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
