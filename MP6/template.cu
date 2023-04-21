// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 512
//@@ insert code here

__global__ void castImg(float *inputImage, unsigned char *ucharImage, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    ucharImage[idx] = (unsigned char) (255 * inputImage[idx]);
  }
}

__global__ void RGB2Gray(unsigned char *ucharImage, unsigned char *grayImage, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    unsigned char r = ucharImage[3 * idx];
    unsigned char g = ucharImage[3 * idx + 1];
    unsigned char b = ucharImage[3 * idx + 2];
    grayImage[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void computeHistogram(unsigned char * grayImage, int *output, int len) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  __shared__ unsigned int histogram[HISTOGRAM_LENGTH];

  if (tid < HISTOGRAM_LENGTH) {
    histogram[tid] = 0;
  }
  __syncthreads();

  if (idx < len) {
    atomicAdd(&(histogram[grayImage[idx]]), 1);
  }
  __syncthreads();

  if (tid < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[tid]), histogram[tid]);
  }
}

__global__ void scan(int *histogram, float *cdf, int len, int pixels) {
    __shared__ float XY[HISTOGRAM_LENGTH];
  int i = threadIdx.x;
  if (i < len){
    XY[threadIdx.x] = histogram[i];
  } else {
    XY[threadIdx.x] = 0;
  }
  if (i + blockDim.x < len){
    XY[threadIdx.x + blockDim.x] = histogram[i + blockDim.x];
  } else {
    XY[threadIdx.x + blockDim.x] = 0;
  }

  for (unsigned int stride = 1; stride <= blockDim.x; stride *=2){
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < len && index - stride >= 0){
      XY[index] += XY[index - stride];
    }
  }

  for (int stride = ceil(len/4.0); stride > 0; stride /= 2){
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if(index + stride < len){
      XY[index + stride] += XY[index];
    }

  }
  __syncthreads();
  
  if (i < len) cdf[i] = ((float)(XY[i] * 1.0)/pixels);
  if (i + blockDim.x < len) cdf[i+blockDim.x] = ((float)(XY[i+blockDim.x] * 1.0)/pixels);
}

__global__ void correctColor(unsigned char *ucharImage, float *cdf, int len) {
  __shared__ float cdf_shared[HISTOGRAM_LENGTH];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    cdf_shared[threadIdx.x] = cdf[threadIdx.x];
  }
  __syncthreads();

  if (idx < len) {
    unsigned char val = ucharImage[idx];
    float cdfmin = cdf_shared[0];
    float clampVal = 255 * (cdf_shared[val] - cdfmin) / (1.0 - cdfmin);
    ucharImage[idx] = (unsigned char) (min(max(clampVal, 0.0), 255.0));
  }
}
__global__ void castBack(unsigned char *ucharImage, float *outputImage, int len) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    outputImage[idx] = (float) (ucharImage[idx] / 255.0);
  }
}
int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;


  //@@ Insert more code here
  float *deviceInputImageData;
  unsigned char *deviceGrayImage;
  unsigned char * deviceUcharImage;
  int *deviceHistogram;
  float *devicecdf;
  float *deviceOutputImageData;
  
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");
  int pixels = imageHeight * imageWidth;
  int totalSize = pixels * imageChannels;

  cudaMalloc((void **) &deviceInputImageData, totalSize * sizeof(float));
  cudaMalloc((void **) &deviceUcharImage, totalSize * sizeof(unsigned char));
  cudaMalloc((void **) &deviceGrayImage, pixels * sizeof(unsigned char));
  cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(int));
  cudaMalloc((void **) &devicecdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **) &deviceOutputImageData, totalSize * sizeof(float));
  cudaMemcpy(deviceInputImageData, hostInputImageData, totalSize * sizeof(float), cudaMemcpyHostToDevice);
  //@@ insert code here

  dim3 DimBlock1(BLOCK_SIZE, 1, 1);
  dim3 DimGrid1(ceil(totalSize/(1.0*BLOCK_SIZE)), 1, 1);
  castImg<<<DimGrid1, DimBlock1>>>(deviceInputImageData, deviceUcharImage, totalSize);
  cudaDeviceSynchronize();

  dim3 DimBlock2(BLOCK_SIZE, 1, 1);
  dim3 DimGrid2(ceil(pixels/(1.0*BLOCK_SIZE)), 1, 1);
  RGB2Gray<<<DimGrid2, DimBlock2>>>(deviceUcharImage, deviceGrayImage, pixels);
  cudaDeviceSynchronize();

  dim3 DimBlock3(256, 1, 1);
  dim3 DimGrid3(ceil(pixels/256.0), 1, 1);
  computeHistogram<<<DimGrid3, DimBlock3>>>(deviceGrayImage, deviceHistogram, pixels);
  cudaDeviceSynchronize();

  dim3 DimBlock4(128, 1, 1);
  dim3 DimGrid4(1, 1, 1);
  scan<<<DimGrid4, DimBlock4>>>(deviceHistogram, devicecdf, 256, pixels);
  cudaDeviceSynchronize();
  

  dim3 DimBlock5(BLOCK_SIZE, 1, 1);
  dim3 DimGrid5(ceil(totalSize/(1.0*BLOCK_SIZE)), 1, 1);
  correctColor<<<DimGrid5, DimBlock5>>>(deviceUcharImage, devicecdf, totalSize);
  cudaDeviceSynchronize();

  dim3 DimBlock6(BLOCK_SIZE, 1, 1);
  dim3 DimGrid6(ceil(totalSize/(1.0*BLOCK_SIZE)), 1, 1);
  castBack<<<DimGrid6, DimBlock6>>>(deviceUcharImage, deviceOutputImageData, totalSize);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceOutputImageData, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);
  //@@ insert code here

  cudaFree(deviceInputImageData);
  cudaFree(deviceUcharImage);
  cudaFree(deviceGrayImage);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceHistogram);
  cudaFree(devicecdf);
  return 0;
}
