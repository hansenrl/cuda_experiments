/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"
#include <stdio.h>
#include <float.h>

__global__ void findMinMaxKernel(const float * const logLuminance, float * const minLogLum, const int op, const int numRows,  const int numCols){
  int totalId = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  
  extern __shared__ float sharedLogLuminance[];
  
  if(totalId >= numRows * numCols)
    return;
    
  //if(totalId == 1000) printf("%0.2f\n", logLuminance[totalId]);
  
  sharedLogLuminance[tid] = logLuminance[totalId];
  __syncthreads();
  
  // do reduction
  for(unsigned int s = blockDim.x / 2; s >= 1; s >>= 1){
    if(tid < s){
      if(op == 0)
        sharedLogLuminance[tid] = min(sharedLogLuminance[tid],sharedLogLuminance[tid + s]);
      else
        sharedLogLuminance[tid] = max(sharedLogLuminance[tid],sharedLogLuminance[tid + s]);
    }
    __syncthreads();
  }
  
  //logLuminance[totalId] = sharedLogLuminance[totalId];
  
  if(tid == 0){
    //logLuminance[blockIdx.x] = sharedLogLuminance[0];
    minLogLum[blockIdx.x] = sharedLogLuminance[0];
  }
}

void findMinMax(const float * const d_logLuminance, float &min_logLum, float &max_logLum, const size_t numRows, const size_t numCols){
  const int threadsPerBlock = 256;
  const int numBlocks = numRows * numCols / threadsPerBlock + 1;  
  
  float h_minLogLum[numBlocks];
  float h_maxLogLum[numBlocks];
  for(int i = 0; i < numBlocks; i++){
    h_minLogLum[i] = FLT_MAX;
    h_maxLogLum[i] = FLT_MIN;
  }
  float * d_minLogLum;
  float * d_maxLogLum;
  checkCudaErrors(cudaMalloc((void **) &d_minLogLum, sizeof(float) * numBlocks));
  checkCudaErrors(cudaMalloc((void **) &d_maxLogLum, sizeof(float) * numBlocks));
  
  //printf("Number of blocks: %d\n", numBlocks);
  
  findMinMaxKernel<<<numBlocks,threadsPerBlock, sizeof(float) * threadsPerBlock>>>(d_logLuminance, d_minLogLum, 0, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  checkCudaErrors(cudaMemcpy(h_minLogLum,d_minLogLum, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
  
  min_logLum = FLT_MAX;
  for(int i = 0; i < numBlocks; i++){
    //printf("%0.2f\n", h_minLogLum[i]);
    if(h_minLogLum[i] < min_logLum)
      min_logLum = h_minLogLum[i]; 
  }
  
  findMinMaxKernel<<<numBlocks,threadsPerBlock, sizeof(float) * threadsPerBlock >>>(d_logLuminance, d_maxLogLum, 1, numRows, numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  checkCudaErrors(cudaMemcpy(h_maxLogLum,d_maxLogLum, sizeof(float) * numBlocks, cudaMemcpyDeviceToHost));
  
  max_logLum = -FLT_MAX;
  for(int i = 0; i < numBlocks; i++){
    if(h_maxLogLum[i] > max_logLum)
      max_logLum = h_maxLogLum[i]; 
  }  

  cudaFree(d_minLogLum);
  cudaFree(d_maxLogLum);  
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

__global__ void computeHistogramKernel(const float* const d_logLuminance, 
                                       unsigned int * const d_histogram, 
                                       const float lumMin, 
                                       float lumRange, 
                                       size_t numBins, 
                                       size_t length){
  size_t totalId = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(totalId >= length)
    return;

  int bin = (int) ((d_logLuminance[totalId] - lumMin) / lumRange * numBins);
  //if(bin < 10) printf("%d ", bin);
  if(bin == numBins) bin -= 1;
  
  if(bin >= numBins || bin < 0) { printf("Numbins: %d ", numBins); printf("Problem! Bin: %d\n", bin, numBins); }
  atomicAdd(d_histogram + bin, 1);
}

__global__ void computeCdfKernel(const unsigned int * const d_histogram, 
                                 unsigned int * const d_cdf, 
                                 unsigned int * const d_temp,
                                 unsigned int * const d_sumresults,
                                 size_t length){
  size_t totalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tId = threadIdx.x;
  
  if(totalId >= length)
    return;
  
  if(totalId > 0)
    d_temp[totalId] = d_histogram[totalId - 1];
  else
    d_temp[totalId] = 0;
  __syncthreads();
  
  for(int s = 1; s < length; s *= 2){
    //if(totalId == 1023) { printf("s: %d, ", s); printf("d_temp: %u ", d_temp[totalId]); printf("d_temp - s: %u\n", d_temp[totalId-s]);}
    //if(totalId == 510) { printf("780 s: %d, ", s); printf("d_temp: %u ", d_temp[totalId]); printf("d_temp - s: %u\n", d_temp[totalId-s]);}
    if(tId >= s){ // we don't want totalId - s to go out of bounds
      d_cdf[totalId] = d_temp[totalId] + d_temp[totalId - s];
    } else {
      d_cdf[totalId] = d_temp[totalId];
    } 
    __syncthreads();
    d_temp[totalId] = d_cdf[totalId];
    __syncthreads();
  }
  
  if(threadIdx.x == blockDim.x - 1){
    d_sumresults[blockIdx.x] = d_cdf[totalId] + d_histogram[totalId];
  }
}

__global__ void sumCdfKernel(unsigned int * const d_cdf, unsigned int * const d_sumresults, size_t length){
  size_t totalId = blockIdx.x * blockDim.x + threadIdx.x;
  if(totalId >= length)
    return;
  for(int s = 1; s < gridDim.x; s++){
    if(blockIdx.x >= s){ // don't want blockIdx.x - s to be out of bounds
      d_cdf[totalId] += d_sumresults[blockIdx.x - s];
    }
  }
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  findMinMax(d_logLuminance, min_logLum, max_logLum, numRows, numCols);
  printf("Min value: %0.2f, max value: %0.2f\n", min_logLum, max_logLum);
  float range = max_logLum - min_logLum;
  
  // HISTOGRAM
  int threadsPerBlock = 512;
  int numBlocks = numRows * numCols / threadsPerBlock + 1;  
  
  unsigned int h_histogram[numBins];
  memset(h_histogram, 0, sizeof(unsigned int) * numBins);
  unsigned int * d_histogram;
  checkCudaErrors(cudaMalloc((void **) &d_histogram, sizeof(int) * numBins));  
  checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(unsigned int) * numBins));
      
  computeHistogramKernel<<<numBlocks, threadsPerBlock>>>(d_logLuminance, d_histogram, min_logLum, range, numBins, numRows * numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));
  
  // SCAN
  threadsPerBlock = 512;
  numBlocks = ceil(float(numBins) / float(threadsPerBlock));    
  unsigned int * d_temp;
  checkCudaErrors(cudaMalloc((void **) &d_temp, sizeof(unsigned int) * numBins));      
  checkCudaErrors(cudaMemset(d_temp, 0, sizeof(unsigned int) * numBins));
  unsigned int * d_sumresults;
  checkCudaErrors(cudaMalloc((void **) &d_sumresults, sizeof(unsigned int) * numBlocks));      
  checkCudaErrors(cudaMemset(d_sumresults, 0, sizeof(unsigned int) * numBlocks));  
  
  computeCdfKernel<<<numBlocks, threadsPerBlock>>>(d_histogram, d_cdf, d_temp, d_sumresults, numBins); 
  sumCdfKernel<<<numBlocks, threadsPerBlock>>>(d_cdf, d_sumresults, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  
  unsigned int h_cdf[numBins];
  checkCudaErrors(cudaMemcpy(h_cdf, d_cdf, sizeof(unsigned int) * numBins, cudaMemcpyDeviceToHost));

  /*
  printf("\n\n");
  for(int i = 0; i < numBins; i++){
    printf("%u ", h_histogram[i]);
  }
  printf("\n\n\n");
  for(int i = 0; i < numBins; i++){
    printf("%u ", h_cdf[i]);
  }
  
  printf("\nnumber of bins: %lu\n", numBins);
  */
  cudaFree(d_histogram);
  cudaFree(d_temp);  
}
