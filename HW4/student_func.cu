//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
#include <limits.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
 
__global__ void mapKernel(unsigned int* const d_inputVals, 
                          unsigned int* const d_mapOut, 
                          size_t numElems, 
                          unsigned int mapVal,
                          unsigned int group){
  int id = blockIdx.x * blockDim.x + threadIdx.x;    
  if(id >= numElems)
    return;
  if((d_inputVals[id] & mapVal) == group * mapVal) 
    d_mapOut[id] = 1;
  //if(id < 10){
  //  printf("0x%x==0x%x ", d_inputVals[id] & mapVal, group);
  //}
}

__global__ void mapAddKernel(unsigned int* const d_scanValues,
                             unsigned int* const d_sortAddresses,
                             unsigned int* const d_radixGroupArray, 
                             size_t numElems, 
                             unsigned int addVal){
  int id = blockIdx.x * blockDim.x + threadIdx.x;    
  
  if(id >= numElems)
    return;
  if(d_radixGroupArray[id] == 1){
    d_sortAddresses[id] = d_scanValues[id] + addVal;
    if(d_sortAddresses[id] > numElems){
      printf("Uh-oh...problem! d_sortAddresses[id]: %u \n", d_sortAddresses[id] );  
    }
  }
}

__global__ void scanKernel(unsigned int* const d_inputVals, 
                           unsigned int* const d_scanValues, 
                           unsigned int* const d_temp,
                           unsigned int* const d_sumresults, 
                           size_t numElems){
  size_t totalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t tId = threadIdx.x;
  
  if(totalId >= numElems)
    return;  
    
  if(tId > 0)
    d_temp[totalId] = d_inputVals[totalId - 1];
  else
    d_temp[totalId] = 0;
    
  __syncthreads();
  
  for(int s = 1; s < numElems; s *= 2){
    //if(totalId == 1023) { printf("s: %d, ", s); printf("d_temp: %u ", d_temp[totalId]); printf("d_temp - s: %u\n", d_temp[totalId-s]);}
    //if(totalId == 510) { printf("780 s: %d, ", s); printf("d_temp: %u ", d_temp[totalId]); printf("d_temp - s: %u\n", d_temp[totalId-s]);}
    if(tId >= s){ // we don't want totalId - s to go out of bounds
      d_scanValues[totalId] = d_temp[totalId] + d_temp[totalId - s];
    } else {
      d_scanValues[totalId] = d_temp[totalId];
    } 
    __syncthreads();
    d_temp[totalId] = d_scanValues[totalId];
    __syncthreads();
  }
  
  if(threadIdx.x == blockDim.x - 1){ // need to total up the sum so it can be added to the block afterward
    d_sumresults[blockIdx.x] = d_scanValues[totalId] + d_inputVals[totalId]; // need to add d_inputValues because the scanValues array contains exclusive sums
  }  
}

__global__ void sumScanKernel(unsigned int * const d_scanValues, unsigned int * const d_sumresults, size_t numElems){
  size_t totalId = blockIdx.x * blockDim.x + threadIdx.x;
  if(totalId >= numElems)
    return;
  for(int s = 1; s < gridDim.x; s++){
    if(blockIdx.x >= s){ // don't want blockIdx.x - s to be out of bounds
      d_scanValues[totalId] += d_sumresults[blockIdx.x - s];
    }
  }
}

__global__ void resortAddresses(unsigned int * const d_outputVals, 
                                unsigned int * const d_outputPos, 
                                unsigned int * const d_sortAddresses, 
                                unsigned int * const d_temp,
                                unsigned int * const d_temp2,
                                size_t numElems,
                                unsigned int stage){
  size_t totalId = blockIdx.x * blockDim.x + threadIdx.x;
  if(totalId >= numElems)
    return;
    
  if(stage == 0){
    d_temp[totalId] = d_outputVals[totalId];
    d_temp2[totalId] = d_outputPos[totalId];
  } else {
    if( d_sortAddresses[totalId] > numElems ){
      //printf("Uh-oh...problem! d_sortAddresses[totalId]: %u \n", d_sortAddresses[totalId] );  
      //printf("numElems: %u\n", numElems);
    }
    d_outputVals[d_sortAddresses[totalId]] = d_temp[totalId];
    d_outputPos[d_sortAddresses[totalId]] = d_temp2[totalId];
  }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
  //TODO
  //PUT YOUR SORT HERE
  
  int threadsPerBlock = 512;
  int numBlocks = ceil(float(numElems) / threadsPerBlock);   
  
  unsigned int h_temp[numElems]; 
  
  unsigned int * d_mapOutGroup0;
  checkCudaErrors(cudaMalloc( (void **) &d_mapOutGroup0, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemset( d_mapOutGroup0, 0, sizeof(unsigned int) * numElems));
  unsigned int * d_mapOutGroup1;
  checkCudaErrors(cudaMalloc( (void **) &d_mapOutGroup1, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemset( d_mapOutGroup1, 0, sizeof(unsigned int) * numElems));  
  
  unsigned int * d_scanValues;
  unsigned int * d_temp;
  unsigned int * d_temp2;
  unsigned int * d_sumResults;
  
  checkCudaErrors(cudaMalloc( (void **) &d_scanValues, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMalloc( (void **) &d_temp, sizeof(unsigned int) * numElems));  
  checkCudaErrors(cudaMalloc( (void **) &d_temp2, sizeof(unsigned int) * numElems));    
  checkCudaErrors(cudaMalloc( (void **) &d_sumResults, sizeof(unsigned int) * numBlocks));
  checkCudaErrors(cudaMemset( d_scanValues, 0, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemset( d_temp, 0, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemset( d_temp2, 0, sizeof(unsigned int) * numElems));  
  checkCudaErrors(cudaMemset( d_sumResults, 0, sizeof(unsigned int) * numBlocks));
  
  unsigned int * d_sortAddresses;
  checkCudaErrors(cudaMalloc( (void **) &d_sortAddresses, sizeof(unsigned int) * numElems));
  checkCudaErrors(cudaMemset( d_sortAddresses, 0, sizeof(unsigned int) * numElems));  
  
  checkCudaErrors(cudaMemcpy( d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy( d_outputPos, d_inputPos, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));  
      
  //for loop (2^20)
  //for(unsigned int i = 1; i <= UINT_MAX/2; i <<= 1){
  for(int i = 1; i <= 4; i <<= 1){ // 1040146228
    checkCudaErrors(cudaMemset( d_mapOutGroup0, 0, sizeof(unsigned int) * numElems));  
    checkCudaErrors(cudaMemset( d_mapOutGroup1, 0, sizeof(unsigned int) * numElems)); 
    checkCudaErrors(cudaMemset( d_scanValues, 0, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemset( d_temp, 0, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemset( d_temp2, 0, sizeof(unsigned int) * numElems));  
    checkCudaErrors(cudaMemset( d_sumResults, 0, sizeof(unsigned int) * numBlocks));  
    checkCudaErrors(cudaMemset( d_sortAddresses, 0, sizeof(unsigned int) * numElems));   
     
    // map to seperate into two groups
    mapKernel<<<numBlocks, threadsPerBlock>>>(d_outputVals, d_mapOutGroup0, numElems, i, 0);
    mapKernel<<<numBlocks, threadsPerBlock>>>(d_outputVals, d_mapOutGroup1, numElems, i, 1);
    
    // exclusive sum scan to get scatter addresses
    scanKernel<<<numBlocks, threadsPerBlock>>>(d_mapOutGroup0, d_scanValues, d_temp, d_sumResults, numElems);
/*  checkCudaErrors(cudaMemcpy( h_temp, d_scanValues, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  int sum2 = 0;
  printf("\n");
  for(int j = threadsPerBlock; j < threadsPerBlock*2; j++){
    printf("%u ", h_temp[j]);
    //sum2 += h_temp[j];
  }
  printf("\n");
  checkCudaErrors(cudaMemcpy( h_temp, d_mapOutGroup0, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  sum2--;
  for(int j = threadsPerBlock; j < threadsPerBlock*2; j++){
    sum2 += h_temp[j];
    printf("%u ", sum2);
  }  
  printf("\n\n");  */     
    sumScanKernel<<<numBlocks, threadsPerBlock>>>(d_scanValues, d_sumResults, numElems);
    
    unsigned int adder;
    checkCudaErrors(cudaMemcpy( &adder, d_scanValues + numElems - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));    
    adder++;
    printf("Adder: %u ", adder);
    printf("i: %u ", i);
    // copy the addresses for group 0 into sortAddresses
    checkCudaErrors(cudaMemcpy( d_sortAddresses, d_scanValues, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset( d_scanValues, 0, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemset( d_temp, 0, sizeof(unsigned int) * numElems));
    checkCudaErrors(cudaMemset( d_sumResults, 0, sizeof(unsigned int) * numBlocks)); 

    // do a scan to get the delta addresses for group 1
    scanKernel<<<numBlocks, threadsPerBlock>>>(d_mapOutGroup1, d_scanValues, d_temp, d_sumResults, numElems);
    sumScanKernel<<<numBlocks, threadsPerBlock>>>(d_scanValues, d_sumResults, numElems); 
    cudaDeviceSynchronize();
     
    // figure out what to add to the group 1 addresses 
    mapAddKernel<<<numBlocks, threadsPerBlock>>>(d_scanValues, d_sortAddresses, d_mapOutGroup1, numElems, adder);
    
   
    // resort by new scatter addresses
    checkCudaErrors(cudaMemset( d_temp, 0, sizeof(unsigned int) * numElems));
    resortAddresses<<<numBlocks, threadsPerBlock>>>(d_outputVals, d_outputPos, d_sortAddresses, d_temp, d_temp2, numElems, 0);
    resortAddresses<<<numBlocks, threadsPerBlock>>>(d_outputVals, d_outputPos, d_sortAddresses, d_temp, d_temp2, numElems, 1);   
    
  checkCudaErrors(cudaMemcpy( h_temp, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  
  unsigned int tempoutput = 0;
  if(((h_temp[adder-1] & i) == i) && ((h_temp[adder] & i) == 0)){
  //if((h_temp[adder-1] & (i * 2 - 1)) > (h_temp[adder] & (i * 2 - 1))){
    printf("Swap! ");
    tempoutput = h_temp[adder-1];
    h_temp[adder-1] = h_temp[adder];
    h_temp[adder] = tempoutput;
  }
  //printf("h_temp[adder-1]: %u ", (h_temp[adder-1] & i));
  //printf("h_temp[adder]: %u ", (h_temp[adder] & i));
  checkCudaErrors(cudaMemcpy( d_outputVals, h_temp, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));
  
  unsigned int sum2 = 0;
  for(int j = 0; j < adder; j++){
    if((h_temp[j] & i) == i){
      sum2++;
      //printf("h_temp[j]: %u ", h_temp[j]);
    }
  }
  printf("Problem: %u ", sum2);  
  if(sum2 > 0){
    printf("Val1: %u ", h_temp[adder-1]);
    printf("Val2: %u ", h_temp[adder]);
  }
  sum2 = 0;
  for(int j = adder; j < numElems; j++){
    if((h_temp[j] & i) == 0){
      sum2++;
    }
  }
  printf("Problem: %u ", sum2);  
  if(sum2 > 0){
    printf("Val: %u ", h_temp[adder]);
  }  
  
    printf("\n");
  }
  
  // DEBUG
  int numPrint = 20;
  checkCudaErrors(cudaMemcpy( h_temp, d_sortAddresses, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  printf("\n");
  for(int j = numElems - 1; j > numElems - 20; j--){
    printf("%u ", h_temp[j]);
  }
  printf("\n");    
  checkCudaErrors(cudaMemcpy( h_temp, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  printf("\n");
  for(int j = numElems-1; j > numElems-20; j--){
    printf("%u ", h_temp[j]);
  }
  printf("\n");    
  unsigned int count = 0;
  checkCudaErrors(cudaMemcpy( h_temp, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
  printf("\n");
  for(int j = 1; j < numElems; j++){
    if((h_temp[j] & 7) < (h_temp[j-1] & 7)){
      count++;
      printf("h_temp[j-1]: %u ", h_temp[j-1]);
      printf("h_temp[j]: %u\n", h_temp[j]);
    }
  }
  printf("Count: %u\n", count);        
  printf("Numblocks: %d\n", numBlocks);
  //endfor
  
  printf("Number of elements: %lu", numElems);
  
  cudaFree(d_mapOutGroup0);
  cudaFree(d_mapOutGroup1);
  cudaFree(d_scanValues);
  cudaFree(d_temp);
  cudaFree(d_sumResults);
  cudaFree(d_sortAddresses);  
}
