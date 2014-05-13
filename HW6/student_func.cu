//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */



#include "utils.h"
#include <stdio.h>
#include <thrust/host_vector.h>

__global__ void computeMask(const uchar4* const sourceImg,
                            unsigned char * const mask,
                            size_t numPix){
  size_t pId = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(pId >= numPix)
    return;
  
  const uchar4 * pixel = sourceImg + pId;
  if(pixel->x == 255 && pixel->y == 255 && pixel->z == 255)
    mask[pId] = 0;
  else
    mask[pId] = 1;
}

__global__ void computeInterior(const unsigned char * const mask,
                                unsigned char * const interiorMask,
                                size_t numRows,
                                size_t numCols){
  size_t pId = blockIdx.x * blockDim.x + threadIdx.x;
  
  if( (pId >= numRows * numCols) || (mask[pId] == 0) )
    return;          
              
  if( mask[pId+1] && mask[pId-1] && mask[pId-numCols] && mask[pId+numCols] ){
    interiorMask[pId] = 1;
  }
}

__global__ void seperateImg(const uchar4 * const sourceImg,
                            float * const imgx,
                            float * const imgy,
                            float * const imgz,
                            size_t numRows,
                            size_t numCols){
  size_t pId = blockIdx.x * blockDim.x + threadIdx.x;
  
  if(pId >= numRows * numCols)
    return;   
    
  imgx[pId] = float(sourceImg[pId].x);
  imgy[pId] = float(sourceImg[pId].y);
  imgz[pId] = float(sourceImg[pId].z);                            
}

void allocateChannelBuffs(float ** d_srcimgx, 
                          float ** d_srcimgy,
                          float ** d_srcimgz,
                          float ** d_dstimgx,
                          float ** d_dstimgy,
                          float ** d_dstimgz,
                          float ** d_blendx1, 
                          float ** d_blendx2,
                          float ** d_blendy1,
                          float ** d_blendy2,
                          float ** d_blendz1,
                          float ** d_blendz2,
                          const size_t numRowsSource,
                          const size_t numColsSource){
                          
  checkCudaErrors(cudaMalloc(d_srcimgx, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_srcimgx, 0, sizeof(float) * numRowsSource * numColsSource));  
  checkCudaErrors(cudaMalloc(d_srcimgy, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_srcimgy, 0, sizeof(float) * numRowsSource * numColsSource));  
  checkCudaErrors(cudaMalloc(d_srcimgz, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_srcimgz, 0, sizeof(float) * numRowsSource * numColsSource));    

  checkCudaErrors(cudaMalloc(d_dstimgx, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_dstimgx, 0, sizeof(float) * numRowsSource * numColsSource));  
  checkCudaErrors(cudaMalloc(d_dstimgy, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_dstimgy, 0, sizeof(float) * numRowsSource * numColsSource));  
  checkCudaErrors(cudaMalloc(d_dstimgz, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_dstimgz, 0, sizeof(float) * numRowsSource * numColsSource));                           
                          
  checkCudaErrors(cudaMalloc(d_blendx1, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_blendx1, 0, sizeof(float) * numRowsSource * numColsSource));  
  checkCudaErrors(cudaMalloc(d_blendx2, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_blendx2, 0, sizeof(float) * numRowsSource * numColsSource));  

  checkCudaErrors(cudaMalloc(d_blendy1, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_blendy1, 0, sizeof(float) * numRowsSource * numColsSource));  
  checkCudaErrors(cudaMalloc(d_blendy2, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_blendy2, 0, sizeof(float) * numRowsSource * numColsSource)); 
  
  checkCudaErrors(cudaMalloc(d_blendz1, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_blendz1, 0, sizeof(float) * numRowsSource * numColsSource));  
  checkCudaErrors(cudaMalloc(d_blendz2, sizeof(float) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(*d_blendz2, 0, sizeof(float) * numRowsSource * numColsSource));      
}

__global__ void computeIteration(const float * const srcImg, 
                                 const float * const dstImg, 
                                 float * const blendprev,
                                 float * const blendnext,
                                 const unsigned char * const interiorMask, 
                                 const size_t numRows,
                                 const size_t numCols){

/*
   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]        
*/                      
  size_t pId = blockIdx.x * blockDim.x + threadIdx.x;
  
  if( (pId >= numRows * numCols) || (interiorMask[pId] == 0) )
    return;    
    
  float sum1 = 0.0;
  float sum2 = 0.0;
  
  if(interiorMask[pId-1]){
    sum1 += blendprev[pId-1];
    sum2 += srcImg[pId] - srcImg[pId-1];
  } else {
    sum1 += dstImg[pId-1];
    sum2 += srcImg[pId] - srcImg[pId-1];
  }
  
  if(interiorMask[pId+1]){
    sum1 += blendprev[pId+1];
    sum2 += srcImg[pId] - srcImg[pId+1];
  } else {
    sum1 += dstImg[pId+1];
    sum2 += srcImg[pId] - srcImg[pId+1];
  }
  
  if(interiorMask[pId-numCols]){
    sum1 += blendprev[pId-numCols];
    sum2 += srcImg[pId] - srcImg[pId-numCols];
  } else {
    sum1 += dstImg[pId-numCols];
    sum2 += srcImg[pId] - srcImg[pId-numCols];
  }
  
  if(interiorMask[pId+numCols]){
    sum1 += blendprev[pId+numCols];
    sum2 += srcImg[pId] - srcImg[pId+numCols];
  } else {
    sum1 += dstImg[pId+numCols];
    sum2 += srcImg[pId] - srcImg[pId+numCols];
  }      
  float newVal = (sum1 + sum2) / 4.f;
  blendnext[pId] = min(255.0, max(0.0, newVal));
}

__global__ void combineResults(uchar4 * const d_blendedImg, 
                               const float * const d_blendx, 
                               const float * const d_blendy,
                               const float * const d_blendz,
                               const uchar4 * const d_destImg,
                               const unsigned char * const interiorMask,
                               const size_t numRows,
                               const size_t numCols){
  size_t pId = blockIdx.x * blockDim.x + threadIdx.x;
                               
  if( (pId >= numRows * numCols) )
    return;    
  if(interiorMask[pId] == 1){
    d_blendedImg[pId].x = (unsigned char) d_blendx[pId];  
    d_blendedImg[pId].y = (unsigned char) d_blendy[pId]; 
    d_blendedImg[pId].z = (unsigned char) d_blendz[pId]; 
  } else {
    d_blendedImg[pId].x = (unsigned char) d_destImg[pId].x;  
    d_blendedImg[pId].y = (unsigned char) d_destImg[pId].y; 
    d_blendedImg[pId].z = (unsigned char) d_destImg[pId].z;    
    //d_blendedImg[pId].x = 20;
    //d_blendedImg[pId].y = 20;
    //d_blendedImg[pId].z = 20;      
  }                             
}

void your_blend(const uchar4* const h_sourceImg,  //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4* const h_destImg, //IN
                uchar4* const h_blendedImg) //OUT
{

  /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
  int threadsPerBlock = 512;
  int numBlocks = ceil(double(numRowsSource * numColsSource) / threadsPerBlock);
    
  uchar4 * d_sourceImg;
  checkCudaErrors(cudaMalloc( &d_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemcpy( d_sourceImg, h_sourceImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice));
  uchar4 * d_destImg;
  checkCudaErrors(cudaMalloc( &d_destImg, sizeof(uchar4) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemcpy( d_destImg, h_destImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyHostToDevice));  
  // compute mask
  unsigned char * d_mask;
  checkCudaErrors(cudaMalloc(&d_mask, sizeof(unsigned char) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(d_mask, 0, sizeof(unsigned char) * numRowsSource * numColsSource));
  computeMask<<<numBlocks, threadsPerBlock>>>(d_sourceImg, d_mask, numRowsSource * numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // compute interior
  unsigned char * d_interiorMask;
  checkCudaErrors(cudaMalloc(&d_interiorMask, sizeof(unsigned char) * numRowsSource * numColsSource));
  checkCudaErrors(cudaMemset(d_interiorMask, 0, sizeof(unsigned char) * numRowsSource * numColsSource));
  computeInterior<<<numBlocks, threadsPerBlock>>>(d_mask, d_interiorMask, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  float *d_srcimgx, *d_srcimgy, *d_srcimgz ,*d_dstimgx, *d_dstimgy, *d_dstimgz;
  float *d_blendx1, *d_blendx2, *d_blendy1, *d_blendy2, *d_blendz1, *d_blendz2;
  allocateChannelBuffs(&d_srcimgx, &d_srcimgy, &d_srcimgz, &d_dstimgx, &d_dstimgy, &d_dstimgz, &d_blendx1, &d_blendx2, &d_blendy1, &d_blendy2, &d_blendz1, &d_blendz2, numRowsSource, numColsSource);

  seperateImg<<<numBlocks,threadsPerBlock>>>(d_sourceImg, d_srcimgx, d_srcimgy, d_srcimgz, numRowsSource, numColsSource);  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  seperateImg<<<numBlocks,threadsPerBlock>>>(d_destImg, d_dstimgx, d_dstimgy, d_dstimgz, numRowsSource, numColsSource);  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    

    
  // initialize guess buffers
  checkCudaErrors(cudaMemcpy( d_blendx1, d_srcimgx, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy( d_blendx2, d_srcimgx, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy( d_blendy1, d_srcimgy, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy( d_blendy2, d_srcimgy, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy( d_blendz1, d_srcimgz, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy( d_blendz2, d_srcimgz, sizeof(float) * numRowsSource * numColsSource, cudaMemcpyDeviceToDevice));
  
  for(int i = 0; i < 800; i++){
    if(i % 2 == 0){
      computeIteration<<<numBlocks,threadsPerBlock>>>(d_srcimgx, d_dstimgx, d_blendx1, d_blendx2, d_interiorMask, numRowsSource, numColsSource);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      computeIteration<<<numBlocks,threadsPerBlock>>>(d_srcimgy, d_dstimgy, d_blendy1, d_blendy2, d_interiorMask, numRowsSource, numColsSource);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      computeIteration<<<numBlocks,threadsPerBlock>>>(d_srcimgz, d_dstimgz, d_blendz1, d_blendz2, d_interiorMask, numRowsSource, numColsSource);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    } else {
      computeIteration<<<numBlocks,threadsPerBlock>>>(d_srcimgx, d_dstimgx, d_blendx2, d_blendx1, d_interiorMask, numRowsSource, numColsSource);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      computeIteration<<<numBlocks,threadsPerBlock>>>(d_srcimgy, d_dstimgy, d_blendy2, d_blendy1, d_interiorMask, numRowsSource, numColsSource);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
      computeIteration<<<numBlocks,threadsPerBlock>>>(d_srcimgz, d_dstimgz, d_blendz2, d_blendz1, d_interiorMask, numRowsSource, numColsSource);   
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 
    }
  }

  uchar4 * d_blendedImg;
  checkCudaErrors(cudaMalloc( &d_blendedImg, sizeof(uchar4) * numRowsSource * numColsSource));
  
  combineResults<<<numBlocks,threadsPerBlock>>>(d_blendedImg, d_blendx1, d_blendy1, d_blendz1, d_destImg, d_interiorMask, numRowsSource, numColsSource);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  
  checkCudaErrors(cudaMemcpy( h_blendedImg, d_blendedImg, sizeof(uchar4) * numRowsSource * numColsSource, cudaMemcpyDeviceToHost));  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
