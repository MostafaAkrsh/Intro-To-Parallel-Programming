#include "utils.h"

#include <stdio.h>
#include "device_launch_parameters.h"

__global__ void gaussian_blur(const unsigned char* const inputChannel,
    unsigned char* const outputChannel,
    int numRows, int numCols,
    const float* const filter, const int filterWidth)
{


    // NOTE: Be sure to compute any intermediate results in floating point
    // before storing the final result as unsigned char.

    // NOTE: Be careful not to try to access memory that is outside the bounds of
    // the image. You'll want code that performs the following check before accessing
    // GPU memory:
    //
    // if ( absolute_image_position_x >= numCols ||
    //      absolute_image_position_y >= numRows )
    // {
    //     return;
    // }

    // NOTE: If a thread's absolute position 2D position is within the image, but some of
    // its neighbors are outside the image, then you will need to be extra careful. Instead
    // of trying to read such a neighbor value from GPU memory (which won't work because
    // the value is out of bounds), you should explicitly clamp the neighbor values you read
    // to be within the bounds of the image. If this is not clear to you, then please refer
    // to sequential reference solution for the exact clamping semantics you should follow.


    //Compute the thread's row and column
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
                             blockIdx.y * blockDim.y + threadIdx.y);

    const int m = p.y * numCols + p.x;

    if (p.x >= numCols || p.y >= numRows)
        return;


    //First threads in every block copy the filter into shared memory
    extern __shared__ float sfilter[250];

    if (threadIdx.x < filterWidth && threadIdx.y < filterWidth)
    {
        int lin_index = threadIdx.x * filterWidth + threadIdx.y;
        sfilter[lin_index] = filter[lin_index];
    }

    __syncthreads();


    // The following loop walks over all coefficients in the filter

    float color = 0.0f;

    for (int f_y = 0; f_y < filterWidth; f_y++) {
        for (int f_x = 0; f_x < filterWidth; f_x++) {

            int c_x = p.x + f_x - filterWidth / 2;
            int c_y = p.y + f_y - filterWidth / 2;
            c_x = min(max(c_x, 0), numCols - 1);
            c_y = min(max(c_y, 0), numRows - 1);
            float filter_value = sfilter[f_y * filterWidth + f_x];
            color += filter_value * static_cast<float>(inputChannel[c_y * numCols + c_x]);

        }
    }

    outputChannel[m] = color;
}
//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
    int numRows,
    int numCols,
    unsigned char* const redChannel,
    unsigned char* const greenChannel,
    unsigned char* const blueChannel)
{
    // TODO
    //
    // NOTE: Be careful not to try to access memory that is outside the bounds of
    // the image. You'll want code that performs the following check before accessing
    // GPU memory:
    //

    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);
    const int m = p.y * numCols + p.x;

    if (p.x >= numCols || p.y >= numRows)
        return;
    redChannel[m] = inputImageRGBA[m].x;
    greenChannel[m] = inputImageRGBA[m].y;
    blueChannel[m] = inputImageRGBA[m].z;
}

//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
    const unsigned char* const greenChannel,
    const unsigned char* const blueChannel,
    uchar4* const outputImageRGBA,
    int numRows,
    int numCols)
{
    const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
        blockIdx.y * blockDim.y + threadIdx.y);

    const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

    //make sure we don't try and access memory outside the image
    //by having any threads mapped there return early
    if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
        return;

    unsigned char red = redChannel[thread_1D_pos];
    unsigned char green = greenChannel[thread_1D_pos];
    unsigned char blue = blueChannel[thread_1D_pos];

    //Alpha should be 255 for no transparency
    uchar4 outputPixel = make_uchar4(red, green, blue, 255);

    outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char* d_red, * d_green, * d_blue;
float* d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
    const float* const h_filter, const size_t filterWidth)
{

    //allocate memory for the three different channels
    //original
    checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
    checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

    //TODO:
    //Allocate memory for the filter on the GPU
    //Use the pointer d_filter that we have already declared for you
    //You need to allocate memory for the filter with cudaMalloc
    //be sure to use checkCudaErrors like the above examples to
    //be able to tell if anything goes wrong
    //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
    checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));
    //TODO:
    //Copy the filter on the host (h_filter) to the memory you just allocated
    //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
    //Remember to use checkCudaErrors!
    checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth, cudaMemcpyHostToDevice));

}

void your_gaussian_blur(const uchar4* const h_inputImageRGBA, uchar4* const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
    unsigned char* d_redBlurred,
    unsigned char* d_greenBlurred,
    unsigned char* d_blueBlurred,
    const int filterWidth)
{
    //TODO: Set reasonable block size (i.e., number of threads per block)
    const dim3 blockSize(32, 32);

    //TODO:
    //Compute correct grid size (i.e., number of blocks per kernel launch)
    //from the image size and and block size.
    const dim3 gridSize(numCols / blockSize.x + 1, numRows / blockSize.y + 1);


    //TODO: Launch a kernel for separating the RGBA image into different color channels
    separateChannels << <gridSize, blockSize >> > (d_inputImageRGBA,
        numRows,
        numCols,
        d_red,
        d_green,
        d_blue);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


    //TODO: Call your convolution kernel here 3 times, once for each color channel.
    gaussian_blur << <gridSize, blockSize >> > (
        d_red,
        d_redBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    gaussian_blur << <gridSize, blockSize >> > (
        d_blue,
        d_blueBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    gaussian_blur << <gridSize, blockSize >> > (
        d_green,
        d_greenBlurred,
        numRows,
        numCols,
        d_filter,
        filterWidth);
    // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
    // launching your kernel to make sure that you didn't make any mistakes.
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Now we recombine your results. We take care of launching this kernel for you.
    //
    // NOTE: This kernel launch depends on the gridSize and blockSize variables,
    // which you must set yourself.
    recombineChannels << <gridSize, blockSize >> > (d_redBlurred,
        d_greenBlurred,
        d_blueBlurred,
        d_outputImageRGBA,
        numRows,
        numCols);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
    checkCudaErrors(cudaFree(d_red));
    checkCudaErrors(cudaFree(d_green));
    checkCudaErrors(cudaFree(d_blue));
}
