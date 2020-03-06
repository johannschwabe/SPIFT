
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

#include <stdio.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <chrono>


cudaError_t initData(const int matrixDim, const int blockDim, cuFloatComplex* res, float** shift);

constexpr int matrixIndex(const int matrix_dim, const int block_dim, int block_Idx, int block_Idy, int thread_idx) {
    return block_Idx * block_dim * matrix_dim + thread_idx * matrix_dim + block_Idy * block_dim;
}

constexpr int rowShiftedVectorIndex(const int matrix_dim, const int shift, const int index) {
    return ((shift + index) % matrix_dim) * 2;
}

__global__ void updateWithRowShift(cuFloatComplex* dev_matrix, cudaTextureObject_t dev_vector, const int matrix_dim, const int shift)
{

    if (blockIdx.x * blockDim.x + threadIdx.x < matrix_dim) {
        unsigned int start_index = matrixIndex(matrix_dim, blockDim.x, blockIdx.x, blockIdx.y, threadIdx.x);
        int end = std::min(matrix_dim - blockIdx.y * blockDim.x, blockDim.x);
        for (int i = 0; i < end; i++) {
            int vectorPos = rowShiftedVectorIndex(matrix_dim, shift * threadIdx.x + blockDim.x * blockIdx.y + blockIdx.x * blockDim.x, i);

            dev_matrix[start_index + i].x += tex2D<float>(dev_vector, vectorPos, 0);
            dev_matrix[start_index + i].y += tex2D<float>(dev_vector, vectorPos + 1, 0);

        }
    }

}




int main()
{


    const int matrixDim = 2048;
    const int blockDim = 32;

    cuFloatComplex* res_matrix = new cuFloatComplex[matrixDim * matrixDim];
    float* shift[30];
    for (int i = 0; i < 30; ++i) {
        float *bla = new float[2 * matrixDim];
        shift[i] = bla;
    }
    // Add vectors in parallel.
    cudaError_t cudaStatus = initData(matrixDim, blockDim, res_matrix, shift);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Cuda failed!");
        return 1;
    }

    for (int i = 0; i < 30; ++i) {
        free(shift[i]);
    }


    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    free(res_matrix);
    return 0;
}

void initTexture(int matrix_dim, float** shift, cudaTextureObject_t* texObj, cudaArray* *cuArray) {
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(cuArray, &channelDesc, matrix_dim * 2, 1);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = *cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL);
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "textureInit failed: %s\n", cudaGetErrorString(cudaStatus));
    }
}

int generateShift(int matrix_dim, float** shift, cudaArray* *cuArray) {
    int x = std::rand() % 30;
    cudaMemcpyToArray(*cuArray, 0, 0, shift[x], matrix_dim * 2 * sizeof(float), cudaMemcpyHostToDevice);
    return 1;
}

cudaError iteration(cuFloatComplex* dev_matrix, const int matrix_dim, const int block_dim, float** shift, cudaArray* *cuArray, cudaTextureObject_t* texObj) {
    int shift_length = generateShift(matrix_dim, shift, cuArray);

    int nr_blocks = ceil((double)matrix_dim / (double)block_dim);
    dim3* blockDim = new dim3(block_dim, 1);
    dim3* gridDim = new dim3(nr_blocks, nr_blocks);

    updateWithRowShift << <*gridDim, * blockDim >> > (dev_matrix, *texObj, matrix_dim, shift_length);



    // Check for any errors launching the kernel
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "updateWithRowShift launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching updateWithRowShift!\n", cudaStatus);
        return cudaStatus;
    }
    return cudaSuccess;


}

// Helper function to init data and launch iteration
cudaError_t initData(const int matrixDim, const int blockDim, cuFloatComplex* res, float** shift)
{
    cuFloatComplex* dev_matrix = 0;
    cudaError_t cudaStatus;

    

    for (int x = 0; x < matrixDim * matrixDim; ++x) {
        cuFloatComplex next;
        next.x = 0;
        next.y = 0;

        res[x] = next;
    }
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(1);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for matrix
    cudaStatus = cudaMalloc((void**)&dev_matrix, matrixDim * matrixDim * sizeof(cuFloatComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }
    
    
    // Copy input Matrix from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_matrix, res, matrixDim * matrixDim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }

    std::uniform_real_distribution<> dist(0, 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int j = 0; j < 30; ++j) {
        for (int x = 0; x < matrixDim * 2; ++x) {
            shift[j][x] = j;
            //shift[j][x] = dist(gen);
        }
    }
    cudaTextureObject_t texObj = 0;
    cudaArray* cuArray = 0;
    initTexture(matrixDim, shift, &texObj, &cuArray);
    std::cout << "lauched" << std::endl;
    int iterations = 100000;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cudaStatus = iteration(dev_matrix, matrixDim, blockDim, shift, &cuArray ,&texObj);

        // Check for any errors in iteration
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "iteration %d failed: %s\n", i, cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    std::cout << "terminated " << iterations << "  " << matrixDim << " dim, time: " << duration << "ms" << std::endl;
    std::cout << "time per iteration: " << duration / (double)iterations << std::endl;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(res, dev_matrix, matrixDim * matrixDim * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    /*
    for (int i = 0; i < matrixDim; ++i) {
        for (int j = 0; j < matrixDim; ++j) {
            std::cout << "(" << res[i * matrixDim + j].x << ", " << res[i * matrixDim + j].y << "), ";
        }
        std::cout << std::endl;
    }
    */
    return cudaSuccess;

}
