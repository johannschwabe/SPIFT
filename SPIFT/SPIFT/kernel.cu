
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

#include <stdio.h>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <chrono>

#include <fstream>
#include <thread>

constexpr int matrixStartIndexRow(const int matrix_dim, const int block_dim, int block_Idx, int block_Idy, int thread_idx) {
    return block_Idx * block_dim * matrix_dim + thread_idx * matrix_dim + block_Idy * block_dim;
}

constexpr int matrixStartIndexColumn(const int matrix_dim, const int block_dim, int block_Idx, int block_Idy, int thread_idx) {
    return block_Idx * block_dim * matrix_dim + block_Idy * block_dim + thread_idx;
}

constexpr int shiftedVectorIndex(const int matrix_dim, const int shift, const int index) {
    return ((shift + index) % matrix_dim) * 2;
}

__global__ void updateWithRowShift(cuFloatComplex* dev_matrix, cudaTextureObject_t dev_vector, const int matrix_dim, const int shift)
{

    if (blockIdx.x * blockDim.x + threadIdx.x < matrix_dim) {
        unsigned int start_index = matrixStartIndexColumn(matrix_dim, blockDim.x, blockIdx.x, blockIdx.y, threadIdx.x);
        int end = std::min(matrix_dim - blockIdx.x * blockDim.x, blockDim.x);
        for (int i = 0; i < end; i++) {
            int vectorPos = blockIdx.x * blockDim.x + i + (threadIdx.x + blockIdx.y * blockDim.x);
            dev_matrix[start_index + i * matrix_dim].x += tex2D<float>(dev_vector, vectorPos, 0);
            dev_matrix[start_index + i * matrix_dim].y += tex2D<float>(dev_vector, vectorPos + 1, 0);
        }
    }

}

__global__ void updateWithColumnShift(cuFloatComplex* dev_matrix, cudaTextureObject_t dev_vector, const int matrix_dim, const int shift)
{
    if (blockIdx.x * blockDim.x + threadIdx.x < matrix_dim) {
        unsigned int start_index = matrixStartIndexColumn(matrix_dim, blockDim.x, blockIdx.x, blockIdx.y, threadIdx.x);
        int end = std::min(matrix_dim - blockIdx.x * blockDim.x, blockDim.x);
        for (int i = 0; i < end; i++) {
            int vectorPos = shiftedVectorIndex(matrix_dim, shift * threadIdx.x + blockDim.x * blockIdx.y * shift + blockIdx.x * blockDim.x, i);

            dev_matrix[start_index + i * matrix_dim].x += tex2D<float>(dev_vector, vectorPos, 0);
            dev_matrix[start_index + i * matrix_dim].y += tex2D<float>(dev_vector, vectorPos + 1, 0);

        }
    }

}



class spift
{
public:
    spift(const int matrixDim, const int blockDim, const int iterations, const int GPU_index);
    ~spift();
    cudaError_t prepareGPU(int GPU_index);
    cudaError_t initTexture();
    cudaError_t iterate();
    cudaError_t iteration();
    auto displayTime();
    void printResult();
    int generateShift();

private:
    const int matrixDim;
    const int blockDim;
    cuFloatComplex* result;
    cuFloatComplex* dev_matrix;
    float** shift;
    cudaTextureObject_t texObj = 0;
    cudaArray* cuArray = 0;
    int iterations;
    long long duration;
};

spift::spift(const int matrixDim, const int blockDim, const int iterations, const int GPU_index) : matrixDim(matrixDim), blockDim(blockDim), iterations(iterations)
{

    cudaError_t cudaStatus;



    cudaStatus = this->initTexture();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU texture init failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    this->result = new cuFloatComplex[this->matrixDim * this->matrixDim];
    //allocate memory for result
    for (int x = 0; x < this->matrixDim * this->matrixDim; ++x) {
        cuFloatComplex next;
        next.x = 0;
        next.y = 0;
        this->result[x] = next;
    }
    cudaStatus = prepareGPU(GPU_index);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU init failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    //allocate memory for shift vectors. testing only
    std::uniform_real_distribution<> dist(0, 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    this->shift = new float* [30];
    for (int i = 0; i < 30; ++i) {
        float* bla = new float[2 * matrixDim];
        for (int x = 0; x < matrixDim * 2; ++x) {
            bla[x] = 1.0f;
            //bla[x] = dist(gen);
        }
        this->shift[i] = bla;
    }
}


cudaError_t spift::prepareGPU(int GPU_index) {
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(GPU_index);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }
    // Allocate GPU buffers for matrix
    cudaStatus = cudaMalloc((void**)&(this->dev_matrix), this->matrixDim * this->matrixDim * sizeof(cuFloatComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    // Copy input Matrix from host memory to GPU buffers.
    cudaStatus = cudaMemcpy((this->dev_matrix), (this->result), this->matrixDim * this->matrixDim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    return cudaSuccess;
}

auto spift::displayTime() {
    std::cout << "terminated " << iterations << "  " << matrixDim << " dim, time: " << duration << "ms" << std::endl;
    std::cout << "time per iteration: " << duration / (double)iterations << std::endl;
    return duration;
}

void spift::printResult() {
    for (int i = 0; i < matrixDim; ++i) {
        for (int j = 0; j < matrixDim; ++j) {
            std::cout << "(" << std::setfill(' ') << std::setw(2) << this->result[i * matrixDim + j].x << ", " << std::setfill(' ') << std::setw(2) << this->result[i * matrixDim + j].y << "), ";
        }
        std::cout << std::endl;
    }
}

cudaError_t spift::iterate() {
    cudaError_t cudaStatus;

    std::cout << "lauched" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < this->iterations; ++i) {
        cudaStatus = iteration();

        // Check for any errors in iteration
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "iteration %d failed: %s\n", i, cudaGetErrorString(cudaStatus));
            return cudaStatus;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(result, dev_matrix, matrixDim * matrixDim * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    return cudaSuccess;
}

int spift::generateShift() {
    int x = std::rand() % 30;
    cudaMemcpyToArray(cuArray, 0, 0, shift[x], matrixDim * 2 * sizeof(float), cudaMemcpyHostToDevice);
    return std::rand() % 30;
}

cudaError spift::iteration() {
    int shift_length = generateShift();

    int nr_blocks = ceil((double)matrixDim / (double)this->blockDim);
    dim3* blockDim = new dim3(this->blockDim, 1);
    dim3* gridDim = new dim3(nr_blocks, nr_blocks);

    updateWithColumnShift << <*gridDim, * blockDim >> > (dev_matrix, texObj, this->matrixDim, shift_length);
    /*
    if (std::rand() % 2) {
        updateWithRowShift << <*gridDim, *blockDim >> > (dev_matrix, texObj, this->matrixDim, shift_length);
    }
    else {
        updateWithColumnShift << <*gridDim, *blockDim >> > (dev_matrix, texObj, this->matrixDim, shift_length);
    }
    */

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
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        return cudaStatus;
    }
    return cudaSuccess;
}


cudaError_t spift::initTexture() {
    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&(this->cuArray), &channelDesc, this->matrixDim * 2, 1);

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = this->cuArray;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "textureInit failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    return cudaSuccess;
}

spift::~spift()
{

    free(this->result);
    for (int i = 0; i < 30; ++i) {
        free(this->shift[i]);
    }
    free(this->shift);
    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
}



void parallel(const int GPU_index, const int dim, std::ofstream *times) {
    spift* tester = new spift(dim, 3, 2, GPU_index);
    tester->iterate();
    *times << dim << "\t" << tester->displayTime() << "\t" << GPU_index << std::endl;
    tester->printResult();
    free(tester);
}



int main()
{
    std::ofstream times;
    times.open("timesGPU.txt");
    parallel(1, 12, &times);
    /*
    std::ofstream times;
    times.open("timesGPU.txt");

    for (int i = 0; i < 7; i++) {
        //std::thread t0(parallel, 0);
        std::thread t1(parallel, 1, std::pow(2, i + 8), &times);
        std::thread t2(parallel, 2, std::pow(2, i + 8), &times);
        std::thread t3(parallel, 3, std::pow(2, i + 8), &times);
        std::thread t4(parallel, 4, std::pow(2, i + 8), &times);
        std::thread t5(parallel, 5, std::pow(2, i + 8), &times);


        //t0.join();
        t1.join();
        t2.join();
        t3.join();
        t4.join();
        t5.join();

        //std::thread t0(parallel, 0);
        std::thread t10(parallel, 1, std::pow(2, i + 8), &times);
        std::thread t20(parallel, 2, std::pow(2, i + 8), &times);
        std::thread t30(parallel, 3, std::pow(2, i + 8), &times);
        std::thread t40(parallel, 4, std::pow(2, i + 8), &times);
        std::thread t50(parallel, 5, std::pow(2, i + 8), &times);


        //t0.join();
        t10.join();
        t20.join();
        t30.join();
        t40.join();
        t50.join();
    }
    */
    return 0;
}
