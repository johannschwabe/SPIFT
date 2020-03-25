
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuComplex.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <fstream>

#include <mutex>
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
            int vectorPos = (shift * (blockIdx.x * blockDim.x + i) + (threadIdx.x + blockIdx.y * blockDim.x)) % matrix_dim * 2;
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
    spift(const int matrixDim, const int blockDim, const int iterations, const int GPU_index, const int wait);
    ~spift();
    spift(const spift&) = delete;
    cudaError_t prepareGPU();
    cudaError_t initTexture();
    cudaError_t iterate();
    cudaError_t iteration(int shift);
    auto displayTime();
    void printResult();
    void generateShifts();
    void initResult();
    void initCoalescence();
    
    void getShiftVector(int i);
    void initShiftTest();
    
    

private:
    //the dimensions of the grid and blocks
    const int matrixDim;
    const int blockDim;

    dim3* blockDim3;
    dim3* gridDim3;

    //the Matrix where the result is loaded into in the end
    cuFloatComplex* result;

    //the device-matrix, where the current state is saved
    cuFloatComplex* dev_matrix;

    //a set of randomgenerated shifts. testing only
    float** shift;

    //the texture object where the current shift is saved during kernel execution
    cudaTextureObject_t* texObj = new cudaTextureObject_t();

    //the data in texobj
    cudaArray* *cuArray = new cudaArray * ();

    //the number of iterations, testing only
    int iterations;

    //measuring the execution time
    long long duration;

    //the aggregation of shifts, first half are rowShifts
    float** coalescence;

    //the index, where it is saved, wheter data is aggregated for this shift
    int* coalescenceSet;

    std::mutex** shiftIndexMutex;

    //boolean wheter execution is done
    int* done;

    int wait;

    const int GPUIndex;



};

spift::spift(const int matrixDim, const int blockDim, const int iterations, const int GPU_index, const int wait) : matrixDim(matrixDim), blockDim(blockDim), iterations(iterations), GPUIndex(GPU_index), wait(wait)
{
    cudaError_t cudaStatus;

    this->initResult();

    this->initCoalescence();

    this->initShiftTest();


    int nr_blocks = ceil((double)matrixDim / (double)this->blockDim);
    this->blockDim3 = new dim3(this->blockDim, 1);;
    this->gridDim3 = new dim3(nr_blocks, nr_blocks);;

    cudaStatus = prepareGPU();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU init failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(cuArray, &channelDesc, matrixDim * 2, 1);

    cudaStatus = this->initTexture();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "GPU texture init failed: %s\n", cudaGetErrorString(cudaStatus));
        return;
    }

}

void spift::initResult() {
    this->result = new cuFloatComplex[this->matrixDim * this->matrixDim];
    //allocate memory for result
    for (int x = 0; x < this->matrixDim * this->matrixDim; ++x) {
        cuFloatComplex next;
        next.x = 0;
        next.y = 0;
        this->result[x] = next;
    }
}

void spift::initCoalescence()
{
    this->shiftIndexMutex = new std::mutex*[matrixDim * 2];
    this->done = new int(0);
    this->coalescenceSet = new int[matrixDim * 2];
    this->coalescence = new float* [matrixDim * 2];

    for (int i = 0; i < matrixDim * 2; ++i) {
        this->coalescenceSet[i] = 0;
        this->shiftIndexMutex[i] = new std::mutex();
        this->coalescence[i] = new float[matrixDim * 2];
    }
}

void spift::initShiftTest() {
    //allocate memory for shift vectors. testing only
    std::uniform_real_distribution<> dist(-1, 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    this->shift = new float* [200];
    for (int i = 0; i < 200; ++i) {
        float* bla = new float[2 * matrixDim];
        for (int x = 0; x < matrixDim * 2; ++x) {
            //bla[x] = x;
            bla[x] = dist(gen);
        }
        this->shift[i] = bla;
    }
}

cudaError_t spift::prepareGPU() {
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(this->GPUIndex);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }
    // Allocate GPU buffers for matrix
    cudaStatus = cudaSetDevice(this->GPUIndex);
    cudaStatus = cudaMalloc((void**)&(this->dev_matrix), this->matrixDim * this->matrixDim * sizeof(cuFloatComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    // Copy input Matrix from host memory to GPU buffers.
    cudaStatus = cudaSetDevice(this->GPUIndex);
    cudaStatus = cudaMemcpy((this->dev_matrix), (this->result), this->matrixDim * this->matrixDim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        return cudaStatus;
    }
    return cudaSuccess;
}

auto spift::displayTime() {
    std::cout << "terminated " << iterations << " iterations,  " << matrixDim << " dim, time: " << duration << "ms waitTime: " << this->wait << " micro seconds" << std::endl;
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

void spift::generateShifts() {
    int updates = 0;
    std::uniform_real_distribution<> dist(0, 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < iterations; ++i) {
        int posiCoal = (int)(dist(gen) * this->matrixDim * 2);
        int posiShift = (int)(dist(gen) * 200);
        this->shiftIndexMutex[posiCoal]->lock();
        if (this->coalescenceSet[posiCoal]) {
            ++updates;
            //std::cout << "updating: " << posiCoal << std::endl;
            for (int j = 0; j < this->matrixDim * 2; ++j) {
                this->coalescence[posiCoal][j] += this->shift[posiShift][j];
            }
        }
        else {
            //std::cout << "first:    " << posiCoal;
            for (int j = 0; j < this->matrixDim * 2; ++j) {
                this->coalescence[posiCoal][j] = this->shift[posiShift][j];
            }
            this->coalescenceSet[posiCoal] = 1;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(this->wait));
        //std::cout << "\t-unlocking" << std::endl;

        this->shiftIndexMutex[posiCoal]->unlock();

    }
    *(this->done) = 1;
}

cudaError_t spift::iterate() {
    /*
    Locks needed for:
    - this->coalescenceSet
    - this->coalescence
    */

    cudaError_t cudaStatus;
    auto t1 = std::chrono::high_resolution_clock::now();
    while(!*done) {
        //std::cout << "done: " << *done << std::endl;
        for (int shiftPos = 0; shiftPos < this->matrixDim * 2; ++shiftPos) {
            if (this->coalescenceSet[shiftPos]) {
                if (this->shiftIndexMutex[shiftPos]->try_lock()) {
                    this->coalescenceSet[shiftPos] = 0;
                    cudaStatus = iteration(shiftPos);
                    // Check for any errors in iteration
                    if (cudaStatus != cudaSuccess) {
                        fprintf(stderr, "iteration %d failed: %s\n", shiftPos, cudaGetErrorString(cudaStatus));
                        return cudaStatus;
                    }
                }
            }
        }
    }
    for (int shiftPos = 0; shiftPos < this->matrixDim * 2; ++shiftPos) {
        if (this->coalescenceSet[shiftPos]) {
            if (this->shiftIndexMutex[shiftPos]->try_lock()) {
                this->coalescenceSet[shiftPos] = 0;
                cudaStatus = iteration(shiftPos);
                // Check for any errors in iteration
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "iteration %d failed: %s\n", shiftPos, cudaGetErrorString(cudaStatus));
                    return cudaStatus;
                }
            }
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

void spift::getShiftVector(int i) {
    cudaSetDevice(this->GPUIndex);
    auto cudaStatus = cudaMemcpyToArray(*cuArray, 0, 0, this->coalescence[i], matrixDim * 2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "updateWithRowShift launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }

}

cudaError spift::iteration(int shift) {
    getShiftVector(shift);
    cudaSetDevice(this->GPUIndex);
    if (shift < this->matrixDim / 2) {
        updateWithRowShift << <*gridDim3, *blockDim3 >> > (dev_matrix, *texObj, this->matrixDim, shift);
    }
    else {
        updateWithColumnShift << <*gridDim3, *blockDim3 >> > (dev_matrix, *texObj, this->matrixDim, shift - this->matrixDim);
    }

    // Check for any errors launching the kernel
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "updateWithRowShift launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaSetDevice(this->GPUIndex);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s after launching Kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    this->shiftIndexMutex[shift]->unlock();
    return cudaSuccess;
}


cudaError_t spift::initTexture() {


    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = *(this->cuArray);
    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaSetDevice(this->GPUIndex);
    cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL);
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "textureInit failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    return cudaSuccess;
}

spift::~spift()
{
    free(this->shiftIndexMutex);
    free(this->result);
    for (int i = 0; i < 200; ++i) {
        free(this->shift[i]);
    }
   
    free(this->shift);
    cudaSetDevice(this->GPUIndex);
    cudaFree(this->dev_matrix);
    cudaFreeArray(*(this->cuArray));
    cudaError cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
    }
    for (int i = 0; i < matrixDim * 2; ++i) {
        free(this->coalescence[i]);
    }
    free(this->coalescenceSet);
    free(this->done);
}



void parallel(const int GPU_index, const int dim, std::ofstream *times, const int wait, std::mutex* writeMutex) {
    spift* tester = new spift(dim, 16, 100000, GPU_index, wait);
    std::cout << "inited" << std::endl;
    //tester->generateShifts();
    //tester->iterate();
    std::thread shifts(&spift::generateShifts, tester);
    std::thread iter(&spift::iterate, tester);
    shifts.join();
    iter.join();
    writeMutex->lock();
    *times << dim << "\t" << wait << "\t" << tester->displayTime() << "\t" << GPU_index << std::endl;
    //tester->printResult();
    writeMutex->unlock();
    delete tester;

}



int main()
{
    /*
    std::ofstream times;
    times.open("timesGPU.txt");
    parallel(1, 4096, &times, 5);
    */
    auto writeMutex = new std::mutex();
    std::ofstream times;
    times.open("timesGPU.txt");
    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 20; i++) {
            std::thread t0(parallel, 0, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t1(parallel, 1, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t2(parallel, 2, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t3(parallel, 3, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t4(parallel, 4, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t5(parallel, 5, pow(2, j + 9), &times, 5 * i, writeMutex);


            t0.join();
            t1.join();
            t2.join();
            t3.join();
            t4.join();
            t5.join();

            std::thread t00(parallel, 0, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t10(parallel, 1, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t20(parallel, 2, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t30(parallel, 3, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t40(parallel, 4, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t50(parallel, 5, pow(2, j + 9), &times, 5 * i, writeMutex);


            t00.join();
            t10.join();
            t20.join();
            t30.join();
            t40.join();
            t50.join();

            std::thread t000(parallel, 0, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t100(parallel, 1, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t200(parallel, 2, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t300(parallel, 3, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t400(parallel, 4, pow(2, j + 9), &times, 5 * i, writeMutex);
            std::thread t500(parallel, 5, pow(2, j + 9), &times, 5 * i, writeMutex);


            t000.join();
            t100.join();
            t200.join();
            t300.join();
            t400.join();
            t500.join();
        }
    }
    
    
    
    
    return 0;
}
