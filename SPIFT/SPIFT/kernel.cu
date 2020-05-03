#include "cuda_runtime.h"#include "device_launch_parameters.h"#include "cuComplex.h"#include <stdio.h>#include <algorithm>#include <numeric>#include <iostream>#include <iomanip>#include <chrono>#include <fstream>#define _USE_MATH_DEFINES#include <math.h>#include <mutex>#include <thread>#include <complex>struct dataPoint{    int u;    int v;    std::complex<float> vis;};std::istream& operator>>(std::ifstream& input, struct dataPoint* data) {    char x;    input >> data->u;    input >> x;    if (x != ',') { input.setstate(std::ios_base::failbit); }    input >> data->v;    input >> x;    if (x != ',') { input.setstate(std::ios_base::failbit); }    input >> data->vis;    return input;}constexpr int matrixStartIndexRow(const int matrix_dim, const int block_dim, int block_Idx, int block_Idy, int thread_idx) {    return block_Idx * block_dim * matrix_dim + thread_idx * matrix_dim + block_Idy * block_dim;}constexpr int matrixStartIndexColumn(const int matrix_dim, const int block_dim, int block_Idx, int block_Idy, int thread_idx) {    return block_Idx * block_dim * matrix_dim + block_Idy * block_dim + thread_idx;}constexpr int shiftedVectorIndex(const int matrix_dim, const int shift, const int index) {    return ((shift + index) % matrix_dim) * 2;}__global__ void updateWithRowShift(cuFloatComplex* dev_matrix, cudaTextureObject_t dev_vector, const int matrix_dim, const int shift){    if (blockIdx.x * blockDim.x + threadIdx.x < matrix_dim) {        unsigned int start_index = matrixStartIndexColumn(matrix_dim, blockDim.x, blockIdx.x, blockIdx.y, threadIdx.x);        int end = std::min(matrix_dim - blockIdx.x * blockDim.x, blockDim.x);        for (int i = 0; i < end; i++) {            int vectorPos = (shift * (blockIdx.x * blockDim.x + i) + (threadIdx.x + blockIdx.y * blockDim.x)) % matrix_dim * 2;            dev_matrix[start_index + i * matrix_dim].x += tex2D<float>(dev_vector, vectorPos, 0);            dev_matrix[start_index + i * matrix_dim].y += tex2D<float>(dev_vector, vectorPos + 1, 0);        }    }}__global__ void updateWithColumnShift(cuFloatComplex* dev_matrix, cudaTextureObject_t dev_vector, const int matrix_dim, const int shift){    if (blockIdx.x * blockDim.x + threadIdx.x < matrix_dim) {        unsigned int start_index = matrixStartIndexColumn(matrix_dim, blockDim.x, blockIdx.x, blockIdx.y, threadIdx.x);        int end = std::min(matrix_dim - blockIdx.x * blockDim.x, blockDim.x);        for (int i = 0; i < end; i++) {            int vectorPos = shiftedVectorIndex(matrix_dim, shift * threadIdx.x + blockDim.x * blockIdx.y * shift + blockIdx.x * blockDim.x, i);            dev_matrix[start_index + i * matrix_dim].x += tex2D<float>(dev_vector, vectorPos, 0);            dev_matrix[start_index + i * matrix_dim].y += tex2D<float>(dev_vector, vectorPos + 1, 0);        }    }}__global__ void divideN2(cuFloatComplex* dev_matrix, const int matrix_dim) {    if (blockIdx.x * blockDim.x + threadIdx.x < matrix_dim) {        unsigned int start_index = matrixStartIndexColumn(matrix_dim, blockDim.x, blockIdx.x, blockIdx.y, threadIdx.x);        int end = std::min(matrix_dim - blockIdx.x * blockDim.x, blockDim.x);        for (int i = 0; i < end; i++) {            dev_matrix[start_index + i * matrix_dim].x /= (matrix_dim * matrix_dim);            dev_matrix[start_index + i * matrix_dim].y /= (matrix_dim * matrix_dim);        }    }}__global__ void sumResults(cuFloatComplex* target, cuFloatComplex* source, const int matrix_dim) {    if (blockIdx.x * blockDim.x + threadIdx.x < matrix_dim) {        unsigned int start_index = matrixStartIndexColumn(matrix_dim, blockDim.x, blockIdx.x, blockIdx.y, threadIdx.x);        int end = std::min(matrix_dim - blockIdx.x * blockDim.x, blockDim.x);        for (int i = 0; i < end; i++) {            target[start_index + i * matrix_dim].x += source[start_index + i * matrix_dim].x;            target[start_index + i * matrix_dim].y += source[start_index + i * matrix_dim].y;        }    }}class spift{public:    spift(const int matrixDim, const int blockDim, const int GPU_index[], const int nrGPUs, std::string fileName);    ~spift();    spift(const spift&) = delete;    cudaError_t prepareGPU(int pos);    cudaError_t initTexture(int pos);    cudaError_t iterate(int pos, int start, int end);    cudaError_t iteration(int shift, int pos);    void splitIteration();    cudaError_t getResult(int pos);    bool shiftType(int u, int v);    int shiftIndex(int u, int v, bool isRowShift);    float* computeShift(int u, int v, std::complex<float> vis, bool isRowShift);    void initTwiddle();    auto displayTime();    void printResult();    bool testResult(std::string original);
    void launchRead();    void readInput(std::mutex* readMutex);    cudaError combine2Cards(int card1, int card2);    cudaError aggregateResult();    void initResult();    void initCoalescence();        void getShiftVector(int i, int pos);        private:    //the dimensions of the grid and blocks    const int matrixDim;    const int blockDim;    dim3* blockDim3;    dim3* gridDim3;    //the Matrix where the result is loaded into in the end    cuFloatComplex* result;    //the device-matrix, where the current state is saved    cuFloatComplex** dev_matrix;    //the texture object where the current shift is saved during kernel execution    cudaTextureObject_t** texObj = new cudaTextureObject_t*();    //the data in texobj    cudaArray** *cuArray;    //measuring the execution time    long long duration;    //the aggregation of shifts, first half are rowShifts    float** coalescence;    //the index, where it is saved, wheter data is aggregated for this shift    int* coalescenceSet;    std::mutex** shiftIndexMutex;    //boolean wheter execution is done    int* done;    //the Index of the GPU on which it is executed    int *GPUIndex;    const int nrGPUS;    //the precomputed twiddleFactors    std::complex<float>* twiddleFactors;    std::ifstream* inputStream;};spift::spift(const int matrixDim, const int blockDim, const int GPU_index[], const int nrGPUs, std::string fileName) : matrixDim(matrixDim), blockDim(blockDim), nrGPUS(nrGPUs){    this->initResult();    this->initCoalescence();    this->initTwiddle();    this->inputStream = new std::ifstream(fileName);    int nr_blocks = ceil((double)matrixDim / (double)this->blockDim);    this->blockDim3 = new dim3(this->blockDim, 1);    this->gridDim3 = new dim3(nr_blocks, nr_blocks);    cudaError_t cudaStatus;    this->dev_matrix = new cuFloatComplex * [nrGPUs];    this->cuArray = new cudaArray** [nrGPUs];    this->texObj = new cudaTextureObject_t*[nrGPUs];    this->GPUIndex = new int[nrGPUs];    for (int i = 0; i < nrGPUs; ++i) {        this->GPUIndex[i] = GPU_index[i];        this->cuArray[i] = new cudaArray * ();        this->texObj[i] = new cudaTextureObject_t();        cudaStatus = prepareGPU(i);        if (cudaStatus != cudaSuccess) {            fprintf(stderr, "GPU init failed: %s\n", cudaGetErrorString(cudaStatus));            return;        }        cudaStatus = this->initTexture(i);        if (cudaStatus != cudaSuccess) {            fprintf(stderr, "GPU texture init failed: %s\n", cudaGetErrorString(cudaStatus));            return;        }    }}void spift::initResult() {    this->result = new cuFloatComplex[this->matrixDim * this->matrixDim];    //allocate memory for result    for (int x = 0; x < this->matrixDim * this->matrixDim; ++x) {        cuFloatComplex next;        next.x = 0;        next.y = 0;        this->result[x] = next;    }}void spift::initCoalescence(){    this->shiftIndexMutex = new std::mutex*[matrixDim * 2];    this->done = new int(0);    this->coalescenceSet = new int[matrixDim * 2];    this->coalescence = new float* [matrixDim * 2];    for (int i = 0; i < matrixDim * 2; ++i) {        this->coalescenceSet[i] = 0;        this->shiftIndexMutex[i] = new std::mutex();        this->coalescence[i] = new float[matrixDim * 2];    }}cudaError_t spift::prepareGPU(int pos) {    cudaError_t cudaStatus;    // Choose which GPU to run on, change this on a multi-GPU system.    cudaStatus = cudaSetDevice(this->GPUIndex[pos]);    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");        return cudaStatus;    }    // Allocate GPU buffers for matrix    cudaStatus = cudaMalloc((void**)&(this->dev_matrix[pos]), this->matrixDim * this->matrixDim * sizeof(cuFloatComplex));    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "cudaMalloc failed!");        return cudaStatus;    }    // Copy input Matrix from host memory to GPU buffers.    cudaStatus = cudaSetDevice(this->GPUIndex[pos]);    cudaStatus = cudaMemcpy((this->dev_matrix[pos]), this->result, this->matrixDim * this->matrixDim * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "cudaMemcpy failed!");        return cudaStatus;    }    // Allocate CUDA array in device memory    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);    cudaMallocArray(cuArray[pos], &channelDesc, matrixDim * 2, 1);    return cudaSuccess;}auto spift::displayTime() {    std::cout << "terminated " << matrixDim << " dim, time: " << duration << std::endl;;    return duration;}void spift::printResult() {    for (int i = 0; i < matrixDim; ++i) {        for (int j = 0; j < matrixDim; ++j) {            std::cout << "(" << std::setfill(' ') << std::setw(4) << std::roundf(this->result[this->matrixDim * i + j].x * 100) / 100 << ", " << std::setfill(' ') << std::setw(4) << std::roundf(this->result[this->matrixDim * i + j].y * 100) / 100 << "),\t";        }        std::cout << std::endl;    }}bool spift::testResult(std::string original){    std::ifstream originalFile(original);    double pos;    int count = 0;    for (int i = 0; i < this->matrixDim; ++i) {        for (int j = 0; j < this->matrixDim; ++j) {            originalFile >> pos;            if (abs(this->result[this->matrixDim * i + j].x - pos) > 0.0001 || abs(this->result[this->matrixDim * i + j].y) > 0.0001) {                count++;            }                    }    }    std::cout << this->matrixDim * this->matrixDim << ", " << count << std::endl;    return true;}void spift::launchRead() {    std::mutex readMutex;    int concurrency = 200;    std::thread** threads = new std::thread * [concurrency];    for (int i = 0; i < concurrency; ++i) {        threads[i] = new std::thread(&spift::readInput, this, &readMutex);    }    for (int i = 0; i < 74; ++i) {        threads[i]->join();    }}void spift::readInput(std::mutex* readMutex) {    struct dataPoint next;    readMutex->lock();    while(*(this->inputStream)>>&next) {        readMutex->unlock();        int posiCoal;        bool isRowShift = this->shiftType(next.u, next.v);        int shiftIdx = this->shiftIndex(next.u, next.v, isRowShift);        float* vector = this->computeShift(next.u, next.v, next.vis, isRowShift);                if (isRowShift) { posiCoal = shiftIdx; }        else { posiCoal = shiftIdx + this->matrixDim; }                this->shiftIndexMutex[posiCoal]->lock();        if (this->coalescenceSet[posiCoal]) {            for (int j = 0; j < this->matrixDim * 2; ++j) {                this->coalescence[posiCoal][j] += vector[j];            }        }        else {            for (int j = 0; j < this->matrixDim * 2; ++j) {                this->coalescence[posiCoal][j] = vector[j];            }            this->coalescenceSet[posiCoal] = 1;        }        this->shiftIndexMutex[posiCoal]->unlock();        free(vector);        readMutex->lock();    }    readMutex->unlock();    *(this->done) = 1;}cudaError spift::combine2Cards(int sourceCard, int targetCard) {    cuFloatComplex* temp;    cudaSetDevice(this->GPUIndex[targetCard]);    auto cudaStatus = cudaMalloc((void**)&temp, this->matrixDim * this->matrixDim * sizeof(cuFloatComplex));    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "Malloc for combine2Cards failed: %s\n", cudaGetErrorString(cudaStatus));        return cudaStatus;    }    cudaStatus = cudaMemcpyPeer(temp, this->GPUIndex[targetCard], this->dev_matrix[sourceCard], this->GPUIndex[sourceCard], this->matrixDim * this->matrixDim * sizeof(cuFloatComplex));    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "Memcpy for combine2Cards failed: %s\n", cudaGetErrorString(cudaStatus));        return cudaStatus;    }    cudaSetDevice(this->GPUIndex[targetCard]);    sumResults << <*gridDim3, * blockDim3 >> > (this->dev_matrix[targetCard], temp, this->matrixDim);    cudaDeviceSynchronize();    cudaStatus = cudaGetLastError();    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "sumResults launch failed: %s\n", cudaGetErrorString(cudaStatus));        return cudaStatus;    }    cudaFree(temp);    return cudaSuccess;}cudaError spift::aggregateResult() {    cudaError cudaStatus;    for (int i = 1; i < this->nrGPUS; ++i) {        cudaStatus = this->combine2Cards(i - 1, i);        if (cudaStatus != cudaSuccess) {            fprintf(stderr, "Memcpy for combine2Cards failed: %s\n", cudaGetErrorString(cudaStatus));            return cudaStatus;        }    }    return cudaSuccess;}void spift::splitIteration() {    std::thread** threads = new std::thread*[this->nrGPUS];    for (int i = 0; i < this->nrGPUS; ++i) {        threads[i] = new std::thread(&spift::iterate, this, i, round((float)this->matrixDim * 2 / this->nrGPUS * i), round((float)this->matrixDim * 2 / this->nrGPUS * (i+1)));    }    for (int i = 0; i < this->nrGPUS; ++i) {        threads[i]->join();    }    auto t1 = std::chrono::high_resolution_clock::now();    auto cudaStatus = this->aggregateResult();    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "Aggregation failed\n", cudaGetErrorString(cudaStatus));        return;    }    cudaStatus = this->getResult(this->nrGPUS-1);    auto t2 = std::chrono::high_resolution_clock::now();    this->duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "getResult failed\n", cudaGetErrorString(cudaStatus));        return;    }}cudaError_t spift::iterate(int pos, int start, int end) {    cudaError_t cudaStatus = cudaSuccess;        while(!*done) {        for (int shiftPos = start; shiftPos < end; ++shiftPos) {            if (this->coalescenceSet[shiftPos]) {                if (this->shiftIndexMutex[shiftPos]->try_lock()) {                    this->coalescenceSet[shiftPos] = 0;                    cudaStatus = iteration(shiftPos, pos);                    // Check for any errors in iteration                    if (cudaStatus != cudaSuccess) {                        fprintf(stderr, "iteration %d failed: %s\n", shiftPos, cudaGetErrorString(cudaStatus));                        return cudaStatus;                    }                }            }        }    }    for (int shiftPos = start; shiftPos < end; ++shiftPos) {        if (this->coalescenceSet[shiftPos]) {            if (this->shiftIndexMutex[shiftPos]->try_lock()) {                this->coalescenceSet[shiftPos] = 0;                cudaStatus = iteration(shiftPos, pos);                // Check for any errors in iteration                if (cudaStatus != cudaSuccess) {                    fprintf(stderr, "iteration %d failed: %s\n", shiftPos, cudaGetErrorString(cudaStatus));                    return cudaStatus;                }            }        }    }        return cudaStatus;}cudaError_t spift::getResult(int pos) {    // Copy output vector from GPU buffer to host memory.    auto cudaStatus = cudaSetDevice(this->GPUIndex[pos]);    divideN2 << <*gridDim3, * blockDim3 >> > (dev_matrix[pos], this->matrixDim);    cudaStatus = cudaDeviceSynchronize();    cudaStatus = cudaGetLastError();    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "divideN2 launch failed: %s\n", cudaGetErrorString(cudaStatus));        return cudaStatus;    }    cudaStatus = cudaMemcpy(result, dev_matrix[pos], matrixDim * matrixDim * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "cudaMemcpy failed!");        return cudaStatus;    }    return cudaSuccess;}void spift::getShiftVector(int i, int pos) {    cudaSetDevice(this->GPUIndex[pos]);    auto cudaStatus = cudaMemcpyToArray(*cuArray[pos], 0, 0, this->coalescence[i], matrixDim * 2 * sizeof(float), cudaMemcpyHostToDevice);    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "updateWithRowShift launch failed: %s\n", cudaGetErrorString(cudaStatus));    }}cudaError spift::iteration(int shift, int pos) {    getShiftVector(shift, pos);    cudaSetDevice(this->GPUIndex[pos]);    if (shift < this->matrixDim) {        updateWithRowShift << <*gridDim3, *blockDim3 >> > (dev_matrix[pos], *texObj[pos], this->matrixDim, shift);    }    else {        updateWithColumnShift << <*gridDim3, *blockDim3 >> > (dev_matrix[pos], *texObj[pos], this->matrixDim, shift - this->matrixDim);    }    // Check for any errors launching the kernel    auto cudaStatus = cudaGetLastError();    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "updateWithRowShift launch failed: %s\n", cudaGetErrorString(cudaStatus));        return cudaStatus;    }    // cudaDeviceSynchronize waits for the kernel to finish, and returns    // any errors encountered during the launch.    cudaStatus = cudaSetDevice(this->GPUIndex[pos]);    cudaStatus = cudaDeviceSynchronize();           if (cudaStatus != cudaSuccess) {        fprintf(stderr, "cudaDeviceSynchronize returned error code %d: %s after launching Kernel!\n", cudaStatus, cudaGetErrorString(cudaStatus));        return cudaStatus;    }    this->shiftIndexMutex[shift]->unlock();    return cudaSuccess;}cudaError_t spift::initTexture(int pos) {    // Specify texture    struct cudaResourceDesc resDesc;    memset(&resDesc, 0, sizeof(resDesc));    resDesc.resType = cudaResourceTypeArray;    resDesc.res.array.array = *(this->cuArray[pos]);    // Specify texture object parameters    struct cudaTextureDesc texDesc;    memset(&texDesc, 0, sizeof(texDesc));    texDesc.addressMode[0] = cudaAddressModeClamp;    texDesc.addressMode[1] = cudaAddressModeClamp;    texDesc.filterMode = cudaFilterModePoint;    texDesc.readMode = cudaReadModeElementType;    texDesc.normalizedCoords = 0;    // Create texture object    cudaSetDevice(this->GPUIndex[pos]);    auto cudaStatus = cudaCreateTextureObject(this->texObj[pos], &resDesc, &texDesc, NULL);    if (cudaStatus != cudaSuccess) {        fprintf(stderr, "textureInit failed: %s\n", cudaGetErrorString(cudaStatus));    }    return cudaSuccess;}bool spift::shiftType(int u, int v) {       //true: RowShift,   false: ColumnShift    return !(v == 0 || ((u % 2) && !(v % 2)) || (std::__gcd(u,this->matrixDim) < std::__gcd(v, this->matrixDim)) && v % 2 == 0);}int spift::shiftIndex(int u, int v, bool isRowShift) {    int uk;    int vk;    if (u == 0 || v == 0) {        return 0;    }    if (isRowShift) {        uk = u;        vk = v;    }    else    {        uk = v;        vk = u;    }    for (int j = 0; j <= this->matrixDim; ++j) {        if (uk == (j * vk) % this->matrixDim) {            return j;        }    }    std::cout << u << ", " << v << ", " << isRowShift << std::endl;    throw 15;}float* spift::computeShift(int u, int v, std::complex<float> vis, bool isRowShift) {    int x;    float* shift = new float[this->matrixDim * 2];    if (isRowShift) { x = v; }    else { x = u; }    for (int j = 0; j < matrixDim; ++j) {        std::complex<float> next = vis * this->twiddleFactors[(j * x) % this->matrixDim];        shift[2 * j] = next.real();        shift[2 * j + 1] = next.imag();    }    return shift;}void spift::initTwiddle() {    this->twiddleFactors = new std::complex<float>[this->matrixDim];    for(int k = 0; k < this->matrixDim; ++k){        std::complex<float> next = std::exp(std::complex<float>(0, k * 2 * M_PI / this->matrixDim));        this->twiddleFactors[k] = next;    }}spift::~spift(){    free(this->shiftIndexMutex);    free(this->result);    for (int i = 0; i < this->nrGPUS; ++i) {        cudaSetDevice(this->GPUIndex[i]);        cudaFree(this->dev_matrix[i]);        cudaFreeArray(*(this->cuArray[i]));        cudaFree(this->texObj[i]);        cudaError cudaStatus = cudaDeviceReset();        if (cudaStatus != cudaSuccess) {            fprintf(stderr, "cudaDeviceReset failed!");        }    }    free(this->dev_matrix);    free(this->cuArray);        for (int i = 0; i < matrixDim * 2; ++i) {        free(this->coalescence[i]);    }    free(this->coalescenceSet);    free(this->done);    free(twiddleFactors);    this->inputStream->close();}void parallel(const int GPU_index, const int dim, std::ofstream *times, std::mutex* writeMutex, std::string fileName) {    int gpuIndex[] = { 0};    spift* tester = new spift(dim, std::min(16, dim), gpuIndex, 1,  fileName);    tester->launchRead();    tester->splitIteration();    //std::thread shifts(&spift::launchRead, tester);    //std::thread iter(&spift::splitIteration, tester);    //shifts.join();    //iter.join();    writeMutex->lock();    *times << dim << "\t" << "\t" << tester->displayTime() << "\t" << GPU_index << std::endl;    if (tester->testResult("originalData.txt")) {        std::cout << "success" << std::endl;    }    else    {        std::cout << "failed" << std::endl;    }    writeMutex->unlock();    delete tester;}int main(){    auto writeMutex = new std::mutex();    std::ofstream times;    times.open("timesGPU.txt");    parallel(1, 8, &times, writeMutex, "testData.txt");    //std::cout << "done3" << std::endl;       times.close();    return 0;}