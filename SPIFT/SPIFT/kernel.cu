#include "cuda_runtime.h"#include "device_launch_parameters.h"#include "cuComplex.h"#include <stdio.h>#include <algorithm>#include <numeric>#include <iostream>#include <iomanip>#include <chrono>#include <fstream>#define _USE_MATH_DEFINES#include <math.h>#include <mutex>#include <thread>#include <complex>struct dataPoint{    int u;    int v;    std::complex<float> vis;};std::istream& operator>>(std::ifstream& input, struct dataPoint* data) {    char x;    input >> data->u;    input >> x;    if (x != ',') { input.setstate(std::ios_base::failbit); }    input >> data->v;    input >> x;    if (x != ',') { input.setstate(std::ios_base::failbit); }    input >> data->vis;    return input;}class spift{public:    spift(const int matrixDim, const int blockDim, const int nrGPUs, const int concurrency);    ~spift();    spift(const spift&) = delete;    void iterate(int pos, int start, int end);    void iteration(int shift, int pos);    void updateWithRowShift(cuFloatComplex* dev_matrix, int shift, int pos);    void updateWithColumnShift(cuFloatComplex* dev_matrix, int shift, int pos);    void splitIteration();    void divedeN2(cuFloatComplex* matrix);    bool shiftType(int u, int v);    int shiftIndex(int u, int v, bool isRowShift);    float* computeShift(int u, int v, std::complex<float> vis, bool isRowShift);    void initTwiddle();    void writeToFile(std::ofstream* file);    void printResult();    bool testResult(std::string original);
    void launchRead();    void readInput();    void aggregateResult();    void initCoalescence();    private:    //the dimensions of the grid and blocks    const int matrixDim;    const int blockDim;    dim3* blockDim3;    dim3* gridDim3;    //the Matrix where the result is loaded into in the end    cuFloatComplex* result;    //the device-matrix, where the current state is saved    cuFloatComplex** dev_matrix;    int readThreads;    //the aggregation of shifts, first half are rowShifts    float** coalescence;    //the index, where it is saved, wheter data is aggregated for this shift    int* coalescenceSet;    std::mutex** shiftIndexMutex;    //boolean wheter execution is done    int* done;    const int nrGPUS;    //the precomputed twiddleFactors    std::complex<float>* twiddleFactors;    std::ifstream* inputStream;    //testing    int count;    //measuring the execution time    unsigned long long durationTotal=0;    unsigned long long durationShiftProcessing=0;    unsigned long long durationShiftProcessingGenerating = 0;    unsigned long long durationShiftProcessingType = 0;    unsigned long long durationShiftProcessingShift=0;    unsigned long long durationShiftProcessingVector=0;    unsigned long long durationShiftAggregating=0;    unsigned long long durationRead=0;    unsigned long long durationFinal=0;    unsigned long long durationWhileRead=0;    unsigned long long durationUpload[6] = {0, 0, 0, 0, 0, 0};    unsigned long long durationFinalPrep=0;    unsigned long long durationRow[6] = { 0, 0, 0, 0, 0, 0 };    unsigned long long durationColumn[6] = { 0, 0, 0, 0, 0, 0 };    long nrUpdatesFinal = 0;    long nrRowUpdates = 0;    std::mutex* readMutex;        int total;};spift::spift(const int matrixDim, const int blockDim, const int nrGPUs, const int concurrency) : matrixDim(matrixDim), blockDim(blockDim), nrGPUS(nrGPUs), readThreads(concurrency){    this->initCoalescence();    this->initTwiddle();    struct dataPoint next;    int nr_blocks = ceil((double)matrixDim / (double)this->blockDim);    this->blockDim3 = new dim3(this->blockDim, 1);    this->gridDim3 = new dim3(nr_blocks, nr_blocks);        this->dev_matrix = new cuFloatComplex * [nrGPUs];       for (int i = 0; i < nrGPUs; ++i) {        this->dev_matrix[i] = new cuFloatComplex[this->matrixDim * this->matrixDim];        for (int y = 0; y < this->matrixDim; y++) {            for (int x = 0; x < this->matrixDim; x++) {                this->dev_matrix[i][x + y * this->matrixDim].x = 0;                this->dev_matrix[i][x + y * this->matrixDim].y = 0;            }        }    }    //testing    this->total = this->matrixDim * this->matrixDim;    this->readMutex = new std::mutex();    std::cout << "intited" << std::endl;    }void spift::initCoalescence(){    this->shiftIndexMutex = new std::mutex*[matrixDim * 2];    this->done = new int(0);    this->coalescenceSet = new int[matrixDim * 2];    this->coalescence = new float* [matrixDim * 2];    for (int i = 0; i < matrixDim * 2; ++i) {        this->coalescenceSet[i] = 0;        this->shiftIndexMutex[i] = new std::mutex();        this->coalescence[i] = new float[matrixDim * 2];        for (int j = 0; j < this->matrixDim * 2; j++) {            this->coalescence[i][j] = 0;        }    }}void spift::writeToFile(std::ofstream* file) {    long long durationRowUpdatelocal = 0;    long long durationColumnUpdatelocal = 0;    long long durationUploadLocal = 0;    for (int i = 0; i < nrGPUS; ++i) {        durationRowUpdatelocal += this->durationRow[i];        durationColumnUpdatelocal += this->durationColumn[i];        durationUploadLocal += this->durationUpload[i];    }    *file << this->readThreads << "\t" << this->matrixDim << "\t" << this->nrGPUS << "\t" << this->durationTotal << "\t" << this->count << "\t" << this->durationWhileRead << "\t" << this->durationFinal << "\t" << this->durationFinalPrep << "\t"<< durationRowUpdatelocal << "\t" << durationColumnUpdatelocal << "\t" << this->nrRowUpdates << "\t" << durationUploadLocal << "\t" << this->durationRead << "\t" << this->durationShiftProcessing << "\t" << this->durationShiftAggregating << "\t" << this->durationShiftProcessingType << "\t" << this->durationShiftProcessingShift << "\t" << this->durationShiftProcessingVector << "\t" << this->durationShiftProcessingGenerating << "\t" << this->nrUpdatesFinal << std::endl;}void spift::printResult() {    for (int i = 0; i < matrixDim; ++i) {        for (int j = 0; j < matrixDim; ++j) {            std::cout << "(" << std::setfill(' ') << std::setw(4) << std::roundf(this->result[this->matrixDim * i + j].x * 100) / 100 << ", " << std::setfill(' ') << std::setw(4) << std::roundf(this->result[this->matrixDim * i + j].y * 100) / 100 << "),\t";        }        std::cout << std::endl;    }}bool spift::testResult(std::string original){    std::ifstream originalFile(original);    double pos;    int count = 0;    for (int i = 0; i < this->matrixDim; ++i) {        for (int j = 0; j < this->matrixDim; ++j) {            originalFile >> pos;            if (abs(this->result[this->matrixDim * i + j].x - pos) > 0.0001 || abs(this->result[this->matrixDim * i + j].y) > 0.0001) {                count++;            }                    }    }    std::cout << this->matrixDim * this->matrixDim << ", " << count << std::endl;    return true;}void spift::launchRead() {    std::thread** threads = new std::thread * [this->readThreads];    auto t1 = std::chrono::high_resolution_clock::now();    for (int i = 0; i < readThreads; ++i) {        threads[i] = new std::thread(&spift::readInput, this);    }    for (int i = 0; i < readThreads; ++i) {        threads[i]->join();    }    auto t2 = std::chrono::high_resolution_clock::now();    this->durationRead = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();}void spift::readInput() {    struct dataPoint next;    next.vis = std::complex<float>(std::rand(), std::rand());    unsigned long long localdurationShiftProcessing = 0;    unsigned long long localdurationShiftAggregating = 0;    unsigned long long localdurationShiftProcessingGenerating = 0;    unsigned long long localdurationShiftProcessingType = 0;    unsigned long long localdurationShiftProcessingShift = 0;    unsigned long long localdurationShiftProcessingVector = 0;    int counter = 0;    auto t5 = std::chrono::high_resolution_clock::now();    while(this->total > 0) {        auto t1 = std::chrono::high_resolution_clock::now();        this->total--;        next.u = std::rand() % this->matrixDim;        next.v = std::rand() % this->matrixDim;        int posiCoal;        auto t11 = std::chrono::high_resolution_clock::now();        bool isRowShift = this->shiftType(next.u, next.v);        auto t12 = std::chrono::high_resolution_clock::now();        int shiftIdx = this->shiftIndex(next.u, next.v, isRowShift);        auto t13 = std::chrono::high_resolution_clock::now();        float* vector = this->computeShift(next.u, next.v, next.vis, isRowShift);                if (isRowShift) { posiCoal = shiftIdx; }        else { posiCoal = shiftIdx + this->matrixDim; }        auto t2 = std::chrono::high_resolution_clock::now();        this->shiftIndexMutex[posiCoal]->lock();        if (this->coalescenceSet[posiCoal]) {            for (int j = 0; j < this->matrixDim * 2; ++j) {                this->coalescence[posiCoal][j] += vector[j];            }        }        else {            for (int j = 0; j < this->matrixDim * 2; ++j) {                this->coalescence[posiCoal][j] = vector[j];            }            this->coalescenceSet[posiCoal] = 1;        }        this->shiftIndexMutex[posiCoal]->unlock();        auto t3 = std::chrono::high_resolution_clock::now();        free(vector);//        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << ", " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << std::endl;        localdurationShiftProcessing += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();        localdurationShiftAggregating += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();        localdurationShiftProcessingGenerating += std::chrono::duration_cast<std::chrono::microseconds>(t11 - t1).count();        localdurationShiftProcessingType += std::chrono::duration_cast<std::chrono::microseconds>(t12 - t11).count();        localdurationShiftProcessingShift += std::chrono::duration_cast<std::chrono::microseconds>(t13- t12).count();        localdurationShiftProcessingVector += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t13).count();        counter++;    }    auto t6 = std::chrono::high_resolution_clock::now();    //std::cout << "counter: " << localdurationShiftAggregating << ", " << localdurationShiftProcessing << std::endl;    *(this->done) = 1;    this->readMutex->lock();    this->durationShiftAggregating += localdurationShiftAggregating;    this->durationShiftProcessing += localdurationShiftProcessing;    this->durationShiftProcessingType += localdurationShiftProcessingType;    this->durationShiftProcessingGenerating += localdurationShiftProcessingGenerating;    this->durationShiftProcessingShift += localdurationShiftProcessingShift;    this->durationShiftProcessingVector += localdurationShiftProcessingVector;    this->readMutex->unlock();}void spift::aggregateResult() {    for (int i = 1; i < this->nrGPUS; ++i) {        for (int y = 0; y < this->matrixDim; ++y) {            for (int x = 0; x < this->matrixDim;++x) {                this->dev_matrix[0][x + y * this->matrixDim].x += this->dev_matrix[i][x + y * this->matrixDim].x;                this->dev_matrix[0][x + y * this->matrixDim].y += this->dev_matrix[i][x + y * this->matrixDim].y;            }        }    }}void spift::splitIteration() {    this->count = 0;    std::thread** threads = new std::thread*[this->nrGPUS];    auto t1 = std::chrono::high_resolution_clock::now();    for (int i = 0; i < this->nrGPUS; ++i) {        threads[i] = new std::thread(&spift::iterate, this, i, round((float)this->matrixDim * 2 / this->nrGPUS * i), round((float)this->matrixDim * 2 / this->nrGPUS * (i+1)));    }    std::cout << "done" << std::endl;    for (int i = 0; i < this->nrGPUS; ++i) {        threads[i]->join();    }    std::cout << "updated" << std::endl;    auto t15 = std::chrono::high_resolution_clock::now();    this->aggregateResult();    std::cout << "aggregated" << std::endl;    this->divedeN2(this->dev_matrix[0]);    std::cout << "divided" << std::endl;    auto t2 = std::chrono::high_resolution_clock::now();    this->durationTotal = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();    this->durationFinalPrep = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t15).count();    std::cout << "done " << std::endl;    return;}void spift::iterate(int pos, int start, int end) {    std::cout << start << "-->" << end << std::endl;    auto t1 = std::chrono::high_resolution_clock::now();    while(!*done) {        for (int shiftPos = start; shiftPos < end; ++shiftPos) {            if (this->coalescenceSet[shiftPos]) {                //std::cout << shiftPos << std::endl;                if (this->shiftIndexMutex[shiftPos]->try_lock()) {                    this->coalescenceSet[shiftPos] = 0;                    iteration(shiftPos, pos);                    this->shiftIndexMutex[shiftPos]->unlock();                    this->count++;                    // Check for any errors in iteration                                    }            }        }    }    auto t2 = std::chrono::high_resolution_clock::now();    int counter2 = 0;    for (int shiftPos = start; shiftPos < end; ++shiftPos) {        std::cout << shiftPos << std::endl;        if (this->coalescenceSet[shiftPos]) {            if (this->shiftIndexMutex[shiftPos]->try_lock()) {                counter2++;                //std::cout << shiftPos << std::endl;                this->coalescenceSet[shiftPos] = 0;                iteration(shiftPos, pos);                this->count++;            }         }    }    this->nrUpdatesFinal += counter2;    auto t3 = std::chrono::high_resolution_clock::now();    this->durationWhileRead += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();    this->durationFinal += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();       return;}void spift::iteration(int shift, int pos) {    //std::cout << "iteration: " << pos << ", shift: " << shift << std::endl;    auto t1 = std::chrono::high_resolution_clock::now();    if (shift < this->matrixDim) {        this->updateWithRowShift (dev_matrix[pos], shift, pos);        auto t2 = std::chrono::high_resolution_clock::now();        this->durationRow[pos] += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();        this->nrRowUpdates++;    }    else {        this->updateWithColumnShift(dev_matrix[pos], shift - this->matrixDim, pos);        // cudaDeviceSynchronize waits for the kernel to finish, and returns               auto t2 = std::chrono::high_resolution_clock::now();        this->durationColumn[pos] += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();    }    return;}

void spift::updateWithRowShift(cuFloatComplex* dev_matrix, int shift, int pos)
{
    for (int y = 0; y < this->matrixDim; ++y) {
        for (int x = 0; x < this->matrixDim; ++x) {
            this->dev_matrix[pos][x + this->matrixDim * y].x += this->coalescence[shift][2 * (x + y * shift) % this->matrixDim];
            this->dev_matrix[pos][x + this->matrixDim * y].y += this->coalescence[shift][2 * (x + y * shift) % this->matrixDim + 1];
        }
    }
}

void spift::updateWithColumnShift(cuFloatComplex* dev_matrix, int shift, int pos)
{
    for (int y = 0; y < this->matrixDim; ++y) {
        for (int x = 0; x < this->matrixDim; ++x) {
            this->dev_matrix[pos][x + this->matrixDim * y].x += this->coalescence[shift][2 * (x + y * shift) % this->matrixDim];
            this->dev_matrix[pos][x + this->matrixDim * y].y += this->coalescence[shift][2 * (x + y * shift) % this->matrixDim + 1];
        }
    }
}void spift::divedeN2(cuFloatComplex* dev_matrix) {    for (int y = 0; y < this->matrixDim; ++y) {
        for (int x = 0; x < this->matrixDim; ++x) {
            this->dev_matrix[0][x + this->matrixDim * y].x /= this->matrixDim;
            this->dev_matrix[0][x + this->matrixDim * y].y /= this->matrixDim;
        }
    }}bool spift::shiftType(int u, int v) {       //true: RowShift,   false: ColumnShift    return !(v == 0 || ((u % 2) && !(v % 2)) || (std::__gcd(u,this->matrixDim) < std::__gcd(v, this->matrixDim)) && v % 2 == 0);}int spift::shiftIndex(int u, int v, bool isRowShift) {    int uk;    int vk;    if (u == 0 || v == 0) {        return 0;    }    if (isRowShift) {        uk = u;        vk = v;    }    else    {        uk = v;        vk = u;    }    for (int j = 0; j <= this->matrixDim; ++j) {        if (uk == (j * vk) % this->matrixDim) {            return j;        }    }    std::cout << u << ", " << v << ", " << isRowShift << std::endl;    throw 15;}float* spift::computeShift(int u, int v, std::complex<float> vis, bool isRowShift) {    int x;    float* shift = new float[this->matrixDim * 2];    if (isRowShift) { x = v; }    else { x = u; }    for (int j = 0; j < matrixDim; ++j) {        std::complex<float> next = vis * this->twiddleFactors[(j * x) % this->matrixDim];        shift[2 * j] = next.real();        shift[2 * j + 1] = next.imag();    }    return shift;}void spift::initTwiddle() {    this->twiddleFactors = new std::complex<float>[this->matrixDim];    for(int k = 0; k < this->matrixDim; ++k){        std::complex<float> next = std::exp(std::complex<float>(0, k * 2 * M_PI / this->matrixDim));        this->twiddleFactors[k] = next;    }}spift::~spift(){    std::cout << "destructor" << std::endl;    free(readMutex);    for (int i = 0; i < this->nrGPUS; ++i) {        std::cout << i << std::endl;        free(dev_matrix[i]);    }    free(this->dev_matrix);    std::cout << "middle" << std::endl;    for (int i = 0; i < matrixDim * 2; ++i) {        free(this->coalescence[i]);        std::cout << this->shiftIndexMutex[i] << std::endl;        free(this->shiftIndexMutex[i]);    }    free(this->shiftIndexMutex);    std::cout << "mutexFreed" << std::endl;    free(this->coalescenceSet);    free(this->done);    free(twiddleFactors);    std::cout << "by" << std::endl;}void parallel(const int nrGPUs, const int dim, std::ofstream *times, const int concurrency) {    spift* tester = new spift(dim, std::min(16, dim), nrGPUs, concurrency);    tester->launchRead();    tester->splitIteration();    //std::cout << concurrency << "\t" << dim << "\t" << nrGPUs << std::endl;    /*    std::thread shifts(&spift::launchRead, tester);    std::thread iter(&spift::splitIteration, tester);    shifts.join();    iter.join();*/    //tester->writeToFile(times);    if (tester->testResult("originalData1024.txt")) {        std::cout << "success" << std::endl;    }    else    {        std::cout << "failed" << std::endl;    }        delete tester;}int main(){    std::ofstream times;    times.open("timesGPU.txt");    parallel(35, std::pow(2, 10), &times, 40);    /*    for (int i = 0; i < 3; i++) {        parallel(6, std::pow(2, 12), &times, 20);    }    for (int i = 0; i < 1; i++) {        parallel(1, std::pow(2, 10), &times, 1);    }    for (int i = 0; i < 3; i++) {        parallel(1, std::pow(2, 12), &times, 20);    }    for (int i = 0; i < 3; i++) {        parallel(6, std::pow(2, 12), &times, 40);    }    */    /*        for (int concurrency = 1; concurrency < 3; concurrency += 1) {        for (int j = 10; j < 13; ++j) {            for (int i = 1; i <= 6; i++) {                std::cout << "Block A: concurrency: " << concurrency << ", dim: " << std::pow(2, j) << ", nrGPUS: " << i << std::endl;                parallel(i, std::pow(2,j), &times, concurrency);            }        }    }        for (int concurrency = 3; concurrency < 30; concurrency += 2) {        for (int j = 10; j < 13; ++j) {            for (int i = 1; i <= 6; i++) {                std::cout << "Block B: concurrency: " << concurrency << ", dim: " << std::pow(2, j) << ", nrGPUS: " << i << std::endl;                parallel(i, std::pow(2, j), &times, concurrency);            }        }    }    for (int concurrency = 30; concurrency < 80; concurrency += 5) {        for (int j = 10; j < 14; ++j) {            for (int i = 1; i <= 6; i++) {                std::cout << "Block C: concurrency: " << concurrency << ", dim: " << std::pow(2, j) << ", nrGPUS: " << i << std::endl;                parallel(i, std::pow(2, j), &times, concurrency);            }        }    }        //std::cout << "done3" << std::endl;   */    times.close();    return 0;}