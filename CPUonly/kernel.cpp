#include <stdio.h>#include <algorithm>#include <numeric>#include <iostream>#include <iomanip>#include <chrono>#include <fstream>#define _USE_MATH_DEFINES#include <math.h>#include <mutex>#include <thread>#include <complex>#include <vector>struct dataPoint{    int u;    int v;    std::complex<float> vis;};std::istream& operator>>(std::ifstream& input, struct dataPoint* data) {    char x;    input >> data->u;    input >> x;    if (x != ',') { input.setstate(std::ios_base::failbit); }    input >> data->v;    input >> x;    if (x != ',') { input.setstate(std::ios_base::failbit); }    input >> data->vis;    return input;}class spift{public:    spift(const int matrixDim, const int nrGPUs, const int concurrency, std::string fileName);    ~spift();    spift(const spift&) = delete;    void iterate(int pos, int start, int end);    void iteration(int shift, int pos);    void updateWithRowShift(int shift, int pos);    void updateWithColumnShift(int shift, int pos);    void splitIteration();    void divedeN2();    bool shiftType(int u, int v);    int shiftIndex(int u, int v, bool isRowShift);    float* computeShift(int u, int v, std::complex<float> vis, bool isRowShift);    void initTwiddle();    void writeToFile(std::ofstream* file);    void printResult(int pos);    void printRawResult();    bool testResult(std::string original);    void launchRead();    void readInput(int i);    void aggregateResult();    void initCoalescence();    void getDatapoint(struct dataPoint* next, int threadId, int readPos);private:    //the dimensions of the grid and blocks    const int matrixDim;    //the device-matrix, where the current state is saved    float** dev_matrix;    int readThreads;    //the aggregation of shifts, first half are rowShifts    float** coalescence;    //the index, where it is saved, wheter data is aggregated for this shift    int* coalescenceSet;    std::mutex** shiftIndexMutex;    //boolean wheter execution is done    int* done;    const int nrGPUS;    //the precomputed twiddleFactors    std::complex<float>* twiddleFactors;    //std::ifstream* inputStream;    //testing    int count;    //measuring the execution time    unsigned long long durationTotal=0;    unsigned long long durationShiftProcessing=0;    unsigned long long durationShiftProcessingGenerating = 0;    unsigned long long durationShiftProcessingType = 0;    unsigned long long durationShiftProcessingShift=0;    unsigned long long durationShiftProcessingVector=0;    unsigned long long durationShiftAggregating=0;    unsigned long long durationRead=0;    unsigned long long durationFinal=0;    unsigned long long durationWhileRead=0;    unsigned long long * durationUpload;    unsigned long long durationFinalPrep=0;    unsigned long long * durationRow;    unsigned long long * durationColumn;    long nrUpdatesFinal = 0;    long nrRowUpdates = 0;    std::mutex* readMutex = new std::mutex();    std::vector<std::vector<int>> inputData;    int total = 4194304;    int length_input;};spift::spift(const int matrixDim,  const int nrGPUs, const int concurrency, std::string fileName) : matrixDim(matrixDim), nrGPUS(nrGPUs), readThreads(concurrency){    //std::cout << "init" << std::endl;    this->initCoalescence();    this->initTwiddle();    //this->inputStream = new std::ifstream(fileName);    this->dev_matrix = new float * [nrGPUs];    this->durationUpload = new unsigned long long[concurrency];    this->durationRow = new unsigned long long[concurrency];    this->durationColumn = new unsigned long long[concurrency];    for (int i = 0; i < nrGPUs; ++i) {        this->dev_matrix[i] = new float[2*this->matrixDim * this->matrixDim];        for (int y = 0; y < this->matrixDim; y++) {            for (int x = 0; x < this->matrixDim*2; x++) {                this->dev_matrix[i][x + y * this->matrixDim] = 0;            }        }    }    this->length_input = (int)(std::ceil(2 * this->total / this->readThreads) * 1.5);    for (int j = 0; j < this->readThreads; j++) {        std::vector<int> a;        for (int i = 0; i < this->length_input; i++) {            a.push_back(std::rand() % this->matrixDim);        }        this->inputData.push_back(a);    }    std::cout << this->readThreads << std::endl;    std::cout << this->length_input << std::endl;    std::cout << "initDone" << std::endl;/*    this->coalescence[1][0]=1;    this->iteration(1,0);    this->printResult();*/    }void spift::initCoalescence(){    this->shiftIndexMutex = new std::mutex*[matrixDim * 2];    this->done = new int(0);    this->coalescenceSet = new int[matrixDim * 2];    this->coalescence = new float* [matrixDim * 2];    for (int i = 0; i < matrixDim * 2; ++i) {        this->shiftIndexMutex[i] = new std::mutex();        this->coalescenceSet[i] = 0;        this->coalescence[i] = new float[matrixDim * 2];        for (int j = 0; j < this->matrixDim * 2; j++) {            this->coalescence[i][j] = 0;        }    }}void spift::writeToFile(std::ofstream* file) {    long long durationRowUpdatelocal = 0;    long long durationColumnUpdatelocal = 0;    long long durationUploadLocal = 0;    for (int i = 0; i < nrGPUS; ++i) {        durationRowUpdatelocal += this->durationRow[i];        durationColumnUpdatelocal += this->durationColumn[i];        durationUploadLocal += this->durationUpload[i];    }    *file << this->readThreads << "\t" << this->matrixDim << "\t" << this->nrGPUS << "\t" << this->durationTotal << "\t" << this->count << "\t" << this->durationWhileRead << "\t" << this->durationFinal << "\t" << this->durationFinalPrep << "\t"<< durationRowUpdatelocal << "\t" << durationColumnUpdatelocal << "\t" << this->nrRowUpdates << "\t" << durationUploadLocal << "\t" << this->durationRead << "\t" << this->durationShiftProcessing << "\t" << this->durationShiftAggregating << "\t" << this->durationShiftProcessingType << "\t" << this->durationShiftProcessingShift << "\t" << this->durationShiftProcessingVector << "\t" << this->durationShiftProcessingGenerating << "\t" << this->nrUpdatesFinal << std::endl;}void spift::printResult(int pos) {    for (int i = 0; i < matrixDim; ++i) {        for (int j = 0; j < matrixDim; ++j) {            std::cout << "(" << std::setfill(' ') << std::setw(4) << std::roundf(this->dev_matrix[pos][2*(this->matrixDim * i + j)] * 100) / 100 << ", " << std::setfill(' ') << std::setw(4) << std::roundf(this->dev_matrix[pos][2 * (this->matrixDim * i + j)+1] * 100) / 100 << "),\t";        }        std::cout << std::endl;    }}void spift::printRawResult() {    for (int i = 0; i < matrixDim; ++i) {        for (int j = 0; j < matrixDim*2; ++j) {            std::cout << "(" << this->dev_matrix[0][2*(this->matrixDim * i + j)]  << ", " << this->dev_matrix[0][2 * (this->matrixDim * i + j)+1] << "),\t";        }        std::cout << std::endl;    }}bool spift::testResult(std::string original){    std::ifstream originalFile(original);    double pos;    int count = 0;    for (int i = 0; i < this->matrixDim; ++i) {        for (int j = 0; j < this->matrixDim; ++j) {            originalFile >> pos;            if (abs(this->dev_matrix[0][2 * (this->matrixDim * i + j)] - pos) > 0.0001 || abs(this->dev_matrix[0][2 * (this->matrixDim * i + j)+1]) > 0.0001) {                count++;            }                    }    }    std::cout << "total Points: " << this->matrixDim * this->matrixDim << ", Failed: " << count << std::endl;    return count == 0;}void spift::launchRead() {    std::thread** threads = new std::thread * [this->readThreads];    auto t1 = std::chrono::high_resolution_clock::now();    for (int i = 0; i < readThreads; ++i) {        threads[i] = new std::thread(&spift::readInput, this, i);    }    for (int i = 0; i < readThreads; ++i) {        threads[i]->join();    }    auto t2 = std::chrono::high_resolution_clock::now();    this->durationRead = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();}void inline spift::getDatapoint(struct dataPoint* next, int threadId, int readPos) {    //next->u = std::rand() % this->matrixDim;    //next->v = std::rand() % this->matrixDim;    /*this->readMutex->lock();    *(this->inputStream) >> next;    this->readMutex->unlock();    */    /*    std::cout << this->length_input << " done" << std::endl;    for(int i = 0; i < this->readThreads; i++){        for(int j = 0; j < this->length_input; j++){            if(this->input[i][j] > this->matrixDim){                std::cout << "failure "<< i << ", " << j << ", " <<this->input[i][j] << std::endl;                throw false;            }        }    }*/    next->u = this->inputData.at(threadId).at(readPos);    readPos++;    next->v = this->inputData.at(threadId).at(readPos);    readPos++;    if(next->u > this->matrixDim || next->v > this->matrixDim){        std::cout << threadId << ", " << readPos << std::endl;        std::cout << next->u << ", " << next->v << std::endl;    }    readPos %= this->length_input - 4;}void spift::readInput(int i) {    struct dataPoint next;    next.vis = std::complex<float>(std::rand(), std::rand());    int readPos = 0;    unsigned long long localdurationShiftProcessing = 0;    unsigned long long localdurationShiftAggregating = 0;    unsigned long long localdurationShiftProcessingGenerating = 0;    unsigned long long localdurationShiftProcessingType = 0;    unsigned long long localdurationShiftProcessingShift = 0;    unsigned long long localdurationShiftProcessingVector = 0;    int counter = 0;    auto t5 = std::chrono::high_resolution_clock::now();    while(this->total > 0) {        //std::cout << readPos <<"." ;        auto t1 = std::chrono::high_resolution_clock::now();        this->total--;        this->getDatapoint(&next,i,readPos);        readPos+=2;        int posiCoal;        auto t11 = std::chrono::high_resolution_clock::now();        bool isRowShift = this->shiftType(next.u, next.v);        auto t12 = std::chrono::high_resolution_clock::now();        int shiftIdx = this->shiftIndex(next.u, next.v, isRowShift);        auto t13 = std::chrono::high_resolution_clock::now();        float* vector = this->computeShift(next.u, next.v, next.vis, isRowShift);        if (isRowShift) { posiCoal = shiftIdx; }        else { posiCoal = shiftIdx + this->matrixDim; }        auto t2 = std::chrono::high_resolution_clock::now();        this->shiftIndexMutex[posiCoal]->lock();        if (this->coalescenceSet[posiCoal]) {            for (int j = 0; j < this->matrixDim * 2; ++j) {                this->coalescence[posiCoal][j] += vector[j];            }        }        else {            this->coalescence[posiCoal] = vector;            this->coalescenceSet[posiCoal] = 1;        }        this->shiftIndexMutex[posiCoal]->unlock();        auto t3 = std::chrono::high_resolution_clock::now();//        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << ", " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << std::endl;        localdurationShiftProcessing += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();        localdurationShiftAggregating += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();        localdurationShiftProcessingGenerating += std::chrono::duration_cast<std::chrono::microseconds>(t11 - t1).count();        localdurationShiftProcessingType += std::chrono::duration_cast<std::chrono::microseconds>(t12 - t11).count();        localdurationShiftProcessingShift += std::chrono::duration_cast<std::chrono::microseconds>(t13- t12).count();        localdurationShiftProcessingVector += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t13).count();        counter++;    }    auto t6 = std::chrono::high_resolution_clock::now();    //std::cout << "counter: " << localdurationShiftAggregating << ", " << localdurationShiftProcessing << std::endl;    *(this->done) = 1;    this->readMutex->lock();    this->durationShiftAggregating += localdurationShiftAggregating;    this->durationShiftProcessing += localdurationShiftProcessing;    this->durationShiftProcessingType += localdurationShiftProcessingType;    this->durationShiftProcessingGenerating += localdurationShiftProcessingGenerating;    this->durationShiftProcessingShift += localdurationShiftProcessingShift;    this->durationShiftProcessingVector += localdurationShiftProcessingVector;    this->readMutex->unlock();}void spift::aggregateResult() {    for (int i = 1; i < this->nrGPUS; ++i) {        for (int y = 0; y < this->matrixDim; ++y) {            for (int x = 0; x < this->matrixDim*2;++x) {                this->dev_matrix[0][x + y * this->matrixDim*2] += this->dev_matrix[i][x + y * this->matrixDim*2];            }        }    }}void spift::splitIteration() {    this->count = 0;    std::thread** threads = new std::thread*[this->nrGPUS];    auto t1 = std::chrono::high_resolution_clock::now();    for (int i = 0; i < this->nrGPUS; ++i) {        threads[i] = new std::thread(&spift::iterate, this, i, round((float)this->matrixDim * 2 / this->nrGPUS * i), round((float)this->matrixDim * 2 / this->nrGPUS * (i+1)));    }    for (int i = 0; i < this->nrGPUS; ++i) {        threads[i]->join();    }    auto t15 = std::chrono::high_resolution_clock::now();    this->aggregateResult();    this->divedeN2();    auto t2 = std::chrono::high_resolution_clock::now();    this->durationTotal = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();    this->durationFinalPrep = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t15).count();}void spift::iterate(int pos, int start, int end) {    auto t1 = std::chrono::high_resolution_clock::now();    while(!*(this->done)) {        for (int shiftPos = start; shiftPos < end; ++shiftPos) {            if (this->coalescenceSet[shiftPos]) {                //std::cout << shiftPos << std::endl;                if (this->shiftIndexMutex[shiftPos]->try_lock()) {                    this->coalescenceSet[shiftPos] = 0;                    iteration(shiftPos, pos);                    delete (coalescence[shiftPos]);                    this->shiftIndexMutex[shiftPos]->unlock();                    this->count++;                    // Check for any errors in iteration                                    }            }        }    }    auto t2 = std::chrono::high_resolution_clock::now();    int counter2 = 0;    for (int shiftPos = start; shiftPos < end; ++shiftPos) {        //std::cout << shiftPos << std::endl;        if (this->coalescenceSet[shiftPos]) {            if (this->shiftIndexMutex[shiftPos]->try_lock()) {                counter2++;                //std::cout << shiftPos << std::endl;                this->coalescenceSet[shiftPos] = 0;                iteration(shiftPos, pos);                delete (coalescence[shiftPos]);                this->shiftIndexMutex[shiftPos]->unlock();                this->count++;            }         }    }    this->nrUpdatesFinal += counter2;    auto t3 = std::chrono::high_resolution_clock::now();    this->durationWhileRead += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();    this->durationFinal += std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();}void spift::iteration(int shift, int pos) {    //std::cout << "iteration: " << pos << ", shift: " << shift << std::endl;    auto t1 = std::chrono::high_resolution_clock::now();    if (shift < this->matrixDim) {        this->updateWithRowShift (shift, pos);        auto t2 = std::chrono::high_resolution_clock::now();        this->durationRow[pos] += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();        this->nrRowUpdates++;    }    else {        this->updateWithColumnShift(shift, pos);        // cudaDeviceSynchronize waits for the kernel to finish, and returns               auto t2 = std::chrono::high_resolution_clock::now();        this->durationColumn[pos] += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();    }}void spift::updateWithRowShift( int shift, int pos){    for (int x = 0; x < this->matrixDim; ++x) {        for (int y = 0; y < this->matrixDim; ++y) {            int coalescencePos = ((y + x * shift) % this->matrixDim) * 2;            //std::cout << x << ", " << y << ": " << coalescencePos << std::endl;            this->dev_matrix[pos][2*(y + this->matrixDim * x)] += this->coalescence[shift][coalescencePos];            this->dev_matrix[pos][2*(y + this->matrixDim * x)+1] += this->coalescence[shift][coalescencePos+1];        }    }}void spift::updateWithColumnShift(int shift, int pos){    for (int x = 0; x < this->matrixDim; ++x) {        for (int y = 0; y < this->matrixDim; ++y) {            int coalescencePos = ((y * shift + x) % this->matrixDim) * 2;            //std::cout << x << ", " << y << ": " << coalescencePos << std::endl;            this->dev_matrix[pos][2*(y + this->matrixDim * x)] += this->coalescence[shift][coalescencePos];            this->dev_matrix[pos][2*(y + this->matrixDim * x)+1] += this->coalescence[shift][coalescencePos+1];        }    }}void spift::divedeN2() {    for (int y = 0; y < this->matrixDim; ++y) {        for (int x = 0; x < this->matrixDim*2; ++x) {            this->dev_matrix[0][x + this->matrixDim * y] /= (float)(this->matrixDim * this->matrixDim);        }    }}bool spift::shiftType(int u, int v) {       //true: RowShift,   false: ColumnShift    return !(v == 0 || ((u % 2) && !(v % 2)) || ((std::gcd(u,this->matrixDim) < std::gcd(v, this->matrixDim)) && v % 2 == 0));}int spift::shiftIndex(int u, int v, bool isRowShift) {    int uk;    int vk;    if (u == 0 || v == 0) {        return 0;    }    if (isRowShift) {        uk = u;        vk = v;    }    else    {        uk = v;        vk = u;    }    for (int j = 0; j <= this->matrixDim; ++j) {        if (uk == (j * vk) % this->matrixDim) {            return j;        }    }    std::cout << u << ", " << v << ", " << isRowShift << std::endl;    throw 15;}float* spift::computeShift(int u, int v, std::complex<float> vis, bool isRowShift) {    int x;    float* shift = new float[this->matrixDim * 2];    if (isRowShift) { x = v; }    else { x = u; }    for (int j = 0; j < matrixDim; ++j) {        std::complex<float> next = vis * this->twiddleFactors[(j * x) % this->matrixDim];        shift[2 * j] = next.real();        shift[2 * j + 1] = next.imag();    }    return shift;}void spift::initTwiddle() {    this->twiddleFactors = new std::complex<float>[this->matrixDim];    for(int k = 0; k < this->matrixDim; ++k){        std::complex<float> next = std::exp(std::complex<float>(0, k * 2 * M_PI / this->matrixDim));        this->twiddleFactors[k] = next;    }}spift::~spift(){    std::cout << "desctr" << std::endl;    for (int i = 0; i < this->nrGPUS; ++i) {        delete(dev_matrix[i]);    }    delete(this->dev_matrix);    std::cout << "dev Matrix" << std::endl;    delete coalescence;    std::cout << "coalescence" << std::endl;    for (int i = 0; i < matrixDim * 2; ++i) {        //std::cout << this->coalescence[i] << std::endl;        delete this->shiftIndexMutex[i];    }    delete this->shiftIndexMutex;    std::cout << "input destructor" << std::endl;    delete this->coalescenceSet;    delete this->done;    delete this->readMutex;    delete twiddleFactors;}void parallel(const int nrGPUs, const int dim, std::ofstream *times, const int concurrency) {    spift* tester = new spift(dim, nrGPUs, concurrency, "testData8.txt");    //tester->launchRead();    //tester->splitIteration();    //std::cout << concurrency << "\t" << dim << "\t" << nrGPUs << std::endl;    std::thread shifts(&spift::launchRead, tester);    std::thread iter(&spift::splitIteration, tester);    shifts.join();    iter.join();    tester->writeToFile(times);    //tester->printResult(0);    /*std::cout << "testing" << std::endl;    if (tester->testResult("originalData8.txt")) {        std::cout << "success" << std::endl;    }    else    {        std::cout << "failed" << std::endl;    }    */    delete tester;}int main(){    std::ofstream times;    times.open("timesGPU.txt");    //parallel(2, std::pow(2, 3), &times, 1);    /*    for (int i = 0; i < 3; i++) {        parallel(6, std::pow(2, 12), &times, 20);    }    for (int i = 0; i < 1; i++) {        parallel(1, std::pow(2, 10), &times, 1);    }    for (int i = 0; i < 3; i++) {        parallel(1, std::pow(2, 12), &times, 20);    }    for (int i = 0; i < 3; i++) {        parallel(6, std::pow(2, 12), &times, 40);    }    */    for (int concurrency = 8; concurrency < 30; concurrency += 2) {        for (int j = 10; j < 13; ++j) {            parallel(1, std::pow(2,j), &times, concurrency);            parallel(6, std::pow(2,j), &times, concurrency);            parallel(20, std::pow(2,j), &times, concurrency);            parallel(40, std::pow(2,j), &times, concurrency);            parallel(60, std::pow(2,j), &times, concurrency);        }    }    for (int concurrency = 30; concurrency < 70; concurrency += 5) {        for (int j = 10; j < 13; ++j) {            parallel(1, std::pow(2,j), &times, concurrency);            parallel(6, std::pow(2,j), &times, concurrency);            parallel(10, std::pow(2,j), &times, concurrency);            if(concurrency<=60){                parallel(20, std::pow(2,j), &times, concurrency);            }            if(concurrency<=50){                parallel(30, std::pow(2,j), &times, concurrency);            }        }    }    //parallel(6, std::pow(2,10), &times, 4);    for (int concurrency = 1; concurrency < 3; concurrency += 1) {        for (int j = 10; j < 13; ++j) {            parallel(1, std::pow(2,j), &times, concurrency);            parallel(6, std::pow(2,j), &times, concurrency);            parallel(20, std::pow(2,j), &times, concurrency);            parallel(40, std::pow(2,j), &times, concurrency);            parallel(60, std::pow(2,j), &times, concurrency);            parallel(70, std::pow(2,j), &times, concurrency);        }    }        //std::cout << "done3" << std::endl;    times.close();    return 0;}