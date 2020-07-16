
__global__ void updateWithRowShift(cuFloatComplex* dev_matrix, cudaTextureObject_t dev_vector, const int matrix_dim, const int shift)
{
    int start_index = blockIdx.x * blockDim.x * matrix_dim + blockIdx.y * blockDim.x + threadIdx.x;
    int vectorPos = (shift * blockIdx.x * blockDim.x + threadIdx.x + blockIdx.y * blockDim.x) % matrix_dim;
    for (int i = 0; i < blockDim.x; i++) {
        dev_matrix[start_index + i * matrix_dim].x += tex2D<float>(dev_vector, 2*vectorPos, 0);
        dev_matrix[start_index + i * matrix_dim].y += tex2D<float>(dev_vector, 2*vectorPos + 1, 0);
        vectorPos += shift;
        vectorPos %= matrix_dim;
    }
}



