# include <math.h>
# include <cuda_runtime.h>

__global__ void SELUKernel(const float* x, float* y, const float alpha, const float lambda, int element_nums) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = gid; i < element_nums; i += blockDim.x * gridDim.x) {

        y[i] = x[i] > 0 ? lambda * x[i] : lambda * alpha * (exp(x[i]) - 1);
    }
}

void customSELUImply(const float* x, float* y, const float alpha, const float lambda, int element_nums) {

    dim3 block(1024);
    dim3 grid((element_nums + block.x - 1) / block.x);

    SELUKernel<<<grid, block>>>(x, y, alpha, lambda, element_nums);
}