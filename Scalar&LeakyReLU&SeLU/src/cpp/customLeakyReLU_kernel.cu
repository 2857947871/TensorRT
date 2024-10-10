# include <math.h>
# include <cuda_runtime.h>


// def forward(ctx, x, slope):
//     return torch.where(x > 0, x, x * slope)
__global__ void leakyReLUKernel(const float* x, float* y, const float slope, int element_nums) {

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < element_nums)
    {
        y[gid] = x[gid] > 0 ? x[gid] : x[gid] * slope;
    } else {
        return;
    }
}

void customLeakyReLUImply(const float* x, float* y, const float slope, int element_nums)
{
    dim3 block(1024);
    dim3 grid((element_nums + block.x - 1) / block.x);

    leakyReLUKernel<<<grid, block>>>(x, y, slope, element_nums);
}