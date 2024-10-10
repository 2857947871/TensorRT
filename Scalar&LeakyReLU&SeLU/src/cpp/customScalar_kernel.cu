# include <math.h>
# include <cuda_runtime.h>


// def forward(ctx, x, r, s):
//     return (x + r) * s
__global__ void customScalarKernel(const float* x, float* y, const float r, const float s, int element_nums)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < element_nums)
    {
        y[gid] = (x[gid] + r) * s;
    } else {
        return;
    }
}

void customScalarImply(const float* x, float* y, const float r, const float s, int element_nums)
{
    dim3 block(1024);
    dim3 grid((element_nums + block.x - 1) / block.x);

    customScalarKernel<<<grid, block>>>(x, y, r, s, element_nums);
}