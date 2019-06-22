#include "cuda_helper.h"
__global__
void scale_kernel(float* ptr, coord_t size, float a, float b)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = (b - a) * ptr[i] + a;
  }
}

__global__
void assign_kernel(float* ptr, coord_t size, float value)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = value;
  }
}

