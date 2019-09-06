/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "optimizer.h"
#include "types.h"
#include "cuda_helper.h"

LegionRuntime::Logger::Category log_parameter("optimizer");

__global__
void add_kernel(int count, DATATYPE scale,
                const DATATYPE* src,
                DATATYPE* dst)
{
  CUDA_KERNEL_LOOP(i, count)
  {
    dst[i] += src[i] * scale;
  }
}

__global__
void scale_kernel(int count, DATATYPE a, DATATYPE b,
                  DATATYPE* ptr)
{
  CUDA_KERNEL_LOOP(i, count)
  {
    ptr[i] = (b - a) * ptr[i] + a;
  }
}

__global__
void adam_update(int count, DATATYPE alpha_t,
                 DATATYPE beta1, DATATYPE beta2,
                 DATATYPE weight_decay, DATATYPE epsilon,
                 const DATATYPE *WGrad, DATATYPE *M,
                 DATATYPE *V, DATATYPE *W)
{
  // Reference for weight decay
  // https://www.fast.ai/2018/07/02/adam-weight-decay/
  CUDA_KERNEL_LOOP(i, count)
  {
    //W[i] -= weight_decay * alpha_t * W[i];
    //DATATYPE gt = WGrad[i];
    DATATYPE gt = WGrad[i] + weight_decay * W[i];
    DATATYPE mt = beta1 * M[i] + (1 - beta1) * gt;
    DATATYPE vt = beta2 * V[i] + (1 - beta2) * gt * gt;
    M[i] = mt;
    V[i] = vt;
    W[i] -= alpha_t * mt / (sqrt(vt) + epsilon);
  }
}

__host__
void AdamOptimizer::update_task(const Task* task,
                                const std::vector<PhysicalRegion>& regions,
                                Context ctx, Runtime* runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const AdamOptimizer* op = (AdamOptimizer*) task->args;
  TensorAccessorR<DATATYPE, 2> accWGrad(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, NULL);
  TensorAccessorW<DATATYPE, 2> accW(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, NULL,
      true/*readOutput*/);
  TensorAccessorW<DATATYPE, 2> accV(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, NULL,
      true/*readOutput*/);
  TensorAccessorW<DATATYPE, 2> accM(
      regions[3], task->regions[3], FID_DATA, ctx, runtime, NULL,
      true/*readOutput*/);
  assert(accWGrad.memory.kind() == Memory::GPU_FB_MEM);
  assert(accW.memory.kind() == Memory::GPU_FB_MEM);
  assert(accV.memory.kind() == Memory::GPU_FB_MEM);
  assert(accM.memory.kind() == Memory::GPU_FB_MEM);
  int numReplicas = accWGrad.rect.volume() / accW.rect.volume();
  // Step 1: gather gradients in the first replica
  for (int i = 1; i < numReplicas; i++) {
    const DATATYPE* src = accWGrad.ptr + i * accW.rect.volume();
    add_kernel<<<GET_BLOCKS(accW.rect.volume()), CUDA_NUM_THREADS>>>(
        accW.rect.volume(), 1.0f, src, (DATATYPE*)accWGrad.ptr);
  }
  //fprintf(stderr, "alpha = %.8lf alpha_t = %.8lf decay = %.8lf\n",
  //        op->alpha, op->alpha_t, op->weight_decay);
  // Step 2: Adam update
  adam_update<<<GET_BLOCKS(accW.rect.volume()), CUDA_NUM_THREADS>>>(
      accW.rect.volume(), op->alpha_t, op->beta1, op->beta2,
      op->weight_decay, op->epsilon,
      accWGrad.ptr, accM.ptr, accV.ptr, accW.ptr);
  checkCUDA(cudaDeviceSynchronize());
}

