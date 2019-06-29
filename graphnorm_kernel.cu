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

#include "gnn.h"
#include "cuda_helper.h"

__global__
void norm_coop_kernel(V_ID rowLeft,
                      V_ID rowRight,
                      E_ID colLeft,
                      int hiddenDim,
                      const NodeStruct* row_ptrs,
                      const DATATYPE *input,
                      DATATYPE* output)
{
  assert(blockDim.x == CUDA_NUM_THREADS);
  __shared__ V_ID inDegree[CUDA_NUM_THREADS];
  for (V_ID blkRowStart = blockIdx.x * blockDim.x + rowLeft;
       blkRowStart <= rowRight;
       blkRowStart += blockDim.x * gridDim.x)
  {
    if (blkRowStart + threadIdx.x <= rowRight)
    {
      V_ID curVtx = threadIdx.x + blkRowStart;
      E_ID startColIdx, endColIdx = row_ptrs[curVtx-rowLeft].index;
      if (curVtx == rowLeft)
        startColIdx = colLeft;
      else
        startColIdx = row_ptrs[curVtx-rowLeft-1].index;
      inDegree[threadIdx.x] = endColIdx - startColIdx;
    }
    __syncthreads();
    E_ID todo = min(blockDim.x, rowRight+1-blkRowStart) * hiddenDim;
    E_ID done = 0;
    while (todo > 0) {
      if (threadIdx.x < todo) {
        output[(blkRowStart-rowLeft)*hiddenDim+done+threadIdx.x] =
          input[(blkRowStart-rowLeft)*hiddenDim+done+threadIdx.x]
            / sqrt((float)inDegree[(done+threadIdx.x)/hiddenDim]);
      }
      done += blockDim.x;
      todo -= (todo > blockDim.x) ? blockDim.x : todo;
    }
  }
}

__host__
void InDegreeNorm::forward_task(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorRO<NodeStruct, 1> accRowPtr(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorRO<EdgeStruct, 1> accColIdx(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  TensorAccessorRO<DATATYPE, 2> accInput(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, manager);
  TensorAccessorWO<DATATYPE, 2> accOutput(
      regions[3], task->regions[3], FID_DATA, ctx, runtime, manager);
  // Assert memories are correctly mapped
  assert(accRowPtr.memory.kind() == Memory::GPU_FB_MEM);
  assert(accColIdx.memory.kind() == Memory::GPU_FB_MEM);
  assert(accInput.memory.kind() == Memory::Z_COPY_MEM);
  assert(accOutput.memory.kind() == Memory::Z_COPY_MEM);
#ifdef DEADCODE
  const AccessorRO<NodeStruct, 1> accRowPtr(regions[0], FID_DATA);
  const AccessorRO<EdgeStruct, 1> accColIdx(regions[1], FID_DATA);
  const AccessorRO<DATATYPE, 2> accInput(regions[2], FID_DATA);
  const AccessorWO<DATATYPE, 2> accOutput(regions[3], FID_DATA);
  Rect<1> rectRowPtr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rectColIdx = runtime->get_index_space_domain(
                             ctx, task->regions[1].region.get_index_space());
  Rect<2> rectInput = runtime->get_index_space_domain(
                            ctx, task->regions[2].region.get_index_space());
  Rect<2> rectOutput = runtime->get_index_space_domain(
                            ctx, task->regions[3].region.get_index_space());
  assert(accRowPtr.accessor.is_dense_arbitrary(rectRowPtr));
  assert(accColIdx.accessor.is_dense_arbitrary(rectColIdx));
  assert(accInput.accessor.is_dense_arbitrary(rectInput));
  assert(accOutput.accessor.is_dense_arbitrary(rectOutput));
  const NodeStruct* rowPtrs = accRowPtr.ptr(rectRowPtr);
  const EdgeStruct* colIdxs = accColIdx.ptr(rectColIdx);
  const DATATYPE* zcInput = accInput.ptr(rectInput);
  DATATYPE* zcOutput = accOutput.ptr(rectOutput);
#endif
  V_ID rowLeft = accRowPtr.rect.lo[0], rowRight = accRowPtr.rect.hi[0];
  E_ID colLeft = accColIdx.rect.lo[0], colRight = accColIdx.rect.hi[0];
  int hiddenDim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  assert(accOutput.rect == accInput.rect);
  assert(accOutput.rect.lo[1] == accRowPtr.rect.lo[0]);
  assert(accOutput.rect.hi[1] == accRowPtr.rect.hi[0]);
  norm_coop_kernel<<<GET_BLOCKS(rowRight-rowLeft+1), CUDA_NUM_THREADS>>>(
      rowLeft, rowRight, colLeft, hiddenDim, accRowPtr.ptr,
      accInput.fbCache, accOutput.fbCache); 
  checkCUDA(cudaMemcpy(accOutput.ptr, accOutput.fbCache,
                       accOutput.rect.volume() * sizeof(DATATYPE),
                       cudaMemcpyDeviceToHost));

  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      printf("[InDegreeNorm] Input[%d][%d]: %.4lf\n", i, j, accInput.ptr[i * hiddenDim + j]);
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      printf("[InDegreeNorm] Output[%d][%d]: %.4lf\n", i, j, accOutput.ptr[i * hiddenDim + j]);
}

__host__
void InDegreeNorm::backward_task(const Task *task,
                                 const std::vector<PhysicalRegion> &regions,
                                 Context ctx, Runtime *runtime)
{
  const InDegreeNorm* op = (InDegreeNorm*) task->args;
  // assert that we should reset input gradient
  assert(op->resetInputGrads[0]);
  // Forward and backward do exact same thing
  return forward_task(task, regions, ctx, runtime);
}


