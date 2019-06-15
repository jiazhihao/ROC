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
void load_input_kernel(DATATYPE* fbInputs,
                       const DATATYPE* inputs,
                       size_t size)
{
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i+= blockDim.x * gridDim.x)
  {
    fbInputs[i] = inputs[i];
  }
}

void gnn_fwd_task_impl(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx,
                       Runtime* runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const GraphPiece *piece = (GraphPiece*) task->local_args;
  const GnnOp* gnnOp = (GnnOp*) task->args;

  const AccessorRO<NodeStruct, 1> accRowPtr(regions[0], FID_DATA);
  const AccessorRO<EdgeStruct, 1> accColIdx(regions[1], FID_DATA);
  const AccessorRO<DATATYPE, 1> accInput(regions[2], FID_DATA);
  const AccessorWO<DATATYPE, 1> accOutput(regions[3], FID_DATA);
  Rect<1> rectRowPtr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rectColIdx = runtime->get_index_space_domain(
                             ctx, task->regions[1].region.get_index_space());
  Rect<1> rectInput = runtime->get_index_space_domain(
                            ctx, task->regions[2].region.get_index_space());
  Rect<1> rectOutput = runtime->get_index_space_domain(
                            ctx, task->regions[3].region.get_index_space());
  assert(accRowPtr.accessor.is_dense_arbitrary(rectRowPtr));
  assert(accColIdx.accessor.is_dense_arbitrary(rectColIdx));
  assert(accInput.accessor.is_dense_arbitrary(rectInput));
  assert(accOutput.accessor.is_dense_arbitrary(rectOutput));
  const NodeStruct* rowPtrs = accRowPtr.ptr(rectRowPtr);
  const EdgeStruct* colIdxs = accColIdx.ptr(rectColIdx);
  const DATATYPE* inputs = accInput.ptr(rectInput);
  DATATYPE* outputs = accOutput.ptr(rectOutput);
  V_ID rowLeft = rectRowPtr.lo[0], rowRight = rectRowPtr.hi[0];
  E_ID colLeft = rectColIdx.lo[0], colRight = rectColIdx.hi[0];
  assert(rectInput.volume() == (rowRight-rowLeft+1) * gnnOp->input.nodeDim);
  assert(rectOutput.volume() == (rowRight-rowLeft+1) * gnnOp->output.nodeDim);
  V_ID myVtx = (piece->numNodes+piece->numParts-1)/piece->numParts;
  size_t size = gnnOp->input.nodeDim * myVtx;
  int rank;
  NCCLCheck(ncclCommUserRank(piece->nccl, &rank));
  printf("my nccl rank = %d\n", rank);
  //assert(size * piece->numParts < piece->fbInputSize);
  cudaStream_t stream;
  checkCUDA(cudaStreamCreate(&stream));
  load_input_kernel<<<GET_BLOCKS(size), CUDA_NUM_THREADS, 0, stream>>>(
      piece->fbInput + size * rank,
      inputs + size * rank, size);
  NCCLCheck(ncclAllGather(piece->fbInput + size * rank, piece->fbInput,
                          size, ncclFloat, piece->nccl, stream));
}

