/* Copyright 2019 Stanford University
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
void softmax_backward(DATATYPE* logitsGrad,
                      const DATATYPE* labels,
                      const int* mask,
                      int hiddenDim,
                      V_ID numVertices)
{
  CUDA_KERNEL_LOOP(i, hiddenDim * numVertices)
  {
    logitsGrad[i] -= labels[i];
    int myVtxID = i % hiddenDim;
    if (mask[myVtxID] == 0)
      logitsGrad[i] = 0;
  }
}

__host__
void SoftmaxCrossEntropy::backward_task(const Task *task,
                                        const std::vector<PhysicalRegion>& regions,
                                        Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3 or regions.size() == 4);
  assert(regions.size() == task->regions.size());
  const SoftmaxCrossEntropy* op = (SoftmaxCrossEntropy*) task->args;
  // assert the three inputs need reset gradient
  assert(op->resetInputGrads[0]);
  assert(op->resetInputGrads[1]);
  assert(op->resetInputGrads[2]);
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorRO<DATATYPE, 2> accLogits(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorRO<DATATYPE, 2> accLabels(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  TensorAccessorWO<DATATYPE, 2> accLogitsGrad(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, manager);
  assert(accLogits.memory.kind() == Memory::Z_COPY_MEM);
  assert(accLabels.memory.kind() == Memory::Z_COPY_MEM);
  assert(accLogitsGrad.memory.kind() == Memory::Z_COPY_MEM);
  V_ID rowLeft = accLogits.rect.lo[1], rowRight = accLogits.rect.hi[1];
  int hiddenDim = accLogits.rect.hi[0] - accLogits.rect.lo[0] + 1;
  if (regions.size() == 4) {
    TensorAccessorRO<int, 2> accMask(
        regions[3], task->regions[3], FID_DATA, ctx, runtime, manager);
    assert(accLogits.rect == accLabels.rect);
    assert(accLogits.rect == accLogitsGrad.rect);
    assert(accMask.rect.lo[0] == accMask.rect.hi[0]);
    assert(accMask.rect.lo[1] == rowLeft);
    assert(accMask.rect.hi[1] == rowRight);
    
    cudnnTensorDescriptor_t inputDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    int dims[] = {(int)(rowRight - rowLeft + 1), hiddenDim, 1, 1};
    int strides[] = {dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT,
        4, dims, strides));
    float alpha = 1.0f, beta = 0.0f;
    checkCUDNN(cudnnSoftmaxForward(manager->dnn, CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, inputDesc, accLogits.fbCache,
        &beta, inputDesc, accLogitsGrad.fbCache));
    softmax_backward<<<GET_BLOCKS(accLogits.rect.volume()), CUDA_NUM_THREADS>>>(
        accLogitsGrad.fbCache, accLabels.fbCache, accMask.fbCache,
        hiddenDim, rowRight - rowLeft + 1);
  } else {
    assert(false);
  }
}
