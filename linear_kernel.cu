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

__host__
void Linear::forward_task(const Task *task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const Linear* op = (Linear*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorRO<DATATYPE, 2> accWeight(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorRO<DATATYPE, 2> accInput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  TensorAccessorWO<DATATYPE, 2> accOutput(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, manager);
  // Assert that regions are mapped correctly
  assert(accWeight.memory.kind() == Memory::GPU_FB_MEM);
  assert(accInput.memory.kind() == Memory::Z_COPY_MEM);
  assert(accOutput.memory.kind() == Memory::Z_COPY_MEM);
#ifdef DEADCODE
  const AccessorRO<DATATYPE, 1> accWeight(regions[0], FID_DATA);
  const AccessorRO<DATATYPE, 2> accInput(regions[1], FID_DATA);
  const AccessorWO<DATATYPE, 2> accOutput(regions[2], FID_DATA);
  Rect<1> rectWeight = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rectInput = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rectOutput = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(accWeight.accessor.is_dense_arbitrary(rectWeight));
  assert(accInput.accessor.is_dense_arbitrary(rectInput));
  assert(accOutput.accessor.is_dense_arbitrary(rectOutput));
  const DATATYPE* fbWeight = accWeight.ptr(rectWeight);
  const DATATYPE* zcInput = accInput.ptr(rectInput);
  const DATATYPE* zcOutput = accInput.ptr(rectOutput);
  assert(rectInput.lo[1] == rectOutput.lo[1]);
  assert(rectInput.hi[1] == rectOutput.hi[1]);
#endif
  // Weight matches outDim
  assert(accWeight.rect.hi[1] == accOutput.rect.hi[0]);
  assert(accWeight.rect.lo[1] == accOutput.rect.lo[0]);
  // Weight matches inDim
  assert(accWeight.rect.hi[0] == accInput.rect.hi[0]);
  assert(accWeight.rect.lo[0] == accInput.rect.lo[0]);
  // input matches output
  assert(accInput.rect.lo[1] == accOutput.rect.lo[1]);
  assert(accInput.rect.hi[1] == accOutput.rect.hi[1]);
  V_ID rowLeft = accInput.rect.lo[1], rowRight = accInput.rect.hi[1];
  int inDim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  int outDim = accOutput.rect.hi[0] - accOutput.rect.lo[0] + 1;
  // Test
  cudnnDropoutDescriptor_t dropoutDesc;
  cudnnTensorDescriptor_t inputDesc, ouputDesc;
  cudnnCreateDropoutDescriptor(&dropoutDesc);
  cudnnCreateTensorDescriptor(&inputDesc);
  double ts_start = Realm::Clock::current_time_in_microseconds();
  checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc, manager->dnn, 0.5, manager->dropoutStates, manager->dropoutSize, 10));
  double ts_end = Realm::Clock::current_time_in_microseconds();
  int dims[] = {rowRight - rowLeft + 1, inDim, 1};
  int strides[] = {dims[1] * dims[2], dims[2], 1};
  checkCUDNN(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT, 3, dims, strides));
  size_t size;
  checkCUDNN(cudnnDropoutGetReserveSpaceSize(inputDesc, &size));
  printf("dims = (%d %d %d) size = %zu time = %.4lfus\n", dims[0], dims[1], dims[2], size, ts_end - ts_start);
  // Test
}

__host__
void Linear::backward_task(const Task *task,
                           const std::vector<PhysicalRegion>& regions,
                           Context ctx, Runtime* runtime)
{
  assert(regions.size() == 5);
  assert(task->regions.size() == 5);
  const Linear* op = (Linear*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorRO<DATATYPE, 1> accWeight(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorRO<DATATYPE, 2> accOutputGrad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  TensorAccessorRO<DATATYPE, 2> accInput(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, manager);
  TensorAccessorWO<DATATYPE, 1> accWeightGrad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime, manager);
  TensorAccessorWO<DATATYPE, 2> accInputGrad(
      regions[4], task->regions[4], FID_DATA, ctx, runtime, manager);
  // Assert that memories are correctly mapped
  assert(accWeight.memory.kind() == Memory::GPU_FB_MEM);
  assert(accOutputGrad.memory.kind() == Memory::Z_COPY_MEM);
  assert(accInput.memory.kind() == Memory::Z_COPY_MEM);
  assert(accWeightGrad.memory.kind() == Memory::GPU_FB_MEM);
  assert(accInputGrad.memory.kind() == Memory::Z_COPY_MEM);
#ifdef DEADCODE
  const AccessorRO<DATATYPE, 1> accWeight(regions[0], FID_DATA);
  const AccessorRO<DATATYPE, 2> accOutputGrad(regions[1], FID_DATA);
  const AccessorRO<DATATYPE, 2> accInput(regions[2], FID_DATA);
  const AccessorWO<DATATYPE, 1> accWeightGrad(regions[3], FID_DATA);
  const AccessorWO<DATATYPE, 2> accInputGrad(regions[4], FID_DATA);
  Rect<1> rectWeight = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<2> rectOutputGrad = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<2> rectInput = runtime->get_index_space_domain(
      ctx, task->regions[3].region.get_index_space());
  Rect<1> rectWeightGrad = runtime->get_index_space_domain(
      ctx, task->regions[4].region.get_index_space());
  Rect<2> rectInputGrad = runtime->get_index_space_domain(
      ctx, task->regions[5].region.get_index_space());
  assert(accWeight.accessor.is_dense_arbitrary(rectWeight));
  assert(accOutputGrad.accessor.is_dense_arbitrary(rectOutputGrad));
  assert(accInput.accessor.is_dense_arbitrary(rectInput));
  assert(accWeightGrad.accessor.is_dense_arbitrary(rectWeightGrad));
  assert(accInputGrad.accessor.is_dense_arbitrary(rectInputGrad));
  const DATATYPE* fbWeight = accWeight.ptr(rectWeight);
  const DATATYPE* zcOutputGrad = accOutputGrad.ptr(rectOutputGrad);
  const DATATYPE* zcInput = accInput.ptr(rectInput);
  DATATYPE* fbWeightGrad = accWeightGrad.ptr(rectWeightGrad);
  DATATYPE* zcInputGrad = accInputGrad.ptr(rectInputGrad);
#endif
}

__host__
void Linear::update_task(const Task *task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime* runtime)
{
  assert(false);
}
