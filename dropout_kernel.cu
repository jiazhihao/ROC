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
void Dropout::init_task(const Task *task,
                        const std::vector<PhysicalRegion>& regions,
                        Context ctx, Runtime* runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  const Dropout* op = (Dropout*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorRO<DATATYPE, 2> accInput(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  assert(accInput.memory.kind() == Memory::Z_COPY_MEM);
  V_ID rowLeft = accInput.rect.lo[1], rowRight = accInput.rect.hi[1];
  int inDim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  cudnnDropoutDescriptor_t dropoutDesc;
  cudnnTensorDescriptor_t inputDesc;
  checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
  checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc, manager->dnn, op->rate,
      manager->dropoutStates, manager->dropoutSize, op->seed));
  int dims[] = {(int)(rowRight - rowLeft + 1), inDim, 1};
  int strides[] = {dims[1] * dims[2], dims[2], 1};
  checkCUDNN(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT,
      3, dims, strides));
  ResourceManager::ReservedSpace space;
  checkCUDNN(cudnnDropoutGetReserveSpaceSize(inputDesc, &(space.size)));
  off_t offset = manager->allocator->alloc_bytes(space.size);
  assert(offset >= 0);
  space.ptr = manager->allocator->get_direct_ptr(offset, 0);
  LogicalRegion my_lr = regions[0].get_logical_region();
  assert(manager->reservedSpace.find(my_lr) == manager->reservedSpace.end());
  manager->reservedSpace[my_lr] = space;
  manager->dropoutDesc[my_lr] = dropoutDesc;
  checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
}

__host__
void Dropout::forward_task(const Task *task,
                           const std::vector<PhysicalRegion>& regions,
                           Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Dropout* op = (Dropout*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  TensorAccessorRO<DATATYPE, 2> accInput(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorWO<DATATYPE, 2> accOutput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  assert(accInput.memory.kind() == Memory::Z_COPY_MEM);
  assert(accOutput.memory.kind() == Memory::Z_COPY_MEM);
  V_ID rowLeft = accInput.rect.lo[1], rowRight = accInput.rect.hi[1];
  int inDim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;

  double ts_start = Realm::Clock::current_time_in_microseconds();
  cudnnTensorDescriptor_t inputDesc, outputDesc;
  checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputDesc));
  int dims[] = {(int)(rowRight - rowLeft + 1), inDim, 1};
  int strides[] = {dims[1] * dims[2], dims[2], 1};
  checkCUDNN(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT,
      3, dims, strides));
  checkCUDNN(cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT,
      3, dims, strides));
  LogicalRegion my_lr = regions[0].get_logical_region();
  assert(manager->reservedSpace.find(my_lr) != manager->reservedSpace.end());
  ResourceManager::ReservedSpace space = manager->reservedSpace[my_lr];
  cudnnDropoutDescriptor_t dropoutDesc = manager->dropoutDesc[my_lr];
  double ts_end = Realm::Clock::current_time_in_microseconds();
  printf("[Dropout::Fwd] pre-process %.4lfus\\n", ts_end - ts_start);
  checkCUDNN(cudnnDropoutForward(manager->dnn, dropoutDesc, inputDesc,
      accInput.fbCache, outputDesc, accOutput.fbCache, space.ptr, space.size));
  checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputDesc));
}

__host__
void Dropout::backward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  //const Dropout* op = (Dropout*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  TensorAccessorRO<DATATYPE, 2> accOutputGrad(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorWO<DATATYPE, 2> accInputGrad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  assert(accOutputGrad.memory.kind() == Memory::Z_COPY_MEM);
  assert(accInputGrad.memory.kind() == Memory::Z_COPY_MEM);
  V_ID rowLeft = accInputGrad.rect.lo[1], rowRight = accInputGrad.rect.hi[1];
  int inDim = accInputGrad.rect.hi[0] - accInputGrad.rect.lo[0] + 1;

  double ts_start = Realm::Clock::current_time_in_microseconds();
  cudnnTensorDescriptor_t inputDesc, outputDesc;
  checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&outputDesc));
  int dims[] = {(int)(rowRight - rowLeft + 1), inDim, 1};
  int strides[] = {dims[1] * dims[2], dims[2], 1};
  checkCUDNN(cudnnSetTensorNdDescriptor(inputDesc, CUDNN_DATA_FLOAT,
      3, dims, strides));
  checkCUDNN(cudnnSetTensorNdDescriptor(outputDesc, CUDNN_DATA_FLOAT,
      3, dims, strides));
  LogicalRegion my_lr = regions[0].get_logical_region();
  assert(manager->reservedSpace.find(my_lr) != manager->reservedSpace.end());
  ResourceManager::ReservedSpace space = manager->reservedSpace[my_lr];
  cudnnDropoutDescriptor_t dropoutDesc = manager->dropoutDesc[my_lr];
  double ts_end = Realm::Clock::current_time_in_microseconds();
  printf("[Dropout::Bwd] pre-process %.4lfus\\n", ts_end - ts_start);
  checkCUDNN(cudnnDropoutBackward(manager->dnn, dropoutDesc, inputDesc,
      accOutputGrad.fbCache, outputDesc, accInputGrad.fbCache, space.ptr, space.size));
  checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
  checkCUDNN(cudnnDestroyTensorDescriptor(outputDesc));
}
