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
void op_kernel(const DATATYPE* input0,
               const DATATYPE* input1,
               DATATYPE* output,
               coord_t size,
               ElementType type)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    switch(type) {
      case EW_TYPE_ADD:
        output[i] = input0[i] + input1[i];
        break;
      case EW_TYPE_MUL:
        output[i] = input0[i] * input1[i];
        break;
      default:
        assert(false);
    }
  }
}

__host__
void Element::forward_task(const Task *task,
                           const std::vector<PhysicalRegion>& regions,
                           Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const Element* op = (Element*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorR<DATATYPE, 2> accInput0(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorR<DATATYPE, 2> accInput1(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  TensorAccessorW<DATATYPE, 2> accOutput(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, manager,
      false/*readOutput*/);
  assert(accInput0.rect == accInput1.rect);
  assert(accOutput.rect == accInput0.rect);
  assert(accInput0.memory.kind() == Memory::Z_COPY_MEM);
  assert(accInput1.memory.kind() == Memory::Z_COPY_MEM);
  assert(accOutput.memory.kind() == Memory::Z_COPY_MEM);
  op_kernel<<<GET_BLOCKS(accOutput.rect.volume()), CUDA_NUM_THREADS>>>(
      accInput0.fbCache, accInput1.fbCache, accOutput.fbCache,
      accOutput.rect.volume(), op->elementType);
  checkCUDA(cudaMemcpy(accOutput.ptr, accOutput.fbCache,
                       accOutput.rect.volume() * sizeof(DATATYPE),
                       cudaMemcpyDeviceToHost));
}

__host__
void Element::backward_task(const Task *task,
                            const std::vector<PhysicalRegion>& regions,
                            Context ctx, Runtime* runtime)
{
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);
  const Element* op = (Element*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorR<DATATYPE, 2> accOutputGrad(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorW<DATATYPE, 2> accInput0Grad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager,
      !(op->resetInputGrads[0])/*readOutput*/);
  TensorAccessorW<DATATYPE, 2> accInput1Grad(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, manager,
      !(op->resetInputGrads[1])/*readOutput*/);
  assert(accOutputGrad.rect == accInput0Grad.rect);
  assert(accOutputGrad.rect == accInput1Grad.rect);
  switch (op->elementType) {
    case EW_TYPE_ADD:
    {
      add_kernel<<<GET_BLOCKS(accOutputGrad.rect.volume()), CUDA_NUM_THREADS>>>(
          accInput0Grad.fbCache, accOutputGrad.fbCache, accOutputGrad.rect.volume());
      add_kernel<<<GET_BLOCKS(accOutputGrad.rect.volume()), CUDA_NUM_THREADS>>>(
          accInput1Grad.fbCache, accOutputGrad.fbCache, accOutputGrad.rect.volume());
      break;
    }
    case EW_TYPE_MUL:
    default:
      assert(false);
  }
  checkCUDA(cudaMemcpy(accInput0Grad.ptr, accInput0Grad.fbCache,
                       accInput0Grad.rect.volume() * sizeof(DATATYPE),
                       cudaMemcpyDeviceToHost));
  checkCUDA(cudaMemcpy(accInput1Grad.ptr, accInput1Grad.fbCache,
                       accInput1Grad.rect.volume() * sizeof(DATATYPE),
                       cudaMemcpyDeviceToHost));
}
