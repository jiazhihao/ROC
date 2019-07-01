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
void Activation::forward_task(const Task *task,
                              const std::vector<PhysicalRegion>& regions,
                              Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  const Activation* op = (Activation*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorR<DATATYPE, 2> accInput(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorW<DATATYPE, 2> accOutput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager,
      false/*readOutput*/);
  assert(accInput.memory.kind() == Memory::Z_COPY_MEM);
  assert(accOutput.memory.kind() == Memory::Z_COPY_MEM);
  assert(accInput.rect == accOutput.rect);
  V_ID rowLeft = accInput.rect.lo[1], rowRight = accInput.rect.hi[1];
  int hiddenDim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  double ts_start = Realm::Clock::current_time_in_microseconds();
  cudnnTensorDescriptor_t inTensor;
  cudnnActivationDescriptor_t actiDesc;
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&inTensor));
  int dims[] = {(int)(rowRight - rowLeft + 1), hiddenDim, 1};
  int strides[] = {dims[1] * dims[2], dims[2], 1};
  float alpha = 1.0f, beta = 0.0f;
  checkCUDNN(cudnnSetTensorNdDescriptor(inTensor, CUDNN_DATA_FLOAT,
      3, dims, strides));
  switch (op->actiMode) {
    case AC_MODE_RELU:
      checkCUDNN(cudnnSetActivationDescriptor(
          actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
      break;
    case AC_MODE_SIGMOID:
      checkCUDNN(cudnnSetActivationDescriptor(
          actiDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));
      break;
    default:
      assert(false);
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  printf("[Activation:forward] preprocess(%.2lfus)\n", ts_end - ts_start);
  checkCUDNN(cudnnActivationForward(manager->dnn, actiDesc,
                                    &alpha, inTensor, accInput.fbCache,
                                    &beta, inTensor, accOutput.fbCache));
  checkCUDA(cudaMemcpy(accOutput.ptr, accOutput.fbCache,
                       accOutput.rect.volume() * sizeof(DATATYPE),
                       cudaMemcpyDeviceToHost));
  checkCUDNN(cudnnDestroyTensorDescriptor(inTensor));
  checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
}

__host__
void Activation::backward_task(const Task *task,
                               const std::vector<PhysicalRegion>& regions,
                               Context ctx, Runtime* runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Activation* op = (Activation*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorR<DATATYPE, 2> accOutputGrad(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorR<DATATYPE, 2> accOutput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  TensorAccessorR<DATATYPE, 2> accInput(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, manager);
  TensorAccessorW<DATATYPE, 2> accInputGrad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime, manager,
      !(op->resetInputGrads[0])/*readOutput*/);
  assert(accOutput.memory.kind() == Memory::Z_COPY_MEM);
  assert(accOutputGrad.memory.kind() == Memory::Z_COPY_MEM);
  assert(accInput.memory.kind() == Memory::Z_COPY_MEM);
  assert(accInputGrad.memory.kind() == Memory::Z_COPY_MEM);
  assert(accOutput.rect == accOutputGrad.rect);
  assert(accOutput.rect == accInput.rect);
  assert(accOutput.rect == accInputGrad.rect);
  double ts_start = Realm::Clock::current_time_in_microseconds();
  V_ID rowLeft = accOutput.rect.lo[1], rowRight = accOutput.rect.hi[1];
  int hiddenDim = accOutput.rect.hi[0] - accOutput.rect.lo[0] + 1;
  float alpha = 1.0f, beta = 0.0f;
  cudnnTensorDescriptor_t outTensor;
  cudnnActivationDescriptor_t actiDesc;
  checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
  checkCUDNN(cudnnCreateTensorDescriptor(&outTensor));
  int dims[] = {(int)(rowRight - rowLeft + 1), hiddenDim, 1};
  int strides[] = {dims[1] * dims[2], dims[2], 1};
  checkCUDNN(cudnnSetTensorNdDescriptor(outTensor, CUDNN_DATA_FLOAT,
        3, dims, strides));
  switch (op->actiMode) {
    case AC_MODE_RELU:
      checkCUDNN(cudnnSetActivationDescriptor(
          actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
      break;
    case AC_MODE_SIGMOID:
      checkCUDNN(cudnnSetActivationDescriptor(
          actiDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));
      break;
    default:
      assert(false);
  }
  double ts_end = Realm::Clock::current_time_in_microseconds();
  printf("[Activation:backward] preprocess(%.2lfus)\n", ts_end - ts_start);
  checkCUDNN(cudnnActivationBackward(manager->dnn, actiDesc,
      &alpha, outTensor, accOutput.fbCache,
      outTensor, accOutputGrad.fbCache,
      outTensor, accInput.fbCache,
      &alpha, outTensor, accInputGrad.fbCache));
  checkCUDA(cudaMemcpy(accInputGrad.ptr, accInputGrad.fbCache,
                       accInputGrad.rect.volume() * sizeof(DATATYPE),
                       cudaMemcpyDeviceToHost));
  checkCUDNN(cudnnDestroyTensorDescriptor(outTensor));
  checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      printf("[Activation:backward](%d, %d): outputGrad(%.4lf) output(%.4lf) input(%.4lf) inputGrad(%.4lf)\n",
             i, j, accOutputGrad.ptr[i*hiddenDim+j], accOutput.ptr[i*hiddenDim+j],
             accInput.ptr[i*hiddenDim+j], accInputGrad.ptr[i*hiddenDim+j]);
}
