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
  TensorAccessorR<DATATYPE, 2> accWeight(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  assert(manager->assigned.size() == 0);
  TensorAccessorR<DATATYPE, 2> accInput(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  assert(manager->assigned.size() == 1);
  TensorAccessorW<DATATYPE, 2> accOutput(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, manager,
      false/*readOutput*/);
  assert(manager->assigned.size() == 2);
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
  float alpha = 1.0f, beta = 0.0f;
  checkCUDA(cublasSgemm(manager->blas, CUBLAS_OP_T, CUBLAS_OP_N,
                        outDim, rowRight-rowLeft+1, inDim,
                        &alpha, accWeight.ptr, inDim,
                        accInput.fbCache, inDim,
                        &beta, accOutput.fbCache, outDim));
  if (op->activation != AC_MODE_NONE) {
    cudnnTensorDescriptor_t outTensor;
    cudnnActivationDescriptor_t actiDesc;
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outTensor));
    int dims[] = {(int)(rowRight - rowLeft + 1), outDim, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(outTensor, CUDNN_DATA_FLOAT,
        3, dims, strides));
    switch (op->activation) {
      case AC_MODE_RELU:
        checkCUDNN(cudnnSetActivationDescriptor(
            actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
        break;
      default:
        assert(false);
    }
    checkCUDNN(cudnnActivationForward(manager->dnn, actiDesc,
                                      &alpha, outTensor, accOutput.fbCache,
                                      &beta, outTensor, accOutput.fbCache));
    checkCUDA(cudaDeviceSynchronize());
    checkCUDNN(cudnnDestroyTensorDescriptor(outTensor));
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
  checkCUDA(cudaMemcpy(accOutput.ptr, accOutput.fbCache,
                       accOutput.rect.volume() * sizeof(DATATYPE),
                       cudaMemcpyDeviceToHost));
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      printf("[Linear:forward] input[%d][%d]: %.4lf\n", i, j, accInput.ptr[i * outDim + j]);
  //for (int i = 0; i < 8; i++)
  //  for (int j = 0; j < 8; j++)
  //    printf("[Linear:forward] weight[%d][%d]: %.4lf\n", i, j, accOutput.ptr[i * outDim + j]);
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      printf("[Linear:forward] output[%d][%d]: %.4lf\n", i, j, accOutput.ptr[i * outDim + j]);
  checkCUDA(cudaDeviceSynchronize());
}

__global__
void reluBackward(float *grad_ptr, const float *output, int n)
{
  CUDA_KERNEL_LOOP(i, n)
  {
    grad_ptr[i] = (output[i] > 0.0f) ? grad_ptr[i] : 0;
  }
}

__host__
void Linear::backward_task(const Task *task,
                           const std::vector<PhysicalRegion>& regions,
                           Context ctx, Runtime* runtime)
{
  assert(regions.size() == 6);
  assert(task->regions.size() == 6);
  const Linear* op = (Linear*) task->args;
  ResourceManager* manager = *((ResourceManager**) task->local_args);
  assert(manager->proc_id == task->current_proc.id);
  manager->reset();
  TensorAccessorR<DATATYPE, 2> accWeight(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, manager);
  TensorAccessorR<DATATYPE, 2> accOutputGrad(
      regions[1], task->regions[1], FID_DATA, ctx, runtime, manager);
  TensorAccessorR<DATATYPE, 2> accOutput(
      regions[2], task->regions[2], FID_DATA, ctx, runtime, manager);
  TensorAccessorR<DATATYPE, 2> accInput(
      regions[3], task->regions[3], FID_DATA, ctx, runtime, manager);
  TensorAccessorW<DATATYPE, 2> accWeightGrad(
      regions[4], task->regions[4], FID_DATA, ctx, runtime, manager,
      true/*readOutput*/);
  TensorAccessorW<DATATYPE, 2> accInputGrad(
      regions[5], task->regions[5], FID_DATA, ctx, runtime, manager,
      !(op->resetInputGrads[0])/*readOutput*/);
  // Assert that memories are correctly mapped
  assert(accWeight.memory.kind() == Memory::GPU_FB_MEM);
  assert(accOutputGrad.memory.kind() == Memory::Z_COPY_MEM);
  assert(accOutput.memory.kind() == Memory::Z_COPY_MEM);
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
  V_ID rowLeft = accInput.rect.lo[1], rowRight = accInput.rect.hi[1];
  int inDim = accInput.rect.hi[0] - accInput.rect.lo[0] + 1;
  int outDim = accOutputGrad.rect.hi[0] - accOutputGrad.rect.lo[0] + 1;
  float alpha = 1.0f, beta = 0.0f;
  if (op->activation != AC_MODE_NONE) {
    cudnnTensorDescriptor_t outTensor;
    cudnnActivationDescriptor_t actiDesc;
    checkCUDNN(cudnnCreateActivationDescriptor(&actiDesc));
    checkCUDNN(cudnnCreateTensorDescriptor(&outTensor));
    int dims[] = {(int)(rowRight - rowLeft + 1), outDim, 1};
    int strides[] = {dims[1] * dims[2], dims[2], 1};
    checkCUDNN(cudnnSetTensorNdDescriptor(outTensor, CUDNN_DATA_FLOAT,
        3, dims, strides));
    switch (op->activation) {
      case AC_MODE_RELU:
        checkCUDNN(cudnnSetActivationDescriptor(
            actiDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
        break;
      default:
        assert(false);
    }
    reluBackward<<<GET_BLOCKS(accOutput.rect.volume()), CUDA_NUM_THREADS>>>(
        accOutputGrad.fbCache, accOutput.fbCache, accOutput.rect.volume());
    //checkCUDNN(cudnnActivationBackward(manager->dnn, actiDesc,
    //    &alpha, outTensor, accOutputGrad.fbCache,
    //    &beta, outTensor, accOutputGrad.fbCache));
    checkCUDA(cudaDeviceSynchronize());
    checkCUDNN(cudnnDestroyTensorDescriptor(outTensor));
    checkCUDNN(cudnnDestroyActivationDescriptor(actiDesc));
  }
  // Compute weight_grad
  // Note that we use alpha = 1.0 to accumulate weight gradients
  checkCUDA(cublasSgemm(manager->blas, CUBLAS_OP_N, CUBLAS_OP_T,
                        inDim, outDim, rowRight - rowLeft + 1,
                        &alpha, accInput.fbCache, inDim,
                        accOutputGrad.fbCache, outDim,
                        &alpha, accWeightGrad.ptr, inDim));
  // Compute input_grad
  // Note that we use alpha = 1.0 to accumulate input gradients
  checkCUDA(cublasSgemm(manager->blas, CUBLAS_OP_N, CUBLAS_OP_N,
                        inDim, rowRight - rowLeft + 1, outDim,
                        &alpha, accWeight.ptr, inDim,
                        accOutputGrad.fbCache, outDim,
                        &alpha, accInputGrad.fbCache, inDim));
  checkCUDA(cudaMemcpy(accInputGrad.ptr, accInputGrad.fbCache,
                       accInputGrad.rect.volume() * sizeof(DATATYPE),
                       cudaMemcpyDeviceToHost));
  //checkCUDA(cudaMemcpy((DATATYPE*)accOutputGrad.ptr, accOutputGrad.fbCache,
  //                     accOutputGrad.rect.volume() * sizeof(DATATYPE),
  //                     cudaMemcpyDeviceToHost));
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      printf("[Linear:backward] OutputGrad[%d][%d]: %.4lf\n", i, j, accOutputGrad.ptr[i * outDim + j]);
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      printf("[Linear:backward] InputGrad[%d][%d]: %.4lf\n", i, j, accInputGrad.ptr[i * inDim + j]);
  checkCUDA(cudaDeviceSynchronize());
}
