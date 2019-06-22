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

#include "initializer.h"
#include "types.h"
#include "cuda_helper.h"
#include <curand.h>

void GlorotUniform::init_task(const Task* task,
                              const std::vector<PhysicalRegion>& regions,
                              Context ctx, Runtime* runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  TensorAccessorWO<DATATYPE, 2> accW(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, NULL);
  int inputDim = accW.rect.hi[0] - accW.rect.lo[0] + 1;
  int outputDim = accW.rect.hi[1] - accW.rect.lo[1] + 1;
  //float scale = *((float*) task->args);
  float scale = sqrt(6.0 / (inputDim + outputDim));
  printf("scale = %.4lf\n", scale);
  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  // TODO: change to random seed before releasing
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  checkCUDA(curandGenerateUniform(gen, accW.ptr, accW.rect.volume()));
  scale_kernel<<<GET_BLOCKS(accW.rect.volume()), CUDA_NUM_THREADS>>>(
      accW.ptr, accW.rect.volume(), -scale, scale);
  checkCUDA(cudaDeviceSynchronize());
  curandDestroyGenerator(gen);
}

void ZerosInitializer::init_task(const Task* task,
                                 const std::vector<PhysicalRegion>& regions,
                                 Context ctx, Runtime* runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  TensorAccessorWO<DATATYPE, 2> accW(
      regions[0], task->regions[0], FID_DATA, ctx, runtime, NULL);
  assign_kernel<<<GET_BLOCKS(accW.rect.volume()), CUDA_NUM_THREADS>>>(
      accW.ptr, accW.rect.volume(), 0);
}
