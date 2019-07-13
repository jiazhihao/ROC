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

#ifndef _RESOURCEMANAGER_H_
#define _RESOURCEMANAGER_H_

#include "legion.h"
#include "types.h"
#include "realm/cuda/cuda_module.h" // For Realm::Cuda::GPUFBMemory
#include "nccl_helper.h"
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>

using namespace Legion;

#define MAX_NUM_CACHES 4

class ResourceManager
{
public:
  struct ReservedSpace {
    void* ptr;
    size_t size;
  };
  ResourceManager(void);
  ~ResourceManager(void);
  void reset(void);
  int assign(PhysicalRegion lr, size_t numElements);
  struct CacheSlot {
    LogicalRegion region;
    size_t volume;
    DATATYPE* ptr;
  };
  unsigned long long proc_id;
  cudnnHandle_t dnn;
  cublasHandle_t blas;
  curandGenerator_t rand;
  ncclComm_t nccl;
  // Dropout state
  void *dropoutStates;
  size_t dropoutSize;
  Realm::Cuda::GPUFBMemory* allocator;
  //V_ID numNodes;
  //E_ID numEdges;
  //int numParts;
  CacheSlot fbCache[MAX_NUM_CACHES];
  std::set<int> assigned;
  std::map<LogicalRegion, ReservedSpace> reservedSpace;
  std::map<LogicalRegion, cudnnDropoutDescriptor_t> dropoutDesc;
};

#endif
