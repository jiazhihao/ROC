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

#ifndef _GNN_H_
#define _GNN_H_

#include "nccl_helper.h"
#include "legion.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace Legion;

template<typename FT, int N, typename T = coord_t> using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = FieldAccessor<WRITE_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;

typedef uint32_t V_ID;
typedef uint64_t E_ID;
typedef float DATATYPE;

#define MAX_FILE_LEN 64
#define MAX_NUM_PARTS 64
#define MAX_NUM_MACHINES 64
#define MAX_NUM_CACHES 4
#define MAX_NUM_INPUTS 8
#define MAX_NUM_OUTPUTS 8
#define MAX_NUM_DIM 4
#define FILE_HEADER_SIZE (sizeof(E_ID) + sizeof(V_ID))
#define MAP_TO_FB_MEMORY 0xABCD0000
#define MAP_TO_ZC_MEMORY 0xABCE0000

struct NodeStruct {
  E_ID index;
};

struct EdgeStruct {
  V_ID src, dst;
};

enum {
  TOP_LEVEL_TASK_ID,
  NCCL_TASK_ID,
  LOAD_TASK_ID,
  INIT_TASK_ID,
  SCATTERGATHER_FWD_TASK_ID,
  SCATTERGATHER_BWD_TASK_ID,
  SCATTERGATHER_UPD_TASK_ID,
  INDEGREENORM_FWD_TASK_ID,
  INDEGREENORM_BWD_TASK_ID,
  INDEGREENORM_UPD_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

enum AggrType {
  AGGR_AVG,
  AGGR_MAX,
  AGGR_MIN,
  AGGR_SUM,
};

struct Config
{
  int numGPUs, numMachines, totalGPUs;
  bool verbose;
  std::string filename;
};

struct NcclInfo
{
  ncclComm_t comms[MAX_NUM_PARTS];
};

struct Graph
{
  Graph(Context _ctx, Runtime* _runtime, const Config& config);
  V_ID numNodes;
  E_ID numEdges;
  int numParts, numMachines;
  int maxHidden;
  //ncclComm_t nccl[MAX_NUM_PARTS];
  LogicalRegion rowPtrLR, rawRowLR, colIdxLR, rawColLR;
  LogicalPartition rowPtrLP, rawRowLP, colIdxLP, rawColLP;
};

class ResourceManager
{
public:
  ResourceManager(void);
  ~ResourceManager(void);
  int assign(LogicalRegion lr, size_t numElements, std::set<int>& assigned);
  struct CacheSlot {
    LogicalRegion region;
    size_t volume;
    DATATYPE* ptr;
  };
  unsigned long long proc_id;
  cublasHandle_t blas;
  ncclComm_t nccl;
  //V_ID numNodes;
  //E_ID numEdges;
  //int numParts;
  CacheSlot fbCache[MAX_NUM_CACHES];
};

struct Tensor
{
  enum Type {
    NODE_TENSOR = 1,
    EDGE_TENSOR = 2,
    INVALID_TENSOR = 9,
  };
  Tensor(void): type(INVALID_TENSOR) {
    region = LogicalRegion::NO_REGION;
    region_grad = LogicalRegion::NO_REGION;
    part = LogicalPartition::NO_PART;
    part_grad = LogicalPartition::NO_PART;
  };
  Tensor(Type _type): type(_type) {
    region = LogicalRegion::NO_REGION;
    region_grad = LogicalRegion::NO_REGION;
    part = LogicalPartition::NO_PART;
    part_grad = LogicalPartition::NO_PART;
  };
  Type type;
  int numDim;
  E_ID dims[MAX_NUM_DIM];
  LogicalRegion region, region_grad;
  LogicalPartition part, part_grad;
};

class GnnOp;

class Model
{
public:
  Model(const Graph& _graph, Context _ctx, Runtime* _runtime, int _numHidden);
  Tensor scatter_gather(const Tensor& _input);
  Tensor indegree_norm(const Tensor& _input);
  Tensor create_node_tensor(int _numHidden) const;
  bool init(const Config& config);
  void forward(void);
  Tensor get_input_tensor(void) const;
  Graph myGraph;
  Context ctx;
  Runtime* runtime;
  IndexSpaceT<1> taskIS;
  ArgumentMap taskArgs;
private:
  Tensor input;
  int numParts;
  std::vector<GnnOp*> layers;
};

class GnnOp
{
public:
  GnnOp(const Tensor& input);
  virtual void init(const Model& model) = 0;
  virtual void forward(const Model& model) = 0;
  virtual void backward(const Model& model) = 0;
  virtual void update(const Model& model) = 0;
public:
  int numInputs, numOutputs;
  Tensor inputs[MAX_NUM_INPUTS], outputs[MAX_NUM_OUTPUTS];
  IndexLauncher *fwdLauncher, *bwdLauncher, *gradLauncher;
};

// Perform sum aggregation
class ScatterGather : public GnnOp
{
public:
  ScatterGather(const Model& model, const Tensor& input);
  virtual void init(const Model& model);
  virtual void forward(const Model& model);
  virtual void backward(const Model& model);
  virtual void update(const Model& model);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void update_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime);
};

class InDegreeNorm : public GnnOp
{
public:
  InDegreeNorm(const Model& model, const Tensor& input);
  virtual void init(const Model& model);
  virtual void forward(const Model& model);
  virtual void backward(const Model& model);
  virtual void update(const Model& model);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void update_task(const Task *task,
                          const std::vector<PhysicalRegion> &regions,
                          Context ctx, Runtime *runtime);
};

class NcclTask : public IndexLauncher
{
public:
  NcclTask(const Graph& graph,
           const IndexSpaceT<1>& domain,
           const ArgumentMap& arg_map);
};

class LoadTask : public IndexLauncher
{
public:
  LoadTask(const Graph& graph,
           const IndexSpaceT<1>& domain,
           const ArgumentMap& arg_map,
           const std::string& fn);
};

class InitTask : public IndexLauncher
{
public:
  InitTask(const Graph& graph,
           const Tensor& input,
           const IndexSpaceT<1>& domain,
           const ArgumentMap& arg_map);
};

template<typename T>
void alloc_fs(Context ctx, Runtime *runtime, FieldSpace fs)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(T), FID_DATA);
}

NcclInfo nccl_task_impl(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);

void load_graph_impl(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime);

void gnn_fwd_task_impl(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime);

ResourceManager* init_task_impl(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime);
#endif //_GNN_H_
