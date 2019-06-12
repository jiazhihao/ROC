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

#include "legion.h"
#include <cuda_runtime.h>

using namespace Legion;

template<typename FT, int N, typename T = coord_t> using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = FieldAccessor<WRITE_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;

#define MAP_TO_FB_MEMORY 0xABCD0000
#define MAP_TO_ZC_MEMORY 0xABCE0000

typedef uint32_t V_ID;
typedef uint64_t E_ID;
typedef float DATATYPE;

struct NodeStruct {
  E_ID index;
};

struct EdgeStruct {
  V_ID src, dst;
};

enum {
  TOP_LEVEL_TASK_ID,
  LOAD_TASK_ID,
  UPD_NODE_TASK_ID,
};

enum FieldIDs {
  FID_DATA,
};

struct Graph
{
  Graph(const std::string& _filename, Context _ctx, Runtime* _runtime, int _numParts);
  LogicalRegion row_ptr_lr, raw_row_lr, col_idx_lr, raw_col_lr;
  LogicalPartition row_ptr_lp, raw_row_lp, col_idx_lp, raw_col_lp;
  V_ID numNodes;
  E_ID numEdges;
  int numParts;
};

struct Tensor
{
  LogicalRegion fwd_lr, bwd_lr;
  LogicalPartition fwd_lp, bwd_lp;
};

class Model
{
public:
  Model(const Graph& _graph, Context _ctx, Runtime* _runtime, int _numHidden);
  Tensor add_layer(const Tensor& _input, int _numHidden);
  Tensor create_tensor(int _numHidden);
private:
  Graph myGraph;
  Context ctx;
  Runtime* runtime;
  Tensor input;
  int numParts;
};

class EdgeUpdate : public IndexLauncher
{
public:
  EdgeUpdate(const Model &model);
  virtual void update();
};

class NodeUpdate : public IndexLauncher
{
public:
  NodeUpdate(const Model &model);
  virtual void update();
};

class GraphUpdate : public IndexLauncher
{
public:
  GraphUpdate(const Model &model);
  virtual void update();
};

template<typename T>
void alloc_fs(Context ctx, Runtime *runtime, FieldSpace fs)
{
  FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
  allocator.allocate_field(sizeof(T), FID_DATA);
}

#endif //_GNN_H_
