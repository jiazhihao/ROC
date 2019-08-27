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
#include "types.h"
#include "optimizer.h"
#include "initializer.h"
#include "resourcemanager.h"

using namespace Legion;

#define MAX_FILE_LEN 64
#define MAX_NUM_PARTS 64
#define MAX_NUM_MACHINES 64
#define MAX_NUM_INPUTS 8
#define MAX_NUM_OUTPUTS 8
#define MAX_NUM_DIM 4
#define FILE_HEADER_SIZE (sizeof(E_ID) + sizeof(V_ID))
#define MAP_TO_FB_MEMORY 0xABCD0000
#define MAP_TO_ZC_MEMORY 0xABCE0000

enum {
  TOP_LEVEL_TASK_ID,
  NCCL_TASK_ID,
  LOAD_GRAPH_TASK_ID,
  LOAD_FEATS_TASK_ID,
  LOAD_LABEL_TASK_ID,
  LOAD_MASK_TASK_ID,
  INIT_TASK_ID,
  SCATTERGATHER_FWD_TASK_ID,
  SCATTERGATHER_BWD_TASK_ID,
  SCATTERGATHER_UPD_TASK_ID,
  INDEGREENORM_FWD_TASK_ID,
  INDEGREENORM_BWD_TASK_ID,
  INDEGREENORM_UPD_TASK_ID,
  LINEAR_FWD_TASK_ID,
  LINEAR_BWD_TASK_ID,
  LINEAR_UPD_TASK_ID,
  ACTIVATION_FWD_TASK_ID,
  ACTIVATION_BWD_TASK_ID,
  DROPOUT_INIT_TASK_ID,
  DROPOUT_FWD_TASK_ID,
  DROPOUT_BWD_TASK_ID,
  DROPOUT_UPD_TASK_ID,
  DROPOUT_INFER_TASK_ID,
  ELEMENT_FWD_TASK_ID,
  ELEMENT_BWD_TASK_ID,
  SOFTMAX_FWD_TASK_ID,
  SOFTMAX_BWD_TASK_ID,
  SOFTMAX_UPD_TASK_ID,
  // Optimizer
  ADAM_UPD_TASK_ID,
  // Initializer
  GLOROT_INIT_TASK_ID,
  ZEROS_INIT_TASK_ID,
  // Internal
  ZERO_GRAD_TASK_ID,
};

enum AggrType {
  AGGR_AVG,
  AGGR_MAX,
  AGGR_MIN,
  AGGR_SUM,
};

enum ActiMode {
  AC_MODE_NONE,
  AC_MODE_RELU,
  AC_MODE_SIGMOID,
};

enum ElementType {
  EW_TYPE_ADD,
  EW_TYPE_MUL,
};

enum ModelMode {
  MD_MODE_TRAIN,
  MD_MODE_INFER,
};

enum MaskType {
  MASK_TRAIN,
  MASK_VAL,
  MASK_TEST,
  MASK_NONE,
};

struct Config
{
  int numGPUs, numMachines, totalGPUs, numEpochs;
  bool verbose;
  float learning_rate, weight_decay, dropout_rate;
  std::string filename;
  std::vector<int> layers;
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

struct Tensor
{
  enum Type {
    NODE_TENSOR = 1,
    EDGE_TENSOR = 2,
    GRAPH_TENSOR = 3,
    WEIGHT_TENSOR = 4,
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
  Model(const Graph& _graph, Context _ctx, Runtime* _runtime);
  Tensor add(const Tensor& _input1, const Tensor& _input2);
  Tensor dropout(const Tensor& _input, float rate, int seed = 0);
  Tensor scatter_gather(const Tensor& _input);
  void softmax_cross_entropy(const Tensor& logits, const Tensor& labels, const Tensor& mask);
  Tensor indegree_norm(const Tensor& _input);
  Tensor linear(const Tensor& _input, int outDim,
                ActiMode activation, Initializer* initializer = NULL);
  Tensor relu(const Tensor& _input);
  Tensor sigmoid(const Tensor& _input);
  template<typename DT>
  Tensor create_node_tensor(int _numHidden) const;
  template<typename DT>
  Tensor create_edge_tensor(int _numHidden) const;
  Tensor create_weight_tensor(int _inDim, int _outDim,
                              Initializer* initializer) const;
  void load_features(const Tensor& input, const std::string& filename);
  void load_labels(const Tensor& label, const std::string& filename);
  void load_train_mask(const Tensor& mask, const std::string& filename);
  bool init(const Config& config);
  void train_mode(void);
  void infer_mode(void);
  void forward(void);
  void backward(void);
  void update(void);
  void zero_gradients(void);
public:
  ModelMode mode;
  Graph myGraph;
  Context ctx;
  Runtime* runtime;
  IndexSpaceT<1> taskIS;
  ArgumentMap taskArgs;
  Optimizer* optimizer;
  std::vector<Tensor> parameters;
private:
  //int numParts;
  std::vector<GnnOp*> layers;
};

class GnnOp
{
public:
  GnnOp(const Tensor& input);
  GnnOp(const Tensor& input1, const Tensor& input2);
  GnnOp(const Tensor& input1, const Tensor& input2, const Tensor& input3);
  virtual void init(const Model& model) = 0;
  virtual void forward(const Model& model) = 0;
  virtual void backward(const Model& model) = 0;
  //virtual void update(const Model& model) = 0;
public:
  int numInputs, numOutputs;
  ModelMode mode;
  Tensor inputs[MAX_NUM_INPUTS], outputs[MAX_NUM_OUTPUTS];
  bool trainableInputs[MAX_NUM_INPUTS];
  bool resetInputGrads[MAX_NUM_INPUTS];
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
  //virtual void update(const Model& model);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  //static void update_task(const Task *task,
  //                        const std::vector<PhysicalRegion> &regions,
  //                        Context ctx, Runtime *runtime);
};

class InDegreeNorm : public GnnOp
{
public:
  InDegreeNorm(const Model& model, const Tensor& input);
  virtual void init(const Model& model);
  virtual void forward(const Model& model);
  virtual void backward(const Model& model);
  //virtual void update(const Model& model);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  //static void update_task(const Task *task,
  //                        const std::vector<PhysicalRegion> &regions,
  //                        Context ctx, Runtime *runtime);
};

class Linear : public GnnOp
{
public:
  Linear(const Model& model, const Tensor& input,
         int outDim, ActiMode _activaiton,
         Initializer* initializer);
  void init(const Model& model);
  void forward(const Model& model);
  void backward(const Model& model);
  //void update(const Model& model);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  //static void update_task(const Task *task,
  //                        const std::vector<PhysicalRegion> &regions,
  //                        Context ctx, Runtime *runtime);
public:
  ActiMode activation;
  Tensor weight;
};

class Activation : public GnnOp
{
public:
  Activation(const Model& model, const Tensor& input,
             ActiMode _actiMode);
  void init(const Model& model);
  void forward(const Model& model);
  void backward(const Model& model);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
public:
  ActiMode actiMode;
};

class Element : public GnnOp
{
public:
  Element(const Model& model, const Tensor& input0,
          const Tensor& input1, ElementType _elementType);
  void init(const Model& model);
  void forward(const Model& model);
  void backward(const Model& model);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
public:
  ElementType elementType;
};

class Dropout : public GnnOp
{
public:
  Dropout(const Model& model, const Tensor& input, float rate, int seed);
  virtual void init(const Model& model);
  virtual void forward(const Model& model);
  virtual void backward(const Model& model);
  //virtual void update(const Model& model);
  static void init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
  static void forward_task(const Task *task,
                           const std::vector<PhysicalRegion> &regions,
                           Context ctx, Runtime *runtime);
  static void backward_task(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx, Runtime *runtime);
  static void infer_task(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx, Runtime *runtime);
public:
  float rate;
  int seed;
};

class SoftmaxCrossEntropy : public GnnOp
{
public:
  SoftmaxCrossEntropy(const Model& model,
                      const Tensor& logits,
                      const Tensor& labels,
                      const Tensor& mask);
  virtual void init(const Model& model);
  virtual void forward(const Model& model);
  virtual void backward(const Model& model);
  //virtual void update(const Model& model);
  static void backward_task(const Task *task,
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

class LoadFeaturesTask : public TaskLauncher
{
public:
  LoadFeaturesTask(const Model& model,
                   const Tensor& input,
                   const std::string& filename);
};

class LoadLabelsTask : public TaskLauncher
{
public:
  LoadLabelsTask(const Model& model,
                 const Tensor& input,
                 const std::string& filename);
};

class LoadMaskTask : public TaskLauncher
{
public:
  LoadMaskTask(const Model& model,
               const Tensor& input,
               const std::string& filename);
};

class LoadGraphTask : public IndexLauncher
{
public:
  LoadGraphTask(const Model& model,
                const std::string& filename);
};

class InitTask : public IndexLauncher
{
public:
  InitTask(const Model& model);
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

void load_features_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);

void load_labels_impl(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime);

void load_mask_impl(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime);

void zero_grad_task_impl(const Task* task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx, Runtime* runtime);

void gnn_fwd_task_impl(const Task *task,
                       const std::vector<PhysicalRegion> &regions,
                       Context ctx, Runtime *runtime);

ResourceManager* init_task_impl(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime *runtime);
#endif //_GNN_H_
