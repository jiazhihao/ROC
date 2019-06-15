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

#include "gnn.h"

LegionRuntime::Logger::Category log_norm("gnn");

Tensor Model::indegree_norm(const Tensor& _input)
{
  GnnOp* op = new InDegreeNorm(*this, _input);
  layers.push_back(op);
  return op->outputs[0];
}

InDegreeNorm::InDegreeNorm(const Model& model,
                           const Tensor& _input)
: GnnOp(_input)
{
  // InDegreeeNorm requires a node tensor
  assert(_input.type == Tensor::NODE_TENSOR);
  assert(_input.numDim == 2);
  assert(_input.dims[1] == model.myGraph.numNodes);
  numOutputs = 1;
  outputs[0] = model.create_node_tensor(_input.dims[0]);
}

void InDegreeNorm::init(const Model& model)
{}

void InDegreeNorm::forward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  //Rect<1> taskRect = runtime->get_index_space_domain(ctx, model.taskIS);
  IndexLauncher launcher(INDEGREENORM_FWD_TASK_ID, model.taskIS,
                         TaskArgument(this, sizeof(InDegreeNorm)), model.taskArgs);
  // regions[0]: in_row_ptrs
  launcher.add_region_requirement(
      RegionRequirement(model.myGraph.rowPtrLP, 0/*projection*/,
                       READ_ONLY, EXCLUSIVE, model.myGraph.rowPtrLR,
                       MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: in_col_idxs
  launcher.add_region_requirement(
      RegionRequirement(model.myGraph.colIdxLP, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, model.myGraph.colIdxLR,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  // regions[3]: ouptut
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void InDegreeNorm::backward(const Model& model)
{
  // TODO
  assert(false);
}

void InDegreeNorm::update(const Model& model)
{
}
