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

LegionRuntime::Logger::Category log_sg("gnn");

Tensor Model::scatter_gather(const Tensor& _input)
{
  GnnOp* op = new ScatterGather(*this, _input);
  layers.push_back(op);
  return op->outputs[0];
}


ScatterGather::ScatterGather(const Model& model,
                             const Tensor& _input)
: GnnOp(_input)
{
  // ScatterGather require a node tensor
  assert(inputs[0].type == Tensor::NODE_TENSOR);
  assert(inputs[0].numDim == 2);
  assert(inputs[0].dims[1] == model.myGraph.numNodes);
  numOutputs = 1;
  outputs[0] = model.create_node_tensor<DATATYPE>(inputs[0].dims[0]);
  //printf("inputs[0].region: ispace = %lld, fspace = %u\n",
  //       inputs[0].region.get_index_space().get_id(),
  //       inputs[0].region.get_field_space().get_id());
  //printf("outputs[0].region: ispace = %lld, fspace = %u\n",
  //       outputs[0].region.get_index_space().get_id(),
  //       outputs[0].region.get_field_space().get_id());
}

void ScatterGather::init(const Model& model)
{}

void ScatterGather::forward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  //Rect<1> taskRect = runtime->get_index_space_domain(ctx, model.taskIS);
  IndexLauncher launcher(SCATTERGATHER_FWD_TASK_ID, model.taskIS,
                         TaskArgument(this, sizeof(ScatterGather)), model.taskArgs);
  // regions[0]: row_ptrs
  launcher.add_region_requirement(
      RegionRequirement(model.myGraph.rowPtrLP, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, model.myGraph.rowPtrLR,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: col_idxs
  launcher.add_region_requirement(
      RegionRequirement(model.myGraph.colIdxLP, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, model.myGraph.colIdxLR,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].region, 0/*projection*/,
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

void ScatterGather::backward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  //Rect<1> taskRect = runtime->get_index_space_domain(ctx, model.taskIS);
  IndexLauncher launcher(SCATTERGATHER_BWD_TASK_ID, model.taskIS,
                         TaskArgument(this, sizeof(ScatterGather)), model.taskArgs);
  // regions[0]: row_ptrs
  launcher.add_region_requirement(
      RegionRequirement(model.myGraph.rowPtrLP, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, model.myGraph.rowPtrLR,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: col_idxs
  launcher.add_region_requirement(
      RegionRequirement(model.myGraph.colIdxLP, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, model.myGraph.colIdxLR,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: outputGrad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].region_grad, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  // regions[3]: inputGrad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection*/,
                        resetInputGrads[0] ? WRITE_ONLY : READ_WRITE,
                        EXCLUSIVE, inputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}


