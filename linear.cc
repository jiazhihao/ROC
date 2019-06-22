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

LegionRuntime::Logger::Category log_linear("gnn");

Tensor Model::linear(const Tensor& _input, int _outDim,
                     ActiMode _activation,
                     Initializer* initializer)
{
  GnnOp* op = new Linear(*this, _input, _outDim, _activation,
                         initializer);
  layers.push_back(op);
  parameters.push_back(op->weight);
  return op->outputs[0];
}

Linear::Linear(const Model& model,
               const Tensor& _input,
               int outDim,
               ActiMode _activation,
               Initializer* initializer)
: GnnOp(_input), activation(_activation)
{
  assert(_input.numDim == 2);
  assert(_input.dims[1] == model.myGraph.numNodes);
  weight = model.create_weight_tensor(_input.dims[0], outDim, initializer);
#ifdef DEADCODE
  // initialize weight tensor
  weight.type = Tensor::WEIGHT_TENSOR;
  weight.numDim = 1;
  weight.dims[0] = _input.dims[0] * outDim;
  Rect<1> rectWeight(Point<1>(0), Point<1>(weight.dims[0]-1));
  Rect<1> rectWeightGrad(Point<1>(0), Point<1>(weight.dims[0]*model.myGraph.numParts-1));
  IndexSpaceT<1> weightIS = runtime->create_index_space(ctx, rectWeight);
  IndexSpaceT<1> weightGradIS = runtime->create_index_space(ctx, rectWeightGrad);
  runtime->attach_name(weightIS, "weight_index_space");
  runtime->attach_name(weightGradIS, "weight_grad_index_space");
  {
    FieldSpace fs = runtime->create_field_space(ctx);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(DATATYPE), FID_DATA);
    weight.region = runtime->create_logical_region(ctx, weightIS, fs);
    weight.region_grad = runtime->create_logical_region(ctx, weightGradIS, fs);
    weight.part = LogicalPartition::NO_PART;
    IndexPartition weightGradIP =
        runtime->create_equal_partition(ctx, weightGradIS, model.taskIS);
    weight.part_grad =
        runtime->get_logical_partition(ctx, weight.region_grad, weightGradIP);
  }
#endif
  // output
  numOutputs = 1;
  switch (_input.type) {
    case Tensor::NODE_TENSOR:
    {
      outputs[0] = model.create_node_tensor(outDim);
      break;
    }
    case Tensor::EDGE_TENSOR:
    {
      outputs[0] = model.create_edge_tensor(outDim);
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

void Linear::init(const Model& model)
{}

void Linear::forward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  IndexLauncher launcher(LINEAR_FWD_TASK_ID, model.taskIS,
                         TaskArgument(this, sizeof(Linear)), model.taskArgs);
  // regions[0]: weight
  launcher.add_region_requirement(
      RegionRequirement(weight.region, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, weight.region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Linear::backward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  IndexLauncher launcher(LINEAR_BWD_TASK_ID, model.taskIS,
                         TaskArgument(this, sizeof(Linear)), model.taskArgs);
  // regions[0]: weight
  launcher.add_region_requirement(
      RegionRequirement(weight.region, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, weight.region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part_grad, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  // regions[3]: weight_grad
  launcher.add_region_requirement(
      RegionRequirement(weight.part_grad, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, weight.region_grad,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(3, FID_DATA);
  // regions[4]: input_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(4, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Linear::update(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  TaskLauncher launcher(LINEAR_UPD_TASK_ID, TaskArgument(this, sizeof(Linear)));
  // regions[0]: weight_grad
  launcher.add_region_requirement(
      RegionRequirement(weight.region_grad,
                        READ_ONLY, EXCLUSIVE, weight.region_grad,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: weight
  launcher.add_region_requirement(
      RegionRequirement(weight.region,
                        READ_WRITE, EXCLUSIVE, weight.region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(1, FID_DATA);
  runtime->execute_task(ctx, launcher);
}


