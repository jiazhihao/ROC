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

LegionRuntime::Logger::Category log_activaion("activation");

Tensor Model::relu(const Tensor& _input)
{
  Activation* op = new Activation(*this, _input, AC_MODE_RELU);
  layers.push_back(op);
  return op->outputs[0];
}

Tensor Model::sigmoid(const Tensor& _input)
{
  Activation* op = new Activation(*this, _input, AC_MODE_SIGMOID);
  layers.push_back(op);
  return op->outputs[0];
}

Activation::Activation(const Model& model,
                       const Tensor& _input,
                       ActiMode _actiMode)
: GnnOp(_input), actiMode(_actiMode)
{
  assert(_input.numDim == 2);
  // output
  numOutputs = 1;
  switch (_input.type) {
    case Tensor::NODE_TENSOR:
    {
      outputs[0] = model.create_node_tensor<DATATYPE>(_input.dims[0]);
      break;
    }
    case Tensor::EDGE_TENSOR:
    {
      outputs[0] = model.create_edge_tensor<DATATYPE>(_input.dims[0]);
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

void Activation::init(const Model& model)
{}

void Activation::forward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  IndexLauncher launcher(ACTIVATION_FWD_TASK_ID, model.taskIS,
      TaskArgument(this, sizeof(Activation)), model.taskArgs);
  // regions[0]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Activation::backward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  IndexLauncher launcher(ACTIVATION_BWD_TASK_ID, model.taskIS,
      TaskArgument(this, sizeof(Activation)), model.taskArgs);
  // regions[0]: output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part_grad, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: output
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: input
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  // regions[3]: input_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection*/,
                        resetInputGrads[0] ? WRITE_ONLY : READ_WRITE,
                        EXCLUSIVE, inputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(3, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}
