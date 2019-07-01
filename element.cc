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

LegionRuntime::Logger::Category log_element("element");

Tensor Model::add(const Tensor& _input1, const Tensor& _input2)
{
  Element* op = new Element(*this, _input1, _input2, EW_TYPE_ADD);
  layers.push_back(op);
  return op->outputs[0];
}

Element::Element(const Model& model,
                 const Tensor& _input1,
                 const Tensor& _input2,
                 ElementType _elementType)
: GnnOp(_input1, _input2), elementType(_elementType)
{
  assert(_input1.numDim == _input2.numDim);
  for (int i = 0; i < _input1.numDim; i++)
    assert(_input1.dims[i] == _input2.dims[i]);
  // output
  numOutputs = 1;
  switch (_input1.type) {
    case Tensor::NODE_TENSOR:
    {
      outputs[0] = model.create_node_tensor<DATATYPE>(_input1.dims[0]);
      break;
    }
    case Tensor::EDGE_TENSOR:
    {
      outputs[0] = model.create_edge_tensor<DATATYPE>(_input1.dims[0]);
      break;
    }
    default:
    {
      assert(false);
    }
  }
}

void Element::init(const Model& model)
{}

void Element::forward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  IndexLauncher launcher(ELEMENT_FWD_TASK_ID, model.taskIS,
      TaskArgument(this, sizeof(Element)), model.taskArgs);
  // regions[0]: input0
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: input1
  launcher.add_region_requirement(
      RegionRequirement(inputs[1].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[1].region,
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

void Element::backward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  IndexLauncher launcher(ELEMENT_BWD_TASK_ID, model.taskIS,
      TaskArgument(this, sizeof(Element)), model.taskArgs);
  // regions[0]: output_grad
  launcher.add_region_requirement(
      RegionRequirement(outputs[0].part_grad, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, outputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: input0_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection*/,
                        resetInputGrads[0] ? WRITE_ONLY : READ_WRITE,
                        EXCLUSIVE, inputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: input1_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[1].part_grad, 0/*projection*/,
                        resetInputGrads[1] ? WRITE_ONLY : READ_WRITE,
                        EXCLUSIVE, inputs[1].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

