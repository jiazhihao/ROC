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

LegionRuntime::Logger::Category log_softmax("softmax");

void Model::softmax_cross_entropy(const Tensor& _logit,
                                  const Tensor& _label,
                                  const Tensor& _mask)
{
  SoftmaxCrossEntropy* op = new SoftmaxCrossEntropy(
      *this, _logit, _label, _mask);
  layers.push_back(op);
}

SoftmaxCrossEntropy::SoftmaxCrossEntropy(const Model& model,
                                         const Tensor& _logit,
                                         const Tensor& _label,
                                         const Tensor& _mask)
: GnnOp(_logit, _label, _mask)
{
  assert(_logit.numDim == 2);
  assert(_label.numDim == 2);
  assert(_label.dims[0] == _logit.dims[0]);
  assert(_label.dims[1] == _logit.dims[1]);
  numOutputs = 0;
}

void SoftmaxCrossEntropy::init(const Model& model)
{}

void SoftmaxCrossEntropy::forward(const Model& model)
{}

void SoftmaxCrossEntropy::backward(const Model& model)
{
  Context ctx = model.ctx;
  Runtime* runtime = model.runtime;
  IndexLauncher launcher(SOFTMAX_BWD_TASK_ID, model.taskIS,
                         TaskArgument(this, sizeof(SoftmaxCrossEntropy)),
                         model.taskArgs);
  // regions[0]: _logit
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[0].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: _label
  launcher.add_region_requirement(
      RegionRequirement(inputs[1].part, 0/*projection*/,
                        READ_ONLY, EXCLUSIVE, inputs[1].region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: _logit_grad
  launcher.add_region_requirement(
      RegionRequirement(inputs[0].part_grad, 0/*projection*/,
                        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  // (Optional) regions[3]: _mask
  if (inputs[2].region != LogicalRegion::NO_REGION) {
    launcher.add_region_requirement(
        RegionRequirement(inputs[2].part, 0/*projection*/,
                          READ_ONLY, EXCLUSIVE, inputs[2].region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

