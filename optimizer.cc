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

#include "optimizer.h"
#include "gnn.h"

Optimizer::Optimizer(const Model* _model)
: model(_model) {}

AdamOptimizer::AdamOptimizer(const Model* _model,
                             double _alpha, double _beta1,
                             double _beta2, double _weight_decay,
                             double _epsilon)
: Optimizer(_model), alpha(_alpha), beta1(_beta1), beta2(_beta2),
  weight_decay(_weight_decay),
  epsilon(_epsilon), alpha_t(_alpha), beta1_t(1.0f), beta2_t(1.0f)
{
  Context ctx = _model->ctx;
  Runtime* runtime = _model->runtime;
  Initializer* initializer = new ZerosInitializer();
  for (size_t i = 0; i < model->parameters.size(); i++) {
    Tensor p = model->parameters[i];
    Domain domain = runtime->get_index_space_domain(
        ctx, p.region.get_index_space());
    switch (domain.get_dim()) {
      case 0:
      {
        // Do not support 0-dim parameter
        assert(false);
        break;
      }
      case 1:
      case 2:
      case 3:
      {
        v_regions[p.region] = runtime->create_logical_region(
            ctx, p.region.get_index_space(), p.region.get_field_space());
        m_regions[p.region] = runtime->create_logical_region(
            ctx, p.region.get_index_space(), p.region.get_field_space());
        break;
      }
      default:
      {
        // Unsupported dim
        assert(false);
        break;
      }
    }
    // Zeros v_regions and m_regions
    Tensor t;
    t.numDim = p.numDim;
    for (int i = 0; i < t.numDim; i++)
      t.dims[i] = p.dims[i];
    t.region = v_regions[p.region];
    initializer->init(_model, &t);
    t.region = m_regions[p.region];
    initializer->init(_model, &t);
  }
  delete initializer;
}

void AdamOptimizer::set_weight_decay(double _weight_decay)
{
  weight_decay = _weight_decay;
}

void AdamOptimizer::next(void)
{
  beta1_t *= beta1;
  beta2_t *= beta2;
  alpha_t = alpha * sqrt(1 - beta2_t) / (1 - beta1_t);
}

void AdamOptimizer::update(const Tensor* p)
{
  Context ctx = model->ctx;
  Runtime* runtime = model->runtime;
  assert(v_regions.find(p->region) != v_regions.end());
  assert(m_regions.find(p->region) != m_regions.end());
  TaskLauncher launcher(ADAM_UPD_TASK_ID, TaskArgument(this, sizeof(AdamOptimizer)));
  // regions[0]: region_grad
  launcher.add_region_requirement(
      RegionRequirement(p->region_grad,
                        READ_ONLY, EXCLUSIVE, p->region_grad,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: region
  launcher.add_region_requirement(
      RegionRequirement(p->region,
                        READ_WRITE, EXCLUSIVE, p->region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(1, FID_DATA);
  // regions[2]: w_region
  launcher.add_region_requirement(
      RegionRequirement(v_regions[p->region],
                        READ_WRITE, EXCLUSIVE, v_regions[p->region],
                        MAP_TO_FB_MEMORY));
  launcher.add_field(2, FID_DATA);
  // regions[3]: m_region
  launcher.add_region_requirement(
      RegionRequirement(m_regions[p->region],
                        READ_WRITE, EXCLUSIVE, m_regions[p->region],
                        MAP_TO_FB_MEMORY));
  launcher.add_field(3, FID_DATA);
  runtime->execute_task(ctx, launcher);
}
