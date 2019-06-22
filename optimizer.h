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

#ifndef _PARAMETER_H_
#define _PARAMETER_H_

#include "legion.h"

using namespace Legion;
class Model;
class Tensor;

class Optimizer
{
public:
  Optimizer(const Model* _model);
  virtual void next(void) = 0;
  virtual void update(const Tensor* p) = 0;
  const Model* model;
};

class AdamOptimizer : public Optimizer
{
public:
  AdamOptimizer(const Model* _model, 
                double _alpha, double _beta1,
                double _beta2, double _epsilon);
  void next(void);
  void update(const Tensor* p);
  static void update_task(const Task* task,
                          const std::vector<PhysicalRegion>& regions,
                          Context ctx, Runtime* runtime);
  double alpha, beta1, beta2, epsilon;
  double alpha_t, beta1_t, beta2_t;
  std::map<LogicalRegion, LogicalRegion> v_regions, m_regions;
};

#endif
