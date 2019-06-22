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

#ifndef _INITIALIZER_H_
#define _INITIALIZER_H_

#include "legion.h"

using namespace Legion;

class Model;
class Tensor;

class Initializer
{
public:
  Initializer(void);
  virtual ~Initializer(void);
  virtual void init(const Model* model, const Tensor* tensor) = 0;
};

class GlorotUniform : public Initializer
{
public:
  GlorotUniform(void);
  ~GlorotUniform(void);
  void init(const Model* model, const Tensor* tensor);
  static void init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
};

class ZerosInitializer : public Initializer
{
public:
  ZerosInitializer(void);
  ~ZerosInitializer(void);
  void init(const Model* model, const Tensor* tensor);
  static void init_task(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime *runtime);
};

#endif
