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

#include "gnn_mapper.h"
#include "gnn.h"

GnnMapper::GnnMapper(Machine m, Runtime *rt, Processor p)
  : DefaultMapper(rt->get_mapper_runtime(), m, p)
{
  numNodes = remote_gpus.size();
  Machine::ProcessorQuery proc_query(machine);
  for (Machine::ProcessorQuery::iterator it = proc_query.begin();
       it != proc_query.end(); it++)
  {
    AddressSpace node = it->address_space();
    std::map<unsigned, std::vector<Processor>* >::const_iterator finder =
      allGPUs.find(node);
    if (finder == allGPUs.end())
      allGPUs[node] = new std::vector<Processor>;
    finder = allCPUs.find(node);
    if (finder == allCPUs.end())
      allCPUs[node] = new std::vector<Processor>;
    switch (it->kind())
    {
      case Processor::TOC_PROC:
      {
        allGPUs[node]->push_back(*it);
        Machine::MemoryQuery fb_query(machine);
        fb_query.only_kind(Memory::GPU_FB_MEM);
        fb_query.best_affinity_to(*it);
        // Assume each GPU has one device memory
        assert(fb_query.count() == 1);
        memFBs[*it] = *(fb_query.begin());
        Machine::MemoryQuery zc_query(machine);
        zc_query.only_kind(Memory::Z_COPY_MEM);
        zc_query.has_affinity_to(*it);
        assert(zc_query.count() == 1);
        memZCs[*it] = *(zc_query.begin());
        break;
      }
      case Processor::LOC_PROC:
      {
        allCPUs[node]->push_back(*it);
        //Machine::MemoryQuery sys_query(machine);
        //sys_query.only_kind(Memory::SYSTEM_MEM);
        //sys_query.has_affinity_to(*it);
        //memSys[*it] = *(sys_query.begin());
        Machine::MemoryQuery zc_query(machine);
        zc_query.only_kind(Memory::Z_COPY_MEM);
        zc_query.has_affinity_to(*it);
        memZCs[*it] = *(zc_query.begin());
        break;
      }
      default:
        break;
    }
  }
}

GnnMapper::~GnnMapper()
{
  std::map<unsigned, std::vector<Processor>* >::iterator it;
  for (it = allGPUs.begin(); it != allGPUs.end(); it++)
    delete it->second;
  for (it = allCPUs.begin(); it != allCPUs.end(); it++)
    delete it->second;
}

void GnnMapper::select_task_options(const MapperContext ctx,
                                    const Task& task,
                                    TaskOptions& output)
{
  DefaultMapper::select_task_options(ctx, task, output);
}

void GnnMapper::slice_task(const MapperContext ctx,
                           const Task& task,
                           const SliceTaskInput& input,
                           SliceTaskOutput& output)
{
  if (task.task_id == UPD_NODE_TASK_ID)
  {
    if (gpuSlices.size() > 0) {
      output.slices = gpuSlices;
      return;
    }
    Rect<1> input_rect = input.domain;
    unsigned cnt = 0;
    for (PointInRectIterator<1> it(input_rect); it(); it++) {
      TaskSlice slice;
      Rect<1> task_rect(*it, *it);
      slice.domain = task_rect;
      slice.proc = allGPUs[cnt % numNodes]->at(((cnt/numNodes)*9) % local_gpus.size());
      cnt ++;
      slice.recurse = false;
      slice.stealable = false;
      gpuSlices.push_back(slice);
    }
    output.slices = gpuSlices;
  } else if (task.task_id == LOAD_TASK_ID) {
    if (cpuSlices.size() > 0) {
      output.slices = cpuSlices;
      return;
    }
    Rect<1> input_rect = input.domain;
    unsigned cnt = 0;
    for (PointInRectIterator<1> it(input_rect); it(); it++) {
      TaskSlice slice;
      Rect<1> task_rect(*it, *it);
      slice.domain = task_rect;
      slice.proc = allCPUs[cnt % numNodes]->at(((cnt/numNodes)*9)%local_cpus.size());
      cnt ++;
      slice.recurse = false;
      slice.stealable = false;
      cpuSlices.push_back(slice);
    }
    output.slices = cpuSlices;
  } else {
    DefaultMapper::slice_task(ctx, task, input, output);
  }
}

Memory GnnMapper::default_policy_select_target_memory(MapperContext ctx,
                                                      Processor target_proc,
                                                      const RegionRequirement &req)
{
  if (req.tag == MAP_TO_FB_MEMORY) {
    assert(memFBs.find(target_proc) != memFBs.end());
    return memFBs[target_proc];
  } else if (req.tag == MAP_TO_ZC_MEMORY) {
    assert(memZCs.find(target_proc) != memZCs.end());
    return memZCs[target_proc];
  } else {
    assert(req.tag == 0);
    assert(memZCs.find(target_proc) != memZCs.end());
    return memZCs[target_proc];
  }
}

