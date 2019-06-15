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
#include "cuda_helper.h"

NcclInfo nccl_task_impl(const Task *task,
                         const std::vector<PhysicalRegion>& regions,
                         Context ctx,
                         Runtime *runtime)
{
  const Graph* graph = (Graph*) task->args;
  NcclInfo info;
  int devs[1] = {0};
  ncclComm_t comms[4];
  int numRanks = graph->numParts / graph->numMachines;
  //for (int i = 0; i < numRanks; i++)
  //  devs[i] = i;
	//volatile int debug = 0;
	//while(!debug);
  printf("processor(%d) numRanks(%d)\n", task->current_proc.id, numRanks);
  NCCLCheck(ncclCommInitAll(comms, 1, devs));
  printf("After processor(%d) numRanks(%d)\n", task->current_proc.id, numRanks);
  checkCUDA(cudaDeviceSynchronize());
  return info;
}

NcclTask::NcclTask(const Graph& graph,
                   const IndexSpaceT<1>& domain,
                   const ArgumentMap& arg_map)
  : IndexLauncher(NCCL_TASK_ID, domain,
                  TaskArgument(&graph, sizeof(Graph)), arg_map)
{
  // no regions
}
