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

GnnFwdTask::GnnFwdTask(const Graph& graph,
                       const GnnOp& gnnOp,
                       const IndexSpaceT<1> &domain,
                       const ArgumentMap &arg_map)
  : IndexLauncher(GNN_FWD_TASK_ID, domain,
                  TaskArgument(&gnnOp, sizeof(GnnOp)), arg_map)
{
  // regions[0]: row_ptrs
  {
    RegionRequirement rr(graph.rowPtrLP, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.rowPtrLR,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: col_idxs
  {
    RegionRequirement rr(graph.colIdxLP, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, graph.colIdxLR,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[2]: input
  {
    RegionRequirement rr(gnnOp.input.nodeFwdLR, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, gnnOp.input.nodeFwdLR,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[3]: output
  {
    RegionRequirement rr(gnnOp.output.nodeFwdLP, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, gnnOp.output.nodeFwdLR,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

