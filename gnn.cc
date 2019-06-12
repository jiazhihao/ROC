/* Copyright 2019 Stanford, UT Austin, LANL
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

#include "legion.h"
#include "gnn.h"
#include "gnn_mapper.h"

LegionRuntime::Logger::Category log_gnn("gnn");

void parse_input_args(char **argv, int argc);

void top_level_task(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    HighLevelRuntime *runtime)
{
  int numGPU = 0, numIter = 0;
  std::string filename;
  bool verbose = false;
  {
    const InputArgs &command_args = HighLevelRuntime::get_input_args();
    char **argv = command_args.argv;
    int argc = command_args.argc;
    parse_input_args(argv, argc);
    size_t numNodes = Realm::Machine::get_machine().get_address_space_count();
    numGPU = numGPU * numNodes;
  }
}

void parse_input_args(char **argv, int argc)
{
  for (int i = 1; i <argc; i++)
  {
  }
}

static void update_mappers(Machine machine, Runtime *rt,
                           const std::set<Processor> &local_procs)
{
  for (std::set<Processor>::const_iterator it = local_procs.begin();
        it != local_procs.end(); it++)
  {
    rt->replace_default_mapper(new GnnMapper(machine, rt, *it), *it);
  }
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);
  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }

  return Runtime::start(argc, argv);
}

#ifdef DEADCODE
EdgeUpdate::EdgeUpdate(const Model& model, int idx)
  : IndexLauncher(UPD_EDGE_TASK_ID, TaskArgument(NULL, 0))
{
  // regions[0]: row_ptrs
  {
    RegionRequirement rr(model.regions.row_ptr_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, model.regions.row_ptr_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: col_idx
  {
    RegionRequirement rr(model.regions.col_idx_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, model.regions.col_idx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[2]: old_nodes
  {
    RegionRequirement rr(model.regions.nodes_lr[idx-1], 0/**identity*/,
                         READ_ONLY, EXCLUSIVE, model.regions.edges_lr[idx-1],
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[3]: new_edges
  {
    RegionRequirement rr(model.regions.edges_lp[idx], 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, model.regions.edges_lr[idx],
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

NodeUpdate::NodeUpdate(const Model& model, int idx)
  : IndexLauncher(UPD_EDGE_TASK_ID, TaskArgument(NULL, 0))
{
  // regions[0]: row_ptrs
  {
    RegionRequirement rr(model.regions.row_ptr_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, model.regions.row_ptr_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: col_idx
  {
    RegionRequirement rr(model.regions.col_idx_lp, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, model.regions.col_idx_lr,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[2]: old_nodes
  {
    RegionRequirement rr(model.regions.nodes_lr[idx-1], 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, model.regions.edges_lr[idx-1],
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[3]: old_nodes
  {
    RegionRequirement rr(model.regions.nodes_lp[idx], 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, model.regions.edges_lr[idx],
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // optional regions[]
}
#endif

Model::Model(const Graph& _graph,
             Context _ctx,
             Runtime* _runtime,
             int _inputDim)
: myGraph(_graph), ctx(_ctx), runtime(_runtime),
numParts(_graph.numParts)
{
  input = create_tensor(_inputDim);
}

Tensor Model::add_layer(const Tensor& _input, int numHidden)
{
  Tensor output = create_tensor(numHidden);
  return output;
}

Tensor Model::create_tensor(int numHidden)
{
  Tensor t;
  Rect<1> rect_row_ptr = runtime->get_index_space_domain(
                             ctx, myGraph.row_ptr_lr.get_index_space());
  // Assert row_ptr_lr matches myGraph.numNodes
  assert(rect_row_ptr.volume() == myGraph.numNodes);
  Rect<1> output_rect(Point<1>(0), Point<1>(myGraph.numNodes * numHidden));
  IndexSpaceT<1> output_is =
      runtime->create_index_space(ctx, output_rect);
  runtime->attach_name(output_is, "activation_index_space");
  Rect<1> color_rect(Point<1>(0), Point<1>(numParts-1));
  // Create logical regions
  {
    FieldSpace output_fs = runtime->create_field_space(ctx);
    FieldAllocator allocator = runtime->create_field_allocator(ctx, output_fs);
    allocator.allocate_field(sizeof(DATATYPE), FID_DATA);
    t.fwd_lr = runtime->create_logical_region(ctx, output_is, output_fs);
    t.bwd_lr = runtime->create_logical_region(ctx, output_is, output_fs);
  }
  // Create logical partitions
  {
    LegionRuntime::Arrays::Rect<1> domain_rect(
        LegionRuntime::Arrays::Point<1>(0),
        LegionRuntime::Arrays::Point<1>(numParts-1));
    Domain color_domain = Domain::from_rect<1>(domain_rect);
    DomainColoring output_coloring;
    for (PointInRectIterator<1> it(color_rect); it(); it++) {
      LogicalRegion sub_lr = runtime->get_logical_subregion_by_color(
          ctx, myGraph.row_ptr_lp, DomainPoint(*it));
      Rect<1> sub_rect = runtime->get_index_space_domain(
          ctx, sub_lr.get_index_space());
      LegionRuntime::Arrays::Rect<1> output_subrect(
          LegionRuntime::Arrays::Point<1>(sub_rect.lo[0]*numHidden),
          LegionRuntime::Arrays::Point<1>((sub_rect.hi[0]+1)*numHidden-1));
      output_coloring[*it] = Domain::from_rect<1>(output_subrect);
    }
    IndexPartition output_ip = runtime->create_index_partition(
        ctx, output_is, color_domain, output_coloring, true);
    assert(runtime->is_index_partition_disjoint(ctx, output_ip));
    assert(runtime->is_index_partition_complete(ctx, output_ip));
    t.fwd_lp = runtime->get_logical_partition(ctx, t.fwd_lr, output_ip);
    t.bwd_lp = runtime->get_logical_partition(ctx, t.bwd_lr, output_ip);
  }
  return t;
}

Graph::Graph(const std::string& file_name,
             Context ctx,
             Runtime* runtime,
             int _numParts)
: numParts(_numParts)
{
  FILE* fd = fopen(file_name.c_str(), "rb");
  assert(fd != NULL);
  size_t fread_ret = fread(&numNodes, sizeof(V_ID), 1, fd);
  assert(fread_ret == 1);
  fread_ret = fread(&numEdges, sizeof(E_ID), 1, fd);
  assert(fread_ret == 1);
  log_gnn.print("Load graph: numNodes(%u) numEdges(%zu)", numNodes, numEdges);
  Rect<1> vtx_rect(Point<1>(0), Point<1>(numNodes - 1));
  IndexSpaceT<1> vtx_is =
    runtime->create_index_space(ctx, vtx_rect);
  runtime->attach_name(vtx_is, "vertices_index_space");
  Rect<1> edge_rect(Point<1>(0), Point<1>(numEdges - 1));
  IndexSpaceT<1> edge_is =
    runtime->create_index_space(ctx, edge_rect);
  runtime->attach_name(edge_is, "edges_index_space");

  FieldSpace row_ptr_fs = runtime->create_field_space(ctx);
  runtime->attach_name(row_ptr_fs, "row_ptrs(NodeStruct)");
  FieldSpace raw_row_fs = runtime->create_field_space(ctx);
  runtime->attach_name(raw_row_fs, "raw_rows(E_ID)");
  FieldSpace col_idx_fs = runtime->create_field_space(ctx);
  runtime->attach_name(col_idx_fs, "col_idxs(EdgeStruct)");
  FieldSpace raw_col_fs = runtime->create_field_space(ctx);
  runtime->attach_name(raw_col_fs, "raw_cols(V_ID)");

  // Allocate fields
  alloc_fs<NodeStruct>(ctx, runtime, row_ptr_fs);
  alloc_fs<E_ID>(ctx, runtime, raw_row_fs);
  alloc_fs<EdgeStruct>(ctx, runtime, col_idx_fs);
  alloc_fs<V_ID>(ctx, runtime, raw_col_fs);

  // Make logical regions
  row_ptr_lr = runtime->create_logical_region(ctx, vtx_is, row_ptr_fs);
  raw_row_lr = runtime->create_logical_region(ctx, vtx_is, raw_row_fs);
  col_idx_lr = runtime->create_logical_region(ctx, edge_is, col_idx_fs);
  raw_col_lr = runtime->create_logical_region(ctx, edge_is, raw_col_fs);

  E_ID* raw_rows = (E_ID*) malloc(numNodes * sizeof(E_ID));
  //double ts_start = Realm::Clock::current_time_in_microseconds();
  assert(fread(raw_rows, sizeof(E_ID), (size_t)numNodes, fd) == (size_t)numNodes);
  for (V_ID v = 1; v < numNodes; v++)
    assert(raw_rows[v] >= raw_rows[v-1]);
  assert(raw_rows[numNodes-1] == numEdges);
  fclose(fd);

  // Partition the graph
  //double ts_mid = Realm::Clock::current_time_in_microseconds();
  //printf("Loading time = %.2lfus\n", ts_mid - ts_start);
  V_ID left_bound = 0;
  E_ID edge_cnt = 0;
  E_ID edge_cap = (numEdges + numParts - 1) / numParts;
  std::vector<std::pair<V_ID, V_ID> > bounds;
  for (V_ID v = 0; v < numNodes; v++)
  {
    if (v == 0)
      edge_cnt += raw_rows[v];
    else
      edge_cnt += raw_rows[v] - raw_rows[v-1];
    if (edge_cnt > edge_cap)
    {
      bounds.push_back(std::make_pair(left_bound, v));
      edge_cnt = 0;
      left_bound = v + 1;
    }
  }
  if (edge_cnt > 0)
  {
    bounds.push_back(std::make_pair(left_bound, numNodes - 1));
  }
  //double ts_end = Realm::Clock::current_time_in_microseconds();
  //printf("Partitioning time = %.2lfus\n", ts_end - ts_mid);
  assert(bounds.size() == (size_t)numParts);

  // First, we partition the vertices
  LegionRuntime::Arrays::Rect<1> color_rect(
      LegionRuntime::Arrays::Point<1>(0),
      LegionRuntime::Arrays::Point<1>(numParts - 1));
  Domain color_domain = Domain::from_rect<1>(color_rect);
  {
    DomainColoring pvt_vtx_coloring;
    for (int color = 0; color < numParts; color++)
    {
      LegionRuntime::Arrays::Rect<1> subrect_pvt(
          LegionRuntime::Arrays::Point<1>(bounds[color].first),
          LegionRuntime::Arrays::Point<1>(bounds[color].second));
      pvt_vtx_coloring[color] = Domain::from_rect<1>(subrect_pvt);
    }
    IndexPartition vtx_ip
      = runtime->create_index_partition(ctx, vtx_is, color_domain,
                                        pvt_vtx_coloring, true);
    row_ptr_lp = runtime->get_logical_partition(ctx, row_ptr_lr, vtx_ip);
    raw_row_lp = runtime->get_logical_partition(ctx, raw_row_lr, vtx_ip);
  }
  // Second, we partition the edges
  {
    DomainColoring edges_coloring;
    E_ID index = 0;
    for (int color = 0; color < numParts; color++)
    {
      log_gnn.print("left_bound = %u right_bound = %u",
                    bounds[color].first, bounds[color].second);
      LegionRuntime::Arrays::Rect<1> subrect(
          LegionRuntime::Arrays::Point<1>(index),
          LegionRuntime::Arrays::Point<1>(raw_rows[bounds[color].second]- 1));
      index = raw_rows[bounds[color].second];
      edges_coloring[color] = Domain::from_rect<1>(subrect);
    }
    IndexPartition col_idx_ip
      = runtime->create_index_partition(ctx, edge_is, color_domain,
                                        edges_coloring, true);
    col_idx_lp = runtime->get_logical_partition(ctx, col_idx_lr, col_idx_ip);
    raw_col_lp = runtime->get_logical_partition(ctx, raw_col_lr, col_idx_ip);
  }
  free(raw_rows);
}
