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
#include "realm/runtime_impl.h"
#include "realm/cuda/cuda_module.h"
#include <fstream>
#include <sstream>

LegionRuntime::Logger::Category log_load("gnn");

void load_features_impl(const Task *task,
                        const std::vector<PhysicalRegion> &regions,
                        Context ctx, Runtime* runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  std::string prefix((char*)task->args);
  const AccessorWO<DATATYPE, 2> accInput(regions[0], FID_DATA);
  Rect<2> rectInput = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  V_ID rowLeft = rectInput.lo[1], rowRight = rectInput.hi[1];
  int inDim = rectInput.hi[0] - rectInput.lo[0] + 1;
  assert(accInput.accessor.is_dense_arbitrary(rectInput));
  DATATYPE* input = accInput.ptr(rectInput.lo);
  std::string csvFile = prefix + ".feats.csv";
  std::string binFile = prefix + ".feats.bin";
  FILE* binFin = fopen(binFile.c_str(), "rb");
  if (binFin == NULL) {
    log_load.print("Load features from CSV: file(%s) rowLeft(%u) rowRight(%u)",
                   csvFile.c_str(), rowLeft, rowRight);
    std::fstream csvFin;
    csvFin.open(csvFile, std::ios::in);
    std::string line, word;
    // Skip the first rowLeft lines
    for (V_ID v = 0; v < rowLeft; v++)
      std::getline(csvFin, line);
    for (V_ID v = rowLeft; v <= rowRight; v++) {
      std::getline(csvFin, line);
      std::stringstream ss(line);
      int feat_cnt = 0;
      while (std::getline(ss, word, ',')) {
        float num = std::stof(word);
        input[(v - rowLeft) * inDim + feat_cnt] = num;
        feat_cnt ++;
      }
      assert(feat_cnt == inDim);
      if (v % 10000 == 0) log_load.print("Loaded %u/%u nodes", v, rowRight);
    }
    FILE* binFout = fopen((prefix + ".feats.bin").c_str(), "wb");
    fwrite(input, sizeof(DATATYPE), rectInput.volume(), binFout);
    fclose(binFout);
    csvFin.close();
  } else {
    log_load.print("Load features from Binary: file(%s) rowLeft(%u) rowRight(%u)\n",
                   binFile.c_str(), rowLeft, rowRight);
    size_t ret = fread(input, sizeof(DATATYPE), rectInput.volume(), binFin);
    assert(ret == rectInput.volume());
    fclose(binFin);
  }
}

LoadFeaturesTask::LoadFeaturesTask(const Model& model,
                                   const Tensor& input,
                                   const std::string& filename)
  : TaskLauncher(LOAD_FEATS_TASK_ID,
                 TaskArgument(filename.c_str(), MAX_FILE_LEN))
{
  // regions[0]: input
  {
    RegionRequirement rr(input.region, WRITE_ONLY, EXCLUSIVE, input.region,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

void load_labels_impl(const Task *task,
                      const std::vector<PhysicalRegion> &regions,
                      Context ctx, Runtime* runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  std::string prefix((char*)task->args);
  const AccessorWO<DATATYPE, 2> accLabel(regions[0], FID_DATA);
  Rect<2> rectLabel = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  V_ID rowLeft = rectLabel.lo[1], rowRight = rectLabel.hi[1];
  int inDim = rectLabel.hi[0] - rectLabel.lo[0] + 1;
  assert(accLabel.accessor.is_dense_arbitrary(rectLabel));
  DATATYPE* label = accLabel.ptr(rectLabel.lo);
  // TODO: remove me
  //for (V_ID v = rowLeft; v<= rowRight; v++)
  //  for (int i = 0; i < inDim; i++)
  //    label[(v-rowLeft)*inDim+i] = i == 0 ? 1.0f : 0.0f;
  //return;
  std::string filename = prefix + ".label";
  log_load.print("Load input labels from %s", filename.c_str());
  FILE* file = fopen(filename.c_str(), "r");
  assert(file != NULL);
  // Skip the first rowLeft lines
  int idx;
  for (V_ID v = 0; v < rowLeft; v++)
    fscanf(file, "%d", &idx);
  for (V_ID v = rowLeft; v <= rowRight; v++) {
    fscanf(file, "%d", &idx);
    assert(idx >= 0 && idx < inDim);
    for (int i = 0; i < inDim; i++)
      label[(v - rowLeft) * inDim + i] = (i == idx) ? 1.0 : 0.0;
  }
  fclose(file);
}

LoadLabelsTask::LoadLabelsTask(const Model& model,
                               const Tensor& input,
                               const std::string& filename)
  : TaskLauncher(LOAD_LABEL_TASK_ID,
                 TaskArgument(filename.c_str(), MAX_FILE_LEN))
{
  // regions[0]: input
  {
    RegionRequirement rr(input.region, WRITE_ONLY, EXCLUSIVE, input.region,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

void load_mask_impl(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime* runtime)
{
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);
  std::string prefix((char*)task->args);
  const AccessorWO<int, 2> accMask(regions[0], FID_DATA);
  Rect<2> rectMask = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  V_ID rowLeft = rectMask.lo[1], rowRight = rectMask.hi[1];
  assert(rectMask.hi[0] == rectMask.lo[0]);
  assert(accMask.accessor.is_dense_arbitrary(rectMask));
  int* mask = accMask.ptr(rectMask.lo);
  // TODO: remove me
  //for (V_ID i = rowLeft; i <= rowRight; i++)
  //  mask[i - rowLeft] = MASK_TRAIN;
  //return;
  std::string filename = prefix + ".mask";
  log_load.print("Load train mask: filename(%s) rowLeft(%u) rowRight(%u)",
                 filename.c_str(), rowLeft, rowRight); 
  std::fstream fin;
  fin.open(filename, std::ios::in);
  std::string line, word;
  // Skip the first rowLeft lines
  for (V_ID v = 0; v < rowLeft; v++)
    std::getline(fin, line);
  for (V_ID v = rowLeft; v <= rowRight; v++) {
    std::getline(fin, line);
    if (line == "Train") {
      mask[v - rowLeft] = MASK_TRAIN;
    } else if (line == "Val") {
      mask[v - rowLeft] = MASK_VAL;
    } else if (line == "Test") {
      mask[v - rowLeft] = MASK_TEST;
    } else if (line == "None") {
      mask[v - rowLeft] = MASK_NONE;
    } else {
      printf("Unrecognized mask: %s\n", line.c_str());
      assert(false);
    }
  }
}

LoadMaskTask::LoadMaskTask(const Model& model,
                           const Tensor& input,
                           const std::string& filename)
  : TaskLauncher(LOAD_MASK_TASK_ID, 
                 TaskArgument(filename.c_str(), MAX_FILE_LEN))
{
  // regions[0]: input
  {
    RegionRequirement rr(input.region, WRITE_ONLY, EXCLUSIVE, input.region,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

void load_graph_impl(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime* runtime)
{
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  std::string prefix((char*) task->args);
  const AccessorWO<E_ID, 1> acc_raw_rows(regions[0], FID_DATA);
  const AccessorWO<V_ID, 1> acc_raw_cols(regions[1], FID_DATA);
  Rect<1> rect_raw_rows = runtime->get_index_space_domain(
                              ctx, task->regions[0].region.get_index_space());
  Rect<1> rect_raw_cols = runtime->get_index_space_domain(
                              ctx, task->regions[1].region.get_index_space());
  V_ID rowLeft = rect_raw_rows.lo[0], rowRight = rect_raw_rows.hi[0];
  E_ID colLeft = rect_raw_cols.lo[0], colRight = rect_raw_cols.hi[0];
  assert(acc_raw_rows.accessor.is_dense_arbitrary(rect_raw_rows));
  assert(acc_raw_cols.accessor.is_dense_arbitrary(rect_raw_cols));
  E_ID* raw_rows = acc_raw_rows.ptr(rect_raw_rows.lo);
  V_ID* raw_cols = acc_raw_cols.ptr(rect_raw_cols.lo);
  std::string filename = prefix + ".add_self_edge.lux";
  log_load.print("Load task: file(%s) rowLeft(%u) rowRight(%u) colLeft(%zu) colRight(%zu)",
                filename.c_str(), rowLeft, rowRight, colLeft, colRight);
  FILE* fd = fopen(filename.c_str(), "rb");
  assert(fd != NULL);
  int fseek_ret;
  size_t fread_ret;
  V_ID nv;
  E_ID ne;
  assert(fread(&nv, sizeof(V_ID), 1, fd) == 1);
  assert(fread(&ne, sizeof(E_ID), 1, fd) == 1);
  fseek_ret =
    fseeko(fd, FILE_HEADER_SIZE + sizeof(E_ID) * (size_t)rowLeft, SEEK_SET);
  assert(fseek_ret == 0);
  fread_ret =
    fread(raw_rows, sizeof(E_ID), (size_t)(rowRight - rowLeft + 1), fd);
  assert(fread_ret == rowRight - rowLeft + 1);
  fseek_ret = 
    fseeko(fd, FILE_HEADER_SIZE + sizeof(E_ID) * (size_t)nv
               + sizeof(V_ID) * (size_t)colLeft, SEEK_SET);
  assert(fseek_ret == 0);
  fread_ret =
    fread(raw_cols, sizeof(V_ID), (size_t)(colRight - colLeft + 1), fd);
  assert(fread_ret == colRight - colLeft + 1);
  fclose(fd);
}

LoadGraphTask::LoadGraphTask(const Model& model,
                             const std::string& filename)
  : IndexLauncher(LOAD_GRAPH_TASK_ID, model.taskIS,
                  TaskArgument(filename.c_str(), MAX_FILE_LEN),
                  model.taskArgs)
{
  // regions[0]: raw_rows
  {
    RegionRequirement rr(model.myGraph.rawRowLP, 0/*projection id*/,
                         WRITE_ONLY, EXCLUSIVE, model.myGraph.rawRowLR,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: raw_cols
  {
    RegionRequirement rr(model.myGraph.rawColLP, 0/*projection id*/,
                         WRITE_ONLY, EXCLUSIVE, model.myGraph.rawColLR,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

__global__
void init_graph_kernel(V_ID rowLeft,
                       V_ID rowRight,
                       E_ID colLeft,
                       NodeStruct* rowPtrs,
                       EdgeStruct* colIdxs,
                       const E_ID* rawRows,
                       const V_ID* rawCols)
{
  for (V_ID n = blockIdx.x * blockDim.x + threadIdx.x;
       n + rowLeft <= rowRight; n += blockDim.x * gridDim.x)
  {
    E_ID startColIdx, endColIdx = rawRows[n];
    if (n == 0)
      startColIdx = colLeft;
    else
      startColIdx = rawRows[n-1];
    rowPtrs[n].index = endColIdx;
    for (E_ID e = startColIdx; e < endColIdx; e++) {
      colIdxs[e - colLeft].src = rawCols[e - colLeft];
      colIdxs[e - colLeft].dst = n + rowLeft;
    }
  }
}

ResourceManager* init_task_impl(const Task *task,
                                const std::vector<PhysicalRegion> &regions,
                                Context ctx, Runtime* runtime)
{
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  const Graph* graph = (Graph*) task->args;
  const AccessorWO<NodeStruct, 1> accRowPtr(regions[0], FID_DATA);
  const AccessorWO<EdgeStruct, 1> accColIdx(regions[1], FID_DATA);
  const AccessorRO<E_ID, 1> accRawRow(regions[2], FID_DATA);
  const AccessorRO<V_ID, 1> accRawCol(regions[3], FID_DATA);
  Rect<1> rectRowPtr = runtime->get_index_space_domain(
                             ctx, task->regions[0].region.get_index_space());
  Rect<1> rectColIdx = runtime->get_index_space_domain(
                             ctx, task->regions[1].region.get_index_space());
  Rect<1> rectRawRow = runtime->get_index_space_domain(
                              ctx, task->regions[2].region.get_index_space());
  Rect<1> rectRawCol = runtime->get_index_space_domain(
                              ctx, task->regions[3].region.get_index_space());

  assert(accRowPtr.accessor.is_dense_arbitrary(rectRowPtr));
  assert(accColIdx.accessor.is_dense_arbitrary(rectColIdx));
  assert(accRawRow.accessor.is_dense_arbitrary(rectRawRow));
  assert(accRawCol.accessor.is_dense_arbitrary(rectRawCol));
  NodeStruct* rowPtrs = accRowPtr.ptr(rectRowPtr);
  EdgeStruct* colIdxs = accColIdx.ptr(rectColIdx);
  const E_ID* rawRows = accRawRow.ptr(rectRawRow);
  const V_ID* rawCols = accRawCol.ptr(rectRawCol);
  V_ID rowLeft = rectRowPtr.lo[0], rowRight = rectRowPtr.hi[0];
  E_ID colLeft = rectColIdx.lo[0], colRight = rectColIdx.hi[0];
  log_load.print("Init task: rowLeft(%u) rowRight(%u) colLeft(%zu) colRight(%zu)",
                 rowLeft, rowRight, colLeft, colRight);
  // init graph
  init_graph_kernel<<<GET_BLOCKS(rowRight - rowLeft + 1), CUDA_NUM_THREADS>>>(
      rowLeft, rowRight, colLeft, rowPtrs, colIdxs, rawRows, rawCols);
  checkCUDA(cudaDeviceSynchronize());

  ResourceManager* manager = new ResourceManager();
  manager->proc_id = task->current_proc.id;
  // init nccl
  //int numRanks = graph->numParts / graph->numMachines;
  //int myRank = task->current_proc.id % numRanks;
  //int node = task->current_proc.address_space();
  //piece.nccl = graph->nccl[node*numRanks+myRank];
  //printf("Before ncclCommInitRank: numRanks(%d) id(%s) myrank(%d) processorId(%d)\n", numRanks, graph->ncclID[node].internal, 0, task->current_proc.id);
  //NCCLCheck(ncclCommInitRank(&piece.nccl, numRanks, graph->ncclID[node], task->current_proc.id % numRanks));
  //printf("After ncclCommInitRank\n");
  // init cublas
  checkCUDA(cublasCreate(&(manager->blas)));
  checkCUDNN(cudnnCreate(&(manager->dnn)));
  // init dropout states
  checkCUDNN(cudnnDropoutGetStatesSize(manager->dnn, &(manager->dropoutSize)));
  checkCUDA(cudaMalloc(&(manager->dropoutStates), manager->dropoutSize));
  //manager->numNodes = graph->numNodes;
  //manager->numEdges = graph->numEdges;
  //manager->numParts = graph->numParts;
  // Allocate fbInput/fbOutput on the same memory as rowPtr
  std::set<Memory> memFB;
  regions[0].get_memories(memFB);
  assert(memFB.size() == 1);
  assert(memFB.begin()->kind() == Memory::GPU_FB_MEM);
  Realm::MemoryImpl* memImpl =
      Realm::get_runtime()->get_memory_impl(*memFB.begin());
  Realm::Cuda::GPUFBMemory* memFBImpl = (Realm::Cuda::GPUFBMemory*) memImpl;
  manager->allocator = memFBImpl;
  for (int i = 0; i < MAX_NUM_CACHES; i++) {
    if (i == 0)
      manager->fbCache[i].volume = graph->maxHidden * (graph->numNodes + 128);
    else
      manager->fbCache[i].volume = graph->maxHidden * (rowRight - rowLeft + 1);
    manager->fbCache[i].region = LogicalRegion::NO_REGION;
    off_t offset = memFBImpl->alloc_bytes(manager->fbCache[i].volume * sizeof(DATATYPE));
    assert(offset >= 0);
    manager->fbCache[i].ptr = (DATATYPE*) memFBImpl->get_direct_ptr(offset, 0);
  }
  return manager;
}

InitTask::InitTask(const Model& model)
  : IndexLauncher(INIT_TASK_ID, model.taskIS,
                  TaskArgument(&(model.myGraph), sizeof(Graph)), model.taskArgs)
{
  // regions[0]: row_ptrs
  {
    RegionRequirement rr(model.myGraph.rowPtrLP, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, model.myGraph.rowPtrLR,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[1]: col_idxs
  {
    RegionRequirement rr(model.myGraph.colIdxLP, 0/*identity*/,
                         WRITE_ONLY, EXCLUSIVE, model.myGraph.colIdxLR,
                         MAP_TO_FB_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[2]: raw_rows
  {
    RegionRequirement rr(model.myGraph.rawRowLP, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, model.myGraph.rawRowLR,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
  // regions[3]: raw_cols
  {
    RegionRequirement rr(model.myGraph.rawColLP, 0/*identity*/,
                         READ_ONLY, EXCLUSIVE, model.myGraph.rawColLR,
                         MAP_TO_ZC_MEMORY);
    rr.add_field(FID_DATA);
    add_region_requirement(rr);
  }
}

