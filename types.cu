#include "types.h"
#include "gnn.h"
#include "cuda_helper.h"

template<typename DT, int dim>
TensorAccessorRO<DT, dim>::TensorAccessorRO(PhysicalRegion region,
                                   RegionRequirement req,
                                   FieldID fid,
                                   Context ctx,
                                   Runtime* runtime,
                                   ResourceManager* manager)
  : acc(region, fid)
{
  rect = runtime->get_index_space_domain(
      ctx, req.region.get_index_space());
  assert(acc.accessor.is_dense_arbitrary(rect));
  ptr = acc.ptr(rect);
  std::set<Memory> memories;
  region.get_memories(memories);
  assert(memories.size() == 1);
  memory = *memories.begin();
  if (memory.kind() == Memory::GPU_FB_MEM) {
    fbCache = NULL;
  } else if (memory.kind() == Memory::Z_COPY_MEM) {
    int id = manager->assign(region, rect.volume());
    assert(id >= 0);
    fbCache = (DT*) manager->fbCache[id].ptr;
    checkCUDA(cudaMemcpyAsync(fbCache, ptr, rect.volume() * sizeof(DT),
        cudaMemcpyHostToDevice));
  } else {
    assert(false);
  }
}

template<typename DT, int dim>
TensorAccessorRW<DT, dim>::TensorAccessorRW(PhysicalRegion region,
                                   RegionRequirement req,
                                   FieldID fid,
                                   Context ctx,
                                   Runtime* runtime,
                                   ResourceManager* manager)
  : acc(region, fid)
{
  rect = runtime->get_index_space_domain(
      ctx, req.region.get_index_space());
  assert(acc.accessor.is_dense_arbitrary(rect));
  ptr = acc.ptr(rect);
  std::set<Memory> memories;
  region.get_memories(memories);
  assert(memories.size() == 1);
  memory = *memories.begin();
  if (memory.kind() == Memory::GPU_FB_MEM) {
    fbCache = NULL;
  } else if (memory.kind() == Memory::Z_COPY_MEM) {
    int id = manager->assign(region, rect.volume());
    assert(id >= 0);
    fbCache = (DT*) manager->fbCache[id].ptr;
    checkCUDA(cudaMemcpyAsync(fbCache, ptr, rect.volume() * sizeof(DT),
        cudaMemcpyHostToDevice));
  } else {
    assert(false);
  }
}

template<typename DT, int dim>
TensorAccessorWO<DT, dim>::TensorAccessorWO(PhysicalRegion region,
                                   RegionRequirement req,
                                   FieldID fid,
                                   Context ctx,
                                   Runtime* runtime,
                                   ResourceManager* manager)
  : acc(region, fid)
{
  rect = runtime->get_index_space_domain(
      ctx, req.region.get_index_space());
  assert(acc.accessor.is_dense_arbitrary(rect));
  ptr = acc.ptr(rect);
  std::set<Memory> memories;
  region.get_memories(memories);
  assert(memories.size() == 1);
  memory = *memories.begin();
  if (memory.kind() == Memory::GPU_FB_MEM) {
    fbCache = NULL;
  } else if (memory.kind() == Memory::Z_COPY_MEM) {
    int id = manager->assign(region, rect.volume());
    assert(id >= 0);
    fbCache = (DT*) manager->fbCache[id].ptr;
  } else {
    assert(false);
  }
}

template class TensorAccessorRO<NodeStruct, 1>;
template class TensorAccessorRO<EdgeStruct, 1>;
template class TensorAccessorRO<DATATYPE, 1>;
template class TensorAccessorRO<DATATYPE, 2>;
template class TensorAccessorRO<DATATYPE, 3>;
template class TensorAccessorRO<int, 2>;

template class TensorAccessorRW<DATATYPE, 1>;
template class TensorAccessorRW<DATATYPE, 2>;
template class TensorAccessorRW<DATATYPE, 3>;

template class TensorAccessorWO<DATATYPE, 1>;
template class TensorAccessorWO<DATATYPE, 2>;
template class TensorAccessorWO<DATATYPE, 3>;


