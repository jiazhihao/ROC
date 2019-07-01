#include "types.h"
#include "gnn.h"
#include "cuda_helper.h"

template<typename DT, int dim>
TensorAccessorR<DT, dim>::TensorAccessorR(PhysicalRegion region,
                                   RegionRequirement req,
                                   FieldID fid,
                                   Context ctx,
                                   Runtime* runtime,
                                   ResourceManager* manager)
{
  const AccessorRO<DT, dim> acc(region, fid);
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

template<typename DT>
__global__
void zero_array(DT* ptr, coord_t size)
{
  CUDA_KERNEL_LOOP(i, size)
  {
    ptr[i] = 0;
  }
}

template<typename DT, int dim>
TensorAccessorW<DT, dim>::TensorAccessorW(PhysicalRegion region,
                                          RegionRequirement req,
                                          FieldID fid,
                                          Context ctx,
                                          Runtime* runtime,
                                          ResourceManager* manager,
                                          bool readOutput)
{
  rect = runtime->get_index_space_domain(
      ctx, req.region.get_index_space());
  if (readOutput) {
    const AccessorRW<DT, dim> acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
  } else {
    const AccessorWO<DT, dim> acc(region, fid);
    assert(acc.accessor.is_dense_arbitrary(rect));
    ptr = acc.ptr(rect);
  }
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
    if (readOutput) {
      checkCUDA(cudaMemcpyAsync(fbCache, ptr, rect.volume() * sizeof(DT),
          cudaMemcpyHostToDevice));
    } else {
      // Currently we zero init the fbCache if not read output
      zero_array<DT><<<GET_BLOCKS(rect.volume()), CUDA_NUM_THREADS>>>(
          fbCache, rect.volume());
    }
  } else {
    assert(false);
  }
}

template class TensorAccessorR<NodeStruct, 1>;
template class TensorAccessorR<EdgeStruct, 1>;
template class TensorAccessorR<DATATYPE, 1>;
template class TensorAccessorR<DATATYPE, 2>;
template class TensorAccessorR<DATATYPE, 3>;
template class TensorAccessorR<int, 2>;

template class TensorAccessorW<DATATYPE, 1>;
template class TensorAccessorW<DATATYPE, 2>;
template class TensorAccessorW<DATATYPE, 3>;
