#ifndef _TYPES_H_
#define _TYPES_H_

#include "legion.h"
typedef uint32_t V_ID;
typedef uint64_t E_ID;
typedef float DATATYPE;

struct NodeStruct {
  E_ID index;
};

struct EdgeStruct {
  V_ID src, dst;
};

using namespace Legion;

template<typename FT, int N, typename T = coord_t> using AccessorRO = FieldAccessor<READ_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorRW = FieldAccessor<READ_WRITE,FT,N,T,Realm::AffineAccessor<FT,N,T> >;
template<typename FT, int N, typename T = coord_t> using AccessorWO = FieldAccessor<WRITE_ONLY,FT,N,T,Realm::AffineAccessor<FT,N,T> >;

enum FieldIDs {
  FID_DATA,
};

class ResourceManager;

template<typename DT, int dim>
struct TensorAccessorRO {
  TensorAccessorRO(PhysicalRegion region,
                   RegionRequirement req,
                   FieldID fid,
                   Context ctx,
                   Runtime* runtime,
                   ResourceManager* manager);
  const AccessorRO<DT, dim> acc;
  Rect<dim> rect;
  Memory memory;
  const DT *ptr;
  DT *fbCache;
};

template<typename DT, int dim>
struct TensorAccessorRW {
  TensorAccessorRW(PhysicalRegion region,
                   RegionRequirement req,
                   FieldID fid,
                   Context ctx,
                   Runtime* runtime,
                   ResourceManager* manager);
  const AccessorRW<DT, dim> acc;
  Rect<dim> rect;
  Memory memory;
  DT *ptr;
  DT *fbCache;
};

template<typename DT, int dim>
struct TensorAccessorWO {
  TensorAccessorWO(PhysicalRegion region,
                   RegionRequirement req,
                   FieldID fid,
                   Context ctx,
                   Runtime* runtime,
                   ResourceManager* manager);
  const AccessorWO<DT, dim> acc;
  Rect<dim> rect;
  Memory memory;
  DT *ptr;
  DT *fbCache;
};


#endif
