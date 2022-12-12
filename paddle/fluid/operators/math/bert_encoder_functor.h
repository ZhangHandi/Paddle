/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>  // NOLINT
#endif
#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>

#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CUDATypeTraits;

template <>
struct CUDATypeTraits<half> {
  typedef platform::float16 TYPE;
};

template <>
struct CUDATypeTraits<float> {
  typedef float TYPE;
};

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
// This functor involves a fusion calculation in Ernie or Bert.
//  The fusion mode is as follows:
//
//      in_var  emb       in_var   emb
//        |      |          |       |
//      lookup_table      lookup_table
//            |                 |
//         lkt_var           lkt_var
//             \                /
//              elementwise_add
//                     |
//                elt_out_var
//
template <typename T>
class EmbEltwiseLayerNormFunctor {
 public:
  void operator()(int batch,
                  int seq_len,
                  int hidden,
                  const int64_t *ids,
                  const T *scale,
                  const T *bias,
                  const int64_t *embs,
                  T *output,
                  float eps,
                  int input_num,
                  gpuStream_t stream);
};

// This functor involves a fusion calculation in Ernie or Bert.
// The fusion mode is as follows:
//
//         |    |
//         matmul
//           |
//       eltwise_add
//           |
//        softmax    /
//           \      /
//             matmul
//               |

template <typename T>
class MultiHeadGPUComputeFunctor {
 public:
  void operator()(const phi::GPUContext &dev_ctx,
                  int batch,
                  int seq_len,
                  int head_num,
                  int head_size,
                  T *qkptr,
                  const T *bias_qk_ptr,
                  bool bias_is_mask,
                  T *tptr,
                  T alpha,
                  T beta);
};

// This functor involves a fusion calculation in Ernie or Bert.
// The fusion mode is as follows:
//
// |           |
// other_op1   other_op2
//      |           |
//      |------elementwise_add
//                  |
//              layer_norm
//                  |
//              other_op3
//                  |

template <typename T>
class SkipLayerNormFunctor {
 public:
  void operator()(const int num,
                  const int hidden,
                  const T *input1,
                  const T *input2,
                  const T *scale,
                  const T *bias,
                  T *output,
                  float eps,
                  gpuStream_t stream);
};

//template <typename T, typename T2>
class SkipLayerNormFunctorInt8 {
 public:
  void operator()(gpuStream_t stream,
                  const int32_t ld,
                  const int32_t total,
                  const half *input1,
                  const int8_t *input2,
                  const half *bias,
                  const half *scale,
                  int8_t *output,
                  const float input1_scale,
                  const float input2_scale,
                  const float output_scale);
};
#endif

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

static inline __device__ uint32_t float4_to_char4(float x,
                                                  float y,
                                                  float z,
                                                  float w) {
  uint32_t dst;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 720
  uint32_t a; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(a) : "f"(x));
  uint32_t b; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(b) : "f"(y));
  uint32_t c; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(c) : "f"(z));
  uint32_t d; asm volatile("cvt.rni.sat.s32.f32 %0, %1;\n" : "=r"(d) : "f"(w));

  asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2,  0;\n" : "=r"(dst) : "r"(d), "r"(c));
  asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %0;\n" : "+r"(dst) : "r"(b), "r"(a));
#else
  char4 tmp;
  tmp.x = x;
  tmp.y = y;
  tmp.z = z;
  tmp.w = w;
  dst = reinterpret_cast<const uint32_t&>(tmp);
#endif
  return dst;
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
