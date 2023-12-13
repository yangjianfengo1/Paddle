// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/put_along_axis_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/kernels/funcs/aligned_vector.h"
#include "paddle/phi/kernels/kv_split_fused_op_kernel.h"

namespace phi {
template <typename T, int VecSize>
__global__ void fusedKV_split_kernel(
    const T *kv_input,
    T *k_out,
    T *v_out,
    const int32_t elem_cnt,
    const int head_num,
    const int head_dim) {
  int64_t global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;
  LoadT bias_vec;
  const int32_t offset = head_num * head_dim;
  const int32_t hidden_size = 2 * offset;
  
  for (int32_t linear_index = global_thread_idx * VecSize,
               step = gridDim.x * blockDim.x * VecSize;
       linear_index < elem_cnt;
       linear_index += step) {
    Load<T, VecSize>(&kv_input[linear_index], &src_vec);
    const int32_t token_id = linear_index / hidden_size;
    const int32_t kv_id = (linear_index % hidden_size) / offset;
    const int32_t fused_head_id = linear_index % offset;
    const int32_t write_idx = token_id * offset + fused_head_id;
    if (kv_id == 0) {
        Store<T, VecSize>(
          src_vec,
          &k_out[write_idx]);
    } else {
        Store<T, VecSize>(
          src_vec,
          &v_out[write_idx]);
    }
  }
}
   
template <typename T, typename Context>
void KvSplitFusedKernel(
        const Context& dev_ctx,
        const DenseTensor& kv_input, 
        DenseTensor* k_output,
        DenseTensor* v_output){
    dev_ctx.template Alloc<T>(k_output);
    dev_ctx.template Alloc<T>(v_output);
    const auto x_dims = kv_input.dims();
    auto cu_stream = dev_ctx.stream();
    const int bsz = x_dims[0];
    const int seq_len = x_dims[1];
    const int num_head = x_dims[3];
    const int dim_head = x_dims[4];
    constexpr int VEC_16B = 16;
  
    constexpr int PackSize = VEC_16B / sizeof(T);
    const int elem_cnt = bsz * seq_len * num_head * dim_head * 2;
    const int pack_num = elem_cnt / PackSize;
    assert(elem_cnt % PackSize == 0);
    const int blocksize = 128;
    const int grid_size = (pack_num + blocksize - 1) / blocksize;
    fusedKV_split_kernel<T, PackSize>
      <<<grid_size, blocksize, 0, cu_stream>>>(
        const_cast<T*>(kv_input.data<T>()),
        k_output->data<T>(),
        v_output->data<T>(),
        elem_cnt,
        num_head,
        dim_head);
}
}

PD_REGISTER_KERNEL(kv_split_fused_op,
                   GPU,
                   ALL_LAYOUT,
                   phi::KvSplitFusedKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
