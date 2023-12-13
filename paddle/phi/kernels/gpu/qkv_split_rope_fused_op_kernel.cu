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
#include "paddle/phi/kernels/qkv_split_rope_fused_op_kernel.h"

namespace phi {
template <typename T, int VecSize, int rope_pack_size>
__global__ void qkv_split_rope_uvit_kernel(
        const T *qkv_input,
        T *q_out,
        T *k_out,
        T *v_out,
        const float *cos_emb,
        const float *sin_emb,
        const int *sequence_lengths,
        const int rotary_emb_dims,
        const int batch_size,
        const int head_num,
        const int seq_len,
        const int last_dim,
        const int qkv_seq_lens_offset) {
    int bi = blockIdx.x;
    int hi = blockIdx.y;
    int si = blockIdx.z;

    using LoadT = AlignedVector<T, VecSize>;
    LoadT q_vec;
    LoadT k_vec;
    LoadT v_vec;
    LoadT bias_vec;
    const int step = VecSize * blockDim.x;
    // 复制V
    for (int ti = threadIdx.x * VecSize; ti < last_dim; ti += step) {
        const int v_index = bi * seq_len * 3 * head_num * last_dim +
            si * 3 * head_num * last_dim + 
            2 * head_num * last_dim + 
            hi * last_dim + ti;
        Load<T, VecSize>(&qkv_input[v_index], &v_vec);
        const int store_index = bi * seq_len * head_num * last_dim +
            si * head_num * last_dim + 
            hi * last_dim + ti;
        Store<T, VecSize>(
            v_vec,
            &v_out[store_index]);
    }
    // 时间维度只复制
    if (si < qkv_seq_lens_offset) {
        for (int ti = threadIdx.x * VecSize; ti < last_dim; ti += step) {
            const int q_index = bi * seq_len * 3 * head_num * last_dim +
                si * 3 * head_num * last_dim + 
                hi * last_dim + ti;
            const int k_index = q_index + head_num * last_dim;
            const int store_index = bi * seq_len * head_num * last_dim +
                si * head_num * last_dim + 
                hi * last_dim + ti;
            Load<T, VecSize>(&qkv_input[q_index], &q_vec);
            Load<T, VecSize>(&qkv_input[k_index], &k_vec);
            Store<T, VecSize>(
                q_vec,
                &q_out[store_index]);
            Store<T, VecSize>(
                k_vec,
                &k_out[store_index]);
        }
        return;
    }


    int quat_lastdim = last_dim / rope_pack_size;
    int emb_idx = (si - qkv_seq_lens_offset)  * last_dim;
    const int store_index = bi * seq_len * head_num * last_dim +
        si * head_num * last_dim + 
        hi * last_dim;

    extern __shared__ __align__(sizeof(LoadT)) char share_mem[];
    T* qk_share_mem = reinterpret_cast<T*>(share_mem);

    for (int ti = threadIdx.x * VecSize; ti < last_dim; ti += step) {
        const int q_index = bi * seq_len * 3 * head_num * last_dim +
                si * 3 * head_num * last_dim + 
                hi * last_dim + ti;
        const int k_index = q_index + head_num * last_dim;
        Load<T, VecSize>(&qkv_input[q_index], &q_vec);
        Load<T, VecSize>(&qkv_input[k_index], &k_vec);
        Store<T, VecSize>(
                q_vec,
                &qk_share_mem[ti]);
        Store<T, VecSize>(
                k_vec,
                &qk_share_mem[ti + last_dim]);
    }
    __syncthreads();
    for (int ti = threadIdx.x; ti < quat_lastdim; ti += blockDim.x) {
        float q_data[rope_pack_size];
        float k_data[rope_pack_size];
        float cos_tmp[rope_pack_size];
        float sin_tmp[rope_pack_size];

        #pragma unroll
        for (int i = 0; i < rope_pack_size; ++i) {
            q_data[i] = static_cast<float>(qk_share_mem[ti + i * quat_lastdim]);
            k_data[i] = static_cast<float>(qk_share_mem[ti + i * quat_lastdim + last_dim]);
            cos_tmp[i] = cos_emb[emb_idx + ti + i * quat_lastdim];
            sin_tmp[i] = sin_emb[emb_idx + ti + i * quat_lastdim];
        }
        
        q_out[store_index + ti] = static_cast<T>(q_data[0] * cos_tmp[0] - q_data[1] * sin_tmp[0]);
        k_out[store_index + ti] = static_cast<T>(k_data[0] * cos_tmp[0] - k_data[1] * sin_tmp[0]);
        
        q_out[store_index + ti + quat_lastdim] = static_cast<T>(q_data[1] * cos_tmp[1] + q_data[0] * sin_tmp[1]);
        k_out[store_index + ti + quat_lastdim] = static_cast<T>(k_data[1] * cos_tmp[1] + k_data[0] * sin_tmp[1]);

        q_out[store_index + ti + 2 * quat_lastdim] = static_cast<T>(q_data[2] * cos_tmp[2] - q_data[3] * sin_tmp[2]);
        k_out[store_index + ti + 2 * quat_lastdim] = static_cast<T>(k_data[2] * cos_tmp[2] - k_data[3] * sin_tmp[2]);
        
        q_out[store_index + ti + 3 * quat_lastdim] = static_cast<T>(q_data[3] * cos_tmp[3] + q_data[2] * sin_tmp[3]);
        k_out[store_index + ti + 3 * quat_lastdim] = static_cast<T>(k_data[3] * cos_tmp[3] + k_data[2] * sin_tmp[3]);
    }
}

   

template <typename T, typename Context>
void QkvSplitRopeFusedKernel(
    const Context& dev_ctx,
    const DenseTensor& qkv_input, 
    const DenseTensor& rotary_emb,
    const DenseTensor& seq_lens, 
    const int rotary_emb_dims, 
    const int qkv_seq_lens_offset,
    DenseTensor* q_output, 
    DenseTensor* k_output,
    DenseTensor* v_output){
  dev_ctx.template Alloc<T>(q_output);
  dev_ctx.template Alloc<T>(k_output);
  dev_ctx.template Alloc<T>(v_output);
  const auto x_dims = qkv_input.dims();
  auto cu_stream = dev_ctx.stream();
  const int bsz = x_dims[0];
  const int seq_len = x_dims[1];
  const int num_head = x_dims[3];
  const int dim_head = x_dims[4];

  const int32_t emb_seq_len = seq_len - qkv_seq_lens_offset;
  constexpr int VEC_16B = 16;
  constexpr int PackSize = VEC_16B / sizeof(T);
  assert(dim_head % PackSize == 0);
  constexpr int rope_pack = 4;

    dim3 grid(bsz, num_head, seq_len * rotary_emb_dims);
    const int last_dim = dim_head / rotary_emb_dims;
    auto getBlockSize = [](int dim) {
        if (dim > 256) {
            return 512;
        } else if (dim > 128) {
            return 256;
        } else if (dim > 64) {
            return 128;
        } else if (dim > 32) {
            return 64;
        } else {
            return 32;
        }
    };
    int BlockSize = getBlockSize(last_dim / rope_pack);
    const float *cos_emb = rotary_emb.data<float>();
    const float *sin_emb = rotary_emb.data<float>() + emb_seq_len * dim_head;
    const int share_mem_size = sizeof(T) * 2 * dim_head;
    qkv_split_rope_uvit_kernel<T, PackSize, rope_pack><<<grid, BlockSize, share_mem_size, cu_stream>>>(
      const_cast<T*>(qkv_input.data<T>()),
      reinterpret_cast<T*>(q_output->data<T>()),
      reinterpret_cast<T*>(k_output->data<T>()),
      reinterpret_cast<T*>(v_output->data<T>()),
      cos_emb,
      sin_emb,
      seq_lens.data<int>(),
      rotary_emb_dims,
      bsz,
      num_head,
      seq_len * rotary_emb_dims,
      last_dim,
      qkv_seq_lens_offset); 
}
}

PD_REGISTER_KERNEL(qkv_split_rope_fused_op,
                   GPU,
                   ALL_LAYOUT,
                   phi::QkvSplitRopeFusedKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
