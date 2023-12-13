#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle import _C_ops, _legacy_C_ops
from paddle.base import core
from paddle.base.data_feeder import check_dtype, check_variable_and_dtype
from paddle.base.framework import default_main_program
from paddle.base.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode, in_dynamic_or_pir_mode


def kv_split_fused_op(kv_input):
    if in_dynamic_mode():
        return _C_ops.kv_split_fused_op(kv_input)
    helper = LayerHelper('kv_split_fused_op', **locals())
    k_out = helper.create_variable_for_type_inference(dtype=kv_input.dtype)
    v_out = helper.create_variable_for_type_inference(dtype=kv_input.dtype)

    helper.append_op(
        type='kv_split_fused_op',
        inputs={'kv_input': kv_input},
        outputs={'k_output': k_out, "v_output":[v_out]})
    return k_out, v_out