class <lambda>(torch.nn.Module):
    def forward(self, arg197_1: i64[2, 2048], arg198_1: i64[2, 2048]):
        # No stacktrace found for following nodes
        _frozen_param0: f32[2050, 768] = self._frozen_param0
        _frozen_param1: f32[50272, 768] = self._frozen_param1
        _frozen_param2: f32[768] = self._frozen_param2
        _frozen_param3: f32[768] = self._frozen_param3
        _frozen_param12: f32[768] = self._frozen_param12
        _frozen_param13: f32[768] = self._frozen_param13
        _frozen_param18: f32[768] = self._frozen_param18
        _frozen_param19: f32[768] = self._frozen_param19
        _frozen_param28: f32[768] = self._frozen_param28
        _frozen_param29: f32[768] = self._frozen_param29
        _frozen_param34: f32[768] = self._frozen_param34
        _frozen_param35: f32[768] = self._frozen_param35
        _frozen_param44: f32[768] = self._frozen_param44
        _frozen_param45: f32[768] = self._frozen_param45
        _frozen_param50: f32[768] = self._frozen_param50
        _frozen_param51: f32[768] = self._frozen_param51
        _frozen_param60: f32[768] = self._frozen_param60
        _frozen_param61: f32[768] = self._frozen_param61
        _frozen_param66: f32[768] = self._frozen_param66
        _frozen_param67: f32[768] = self._frozen_param67
        _frozen_param76: f32[768] = self._frozen_param76
        _frozen_param77: f32[768] = self._frozen_param77
        _frozen_param82: f32[768] = self._frozen_param82
        _frozen_param83: f32[768] = self._frozen_param83
        _frozen_param92: f32[768] = self._frozen_param92
        _frozen_param93: f32[768] = self._frozen_param93
        _frozen_param98: f32[768] = self._frozen_param98
        _frozen_param99: f32[768] = self._frozen_param99
        _frozen_param108: f32[768] = self._frozen_param108
        _frozen_param109: f32[768] = self._frozen_param109
        _frozen_param114: f32[768] = self._frozen_param114
        _frozen_param115: f32[768] = self._frozen_param115
        _frozen_param124: f32[768] = self._frozen_param124
        _frozen_param125: f32[768] = self._frozen_param125
        _frozen_param130: f32[768] = self._frozen_param130
        _frozen_param131: f32[768] = self._frozen_param131
        _frozen_param140: f32[768] = self._frozen_param140
        _frozen_param141: f32[768] = self._frozen_param141
        _frozen_param146: f32[768] = self._frozen_param146
        _frozen_param147: f32[768] = self._frozen_param147
        _frozen_param156: f32[768] = self._frozen_param156
        _frozen_param157: f32[768] = self._frozen_param157
        _frozen_param162: f32[768] = self._frozen_param162
        _frozen_param163: f32[768] = self._frozen_param163
        _frozen_param172: f32[768] = self._frozen_param172
        _frozen_param173: f32[768] = self._frozen_param173
        _frozen_param178: f32[768] = self._frozen_param178
        _frozen_param179: f32[768] = self._frozen_param179
        _frozen_param188: f32[768] = self._frozen_param188
        _frozen_param189: f32[768] = self._frozen_param189
        _frozen_param194: f32[768] = self._frozen_param194
        _frozen_param195: f32[768] = self._frozen_param195
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:824, code: input_ids = input_ids.view(-1, input_shape[-1])
        view: i64[2, 2048] = torch.ops.aten.reshape.default(arg197_1, [-1, 2048]);  arg197_1 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:831, code: inputs_embeds = self.embed_tokens(input_ids)
        embedding: f32[2, 2048, 768] = torch.ops.aten.embedding.default(_frozen_param1, view, 1);  _frozen_param1 = view = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:156, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        full_default: f32[2048, 2048] = torch.ops.aten.full.default([2048, 2048], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='xpu', index=0), pin_memory = False)
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:157, code: mask_cond = torch.arange(mask.size(-1), device=device)
        iota: i64[2048] = torch.ops.prims.iota.default(2048, start = 0, step = 1, dtype = torch.int64, device = device(type='xpu', index=0), requires_grad = False)
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:158, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        add: i64[2048] = torch.ops.aten.add.Tensor(iota, 1)
        view_1: i64[2048, 1] = torch.ops.aten.reshape.default(add, [2048, 1]);  add = None
        lt: b8[2048, 2048] = torch.ops.aten.lt.Tensor(iota, view_1);  iota = view_1 = None
        full_default_1: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='xpu', index=0), pin_memory = False)
        where: f32[2048, 2048] = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:137, code: expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)
        full_default_2: b8[2, 1, 2048, 2048] = torch.ops.aten.full.default([2, 1, 2048, 2048], False, dtype = torch.bool, layout = torch.strided, device = device(type='xpu', index=0), pin_memory = False)
        unsqueeze_4: f32[1, 2048, 2048] = torch.ops.aten.unsqueeze.default(where, 0);  where = None
        unsqueeze_5: f32[1, 1, 2048, 2048] = torch.ops.aten.unsqueeze.default(unsqueeze_4, 1);  unsqueeze_4 = None
        slice_5: f32[1, 1, 2048, 2048] = torch.ops.aten.slice.Tensor(unsqueeze_5, 2, 0, 9223372036854775807);  unsqueeze_5 = None
        slice_6: f32[1, 1, 2048, 2048] = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 9223372036854775807);  slice_5 = None
        expand_2: f32[2, 1, 2048, 2048] = torch.ops.aten.expand.default(slice_6, [2, 1, 2048, 2048]);  slice_6 = None
        full_default_3: f32[] = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='xpu', index=0), pin_memory = False)
        where_2: f32[2, 1, 2048, 2048] = torch.ops.aten.where.self(full_default_2, full_default_3, expand_2);  full_default_2 = expand_2 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:101, code: attention_mask = attention_mask.long()
        full_default_4: i64[2, 2048] = torch.ops.aten.full.default([2, 2048], 1, dtype = torch.int64, layout = torch.strided, device = device(type='xpu', index=0), pin_memory = False)
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:104, code: positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1
        cumsum: i64[2, 2048] = torch.ops.aten.cumsum.default(full_default_4, 1);  full_default_4 = None
        sub_1: i64[2, 2048] = torch.ops.aten.sub.Tensor(cumsum, 1);  cumsum = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:107, code: positions = positions[:, past_key_values_length:]
        slice_7: i64[2, 2048] = torch.ops.aten.slice.Tensor(sub_1, 0, 0, 9223372036854775807);  sub_1 = None
        slice_8: i64[2, 2048] = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 9223372036854775807);  slice_7 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:109, code: return super().forward(positions + self.offset)
        add_1: i64[2, 2048] = torch.ops.aten.add.Tensor(slice_8, 2);  slice_8 = None
        
        # File: /home/sdp/chenjunjie/pytorch/torch/nn/modules/sparse.py:162, code: return F.embedding(
        embedding_1: f32[2, 2048, 768] = torch.ops.aten.embedding.default(_frozen_param0, add_1);  _frozen_param0 = add_1 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:865, code: hidden_states = inputs_embeds + pos_embeds
        add_2: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
        getitem: f32[2, 2048, 1] = var_mean[0]
        getitem_1: f32[2, 2048, 1] = var_mean[1];  var_mean = None
        add_3: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        sub_2: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(add_2, getitem_1);  getitem_1 = None
        mul_1: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_2, rsqrt);  sub_2 = rsqrt = None
        mul_2: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_1, _frozen_param2);  mul_1 = _frozen_param2 = None
        add_4: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_2, _frozen_param3);  mul_2 = _frozen_param3 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_3: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        _frozen_param197: bf16[768] = self._frozen_param197
        view_2: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_3, [4096, 768]);  convert_element_type_3 = None
        _frozen_param198: bf16[768, 768] = self._frozen_param198
        _linear_pointwise_default_72: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_2, _frozen_param198, _frozen_param197, 'none', [], '');  _frozen_param198 = _frozen_param197 = None
        view_3: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_72, [2, 2048, 768]);  _linear_pointwise_default_72 = None
        mul_3: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_3, 0.125);  view_3 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param199: bf16[768] = self._frozen_param199
        _frozen_param200: bf16[768, 768] = self._frozen_param200
        _linear_pointwise_default_71: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_2, _frozen_param200, _frozen_param199, 'none', [], '');  _frozen_param200 = _frozen_param199 = None
        view_5: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_71, [2, 2048, 768]);  _linear_pointwise_default_71 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_6: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_5, [2, -1, 12, 64]);  view_5 = None
        permute_2: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        clone: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param201: bf16[768] = self._frozen_param201
        _frozen_param202: bf16[768, 768] = self._frozen_param202
        _linear_pointwise_default_70: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_2, _frozen_param202, _frozen_param201, 'none', [], '');  view_2 = _frozen_param202 = _frozen_param201 = None
        view_8: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_70, [2, 2048, 768]);  _linear_pointwise_default_70 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_9: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_8, [2, -1, 12, 64]);  view_8 = None
        permute_4: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        clone_1: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_10: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_3, [2, 2048, 12, 64]);  mul_3 = None
        permute_5: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_2: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_11: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_2, [24, -1, 64]);  clone_2 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_12: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_13: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_1, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_6: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        bmm: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_11, permute_6);  view_11 = permute_6 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_14: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm, [2, 12, 2048, 2048]);  bmm = None
        add_5: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_14, where_2);  view_14 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_5, full_default_3);  add_5 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_15: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum, [24, 2048, 2048]);  maximum = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_15, [-1], True)
        sub_3: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_15, amax);  view_15 = amax = None
        exp: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_1: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_3: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div);  div = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_12: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_3, torch.bfloat16);  clone_3 = None
        bmm_1: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_12, view_13);  convert_element_type_12 = view_13 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_16: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_1, [2, 12, 2048, 64]);  bmm_1 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_7: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_4: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_17: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_4, [2, 2048, 768]);  clone_4 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param203: bf16[768] = self._frozen_param203
        view_18: bf16[4096, 768] = torch.ops.aten.reshape.default(view_17, [4096, 768]);  view_17 = None
        _frozen_param204: bf16[768, 768] = self._frozen_param204
        _linear_pointwise_default_69: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_18, _frozen_param204, _frozen_param203, 'none', [], '');  view_18 = _frozen_param204 = _frozen_param203 = None
        view_19: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_69, [2, 2048, 768]);  _linear_pointwise_default_69 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_5: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_19);  view_19 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_6: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(add_2, clone_5);  add_2 = clone_5 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_20: f32[4096, 768] = torch.ops.aten.reshape.default(add_6, [-1, 768]);  add_6 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(view_20, [1], correction = 0, keepdim = True)
        getitem_2: f32[4096, 1] = var_mean_1[0]
        getitem_3: f32[4096, 1] = var_mean_1[1];  var_mean_1 = None
        add_7: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        sub_4: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_20, getitem_3);  getitem_3 = None
        mul_4: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_5: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_4, _frozen_param12);  mul_4 = _frozen_param12 = None
        add_8: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_5, _frozen_param13);  mul_5 = _frozen_param13 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_15: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None
        _frozen_param205: bf16[3072] = self._frozen_param205
        _frozen_param206: bf16[768, 3072] = self._frozen_param206
        _linear_pointwise_default_68: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_15, _frozen_param206, _frozen_param205, 'none', [], '');  convert_element_type_15 = _frozen_param206 = _frozen_param205 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_68);  _linear_pointwise_default_68 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param207: bf16[768] = self._frozen_param207
        _frozen_param208: bf16[3072, 768] = self._frozen_param208
        _linear_pointwise_default_67: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu, _frozen_param208, _frozen_param207, 'none', [], '');  relu = _frozen_param208 = _frozen_param207 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_6: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_67);  _linear_pointwise_default_67 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_9: f32[4096, 768] = torch.ops.aten.add.Tensor(view_20, clone_6);  view_20 = clone_6 = None
        view_21: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_9, [2, 2048, 768]);  add_9 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(view_21, [2], correction = 0, keepdim = True)
        getitem_4: f32[2, 2048, 1] = var_mean_2[0]
        getitem_5: f32[2, 2048, 1] = var_mean_2[1];  var_mean_2 = None
        add_10: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        sub_5: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_21, getitem_5);  getitem_5 = None
        mul_6: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_5, rsqrt_2);  sub_5 = rsqrt_2 = None
        mul_7: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_6, _frozen_param18);  mul_6 = _frozen_param18 = None
        add_11: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_7, _frozen_param19);  mul_7 = _frozen_param19 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_20: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        _frozen_param209: bf16[768] = self._frozen_param209
        view_22: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_20, [4096, 768]);  convert_element_type_20 = None
        _frozen_param210: bf16[768, 768] = self._frozen_param210
        _linear_pointwise_default_66: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_22, _frozen_param210, _frozen_param209, 'none', [], '');  _frozen_param210 = _frozen_param209 = None
        view_23: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_66, [2, 2048, 768]);  _linear_pointwise_default_66 = None
        mul_8: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_23, 0.125);  view_23 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param211: bf16[768] = self._frozen_param211
        _frozen_param212: bf16[768, 768] = self._frozen_param212
        _linear_pointwise_default_65: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_22, _frozen_param212, _frozen_param211, 'none', [], '');  _frozen_param212 = _frozen_param211 = None
        view_25: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_65, [2, 2048, 768]);  _linear_pointwise_default_65 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_26: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_25, [2, -1, 12, 64]);  view_25 = None
        permute_13: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_26, [0, 2, 1, 3]);  view_26 = None
        clone_7: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param213: bf16[768] = self._frozen_param213
        _frozen_param214: bf16[768, 768] = self._frozen_param214
        _linear_pointwise_default_64: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_22, _frozen_param214, _frozen_param213, 'none', [], '');  view_22 = _frozen_param214 = _frozen_param213 = None
        view_28: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_64, [2, 2048, 768]);  _linear_pointwise_default_64 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_29: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_28, [2, -1, 12, 64]);  view_28 = None
        permute_15: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
        clone_8: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_30: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_8, [2, 2048, 12, 64]);  mul_8 = None
        permute_16: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
        clone_9: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_31: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_9, [24, -1, 64]);  clone_9 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_32: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_7, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_33: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_8, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_17: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
        bmm_2: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_31, permute_17);  view_31 = permute_17 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_34: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_2, [2, 12, 2048, 2048]);  bmm_2 = None
        add_12: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_34, where_2);  view_34 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_1: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_12, full_default_3);  add_12 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_35: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_1, [24, 2048, 2048]);  maximum_1 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_1: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_35, [-1], True)
        sub_6: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_35, amax_1);  view_35 = amax_1 = None
        exp_1: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_2: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_10: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_1);  div_1 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_29: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_10, torch.bfloat16);  clone_10 = None
        bmm_3: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_29, view_33);  convert_element_type_29 = view_33 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_36: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_3, [2, 12, 2048, 64]);  bmm_3 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_18: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_11: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
        view_37: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_11, [2, 2048, 768]);  clone_11 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param215: bf16[768] = self._frozen_param215
        view_38: bf16[4096, 768] = torch.ops.aten.reshape.default(view_37, [4096, 768]);  view_37 = None
        _frozen_param216: bf16[768, 768] = self._frozen_param216
        _linear_pointwise_default_63: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_38, _frozen_param216, _frozen_param215, 'none', [], '');  view_38 = _frozen_param216 = _frozen_param215 = None
        view_39: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_63, [2, 2048, 768]);  _linear_pointwise_default_63 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_12: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_39);  view_39 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_13: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_21, clone_12);  view_21 = clone_12 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_40: f32[4096, 768] = torch.ops.aten.reshape.default(add_13, [-1, 768]);  add_13 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(view_40, [1], correction = 0, keepdim = True)
        getitem_6: f32[4096, 1] = var_mean_3[0]
        getitem_7: f32[4096, 1] = var_mean_3[1];  var_mean_3 = None
        add_14: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
        sub_7: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_40, getitem_7);  getitem_7 = None
        mul_9: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_7, rsqrt_3);  sub_7 = rsqrt_3 = None
        mul_10: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_9, _frozen_param28);  mul_9 = _frozen_param28 = None
        add_15: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_10, _frozen_param29);  mul_10 = _frozen_param29 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_32: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16);  add_15 = None
        _frozen_param217: bf16[3072] = self._frozen_param217
        _frozen_param218: bf16[768, 3072] = self._frozen_param218
        _linear_pointwise_default_62: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_32, _frozen_param218, _frozen_param217, 'none', [], '');  convert_element_type_32 = _frozen_param218 = _frozen_param217 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_1: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_62);  _linear_pointwise_default_62 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param219: bf16[768] = self._frozen_param219
        _frozen_param220: bf16[3072, 768] = self._frozen_param220
        _linear_pointwise_default_61: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_1, _frozen_param220, _frozen_param219, 'none', [], '');  relu_1 = _frozen_param220 = _frozen_param219 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_13: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_61);  _linear_pointwise_default_61 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_16: f32[4096, 768] = torch.ops.aten.add.Tensor(view_40, clone_13);  view_40 = clone_13 = None
        view_41: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_16, [2, 2048, 768]);  add_16 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(view_41, [2], correction = 0, keepdim = True)
        getitem_8: f32[2, 2048, 1] = var_mean_4[0]
        getitem_9: f32[2, 2048, 1] = var_mean_4[1];  var_mean_4 = None
        add_17: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
        sub_8: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_41, getitem_9);  getitem_9 = None
        mul_11: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = rsqrt_4 = None
        mul_12: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_11, _frozen_param34);  mul_11 = _frozen_param34 = None
        add_18: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_12, _frozen_param35);  mul_12 = _frozen_param35 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_37: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None
        _frozen_param221: bf16[768] = self._frozen_param221
        view_42: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_37, [4096, 768]);  convert_element_type_37 = None
        _frozen_param222: bf16[768, 768] = self._frozen_param222
        _linear_pointwise_default_60: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_42, _frozen_param222, _frozen_param221, 'none', [], '');  _frozen_param222 = _frozen_param221 = None
        view_43: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_60, [2, 2048, 768]);  _linear_pointwise_default_60 = None
        mul_13: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_43, 0.125);  view_43 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param223: bf16[768] = self._frozen_param223
        _frozen_param224: bf16[768, 768] = self._frozen_param224
        _linear_pointwise_default_59: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_42, _frozen_param224, _frozen_param223, 'none', [], '');  _frozen_param224 = _frozen_param223 = None
        view_45: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_59, [2, 2048, 768]);  _linear_pointwise_default_59 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_46: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_45, [2, -1, 12, 64]);  view_45 = None
        permute_24: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        clone_14: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param225: bf16[768] = self._frozen_param225
        _frozen_param226: bf16[768, 768] = self._frozen_param226
        _linear_pointwise_default_58: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_42, _frozen_param226, _frozen_param225, 'none', [], '');  view_42 = _frozen_param226 = _frozen_param225 = None
        view_48: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_58, [2, 2048, 768]);  _linear_pointwise_default_58 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_49: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_48, [2, -1, 12, 64]);  view_48 = None
        permute_26: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
        clone_15: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_50: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_13, [2, 2048, 12, 64]);  mul_13 = None
        permute_27: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        clone_16: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_51: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_16, [24, -1, 64]);  clone_16 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_52: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_14, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_53: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_15, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_28: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
        bmm_4: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_51, permute_28);  view_51 = permute_28 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_54: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_4, [2, 12, 2048, 2048]);  bmm_4 = None
        add_19: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_54, where_2);  view_54 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_2: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_19, full_default_3);  add_19 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_55: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_2, [24, 2048, 2048]);  maximum_2 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_2: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_55, [-1], True)
        sub_9: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_55, amax_2);  view_55 = amax_2 = None
        exp_2: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_17: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_2);  div_2 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_46: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_17, torch.bfloat16);  clone_17 = None
        bmm_5: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_46, view_53);  convert_element_type_46 = view_53 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_56: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_5, [2, 12, 2048, 64]);  bmm_5 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_29: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_18: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_57: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_18, [2, 2048, 768]);  clone_18 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param227: bf16[768] = self._frozen_param227
        view_58: bf16[4096, 768] = torch.ops.aten.reshape.default(view_57, [4096, 768]);  view_57 = None
        _frozen_param228: bf16[768, 768] = self._frozen_param228
        _linear_pointwise_default_57: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_58, _frozen_param228, _frozen_param227, 'none', [], '');  view_58 = _frozen_param228 = _frozen_param227 = None
        view_59: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_57, [2, 2048, 768]);  _linear_pointwise_default_57 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_19: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_59);  view_59 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_20: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_41, clone_19);  view_41 = clone_19 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_60: f32[4096, 768] = torch.ops.aten.reshape.default(add_20, [-1, 768]);  add_20 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(view_60, [1], correction = 0, keepdim = True)
        getitem_10: f32[4096, 1] = var_mean_5[0]
        getitem_11: f32[4096, 1] = var_mean_5[1];  var_mean_5 = None
        add_21: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_10: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_60, getitem_11);  getitem_11 = None
        mul_14: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_10, rsqrt_5);  sub_10 = rsqrt_5 = None
        mul_15: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_14, _frozen_param44);  mul_14 = _frozen_param44 = None
        add_22: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_15, _frozen_param45);  mul_15 = _frozen_param45 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_49: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None
        _frozen_param229: bf16[3072] = self._frozen_param229
        _frozen_param230: bf16[768, 3072] = self._frozen_param230
        _linear_pointwise_default_56: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_49, _frozen_param230, _frozen_param229, 'none', [], '');  convert_element_type_49 = _frozen_param230 = _frozen_param229 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_2: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_56);  _linear_pointwise_default_56 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param231: bf16[768] = self._frozen_param231
        _frozen_param232: bf16[3072, 768] = self._frozen_param232
        _linear_pointwise_default_55: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_2, _frozen_param232, _frozen_param231, 'none', [], '');  relu_2 = _frozen_param232 = _frozen_param231 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_20: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_55);  _linear_pointwise_default_55 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_23: f32[4096, 768] = torch.ops.aten.add.Tensor(view_60, clone_20);  view_60 = clone_20 = None
        view_61: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_23, [2, 2048, 768]);  add_23 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(view_61, [2], correction = 0, keepdim = True)
        getitem_12: f32[2, 2048, 1] = var_mean_6[0]
        getitem_13: f32[2, 2048, 1] = var_mean_6[1];  var_mean_6 = None
        add_24: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
        rsqrt_6: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
        sub_11: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_61, getitem_13);  getitem_13 = None
        mul_16: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_11, rsqrt_6);  sub_11 = rsqrt_6 = None
        mul_17: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_16, _frozen_param50);  mul_16 = _frozen_param50 = None
        add_25: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_17, _frozen_param51);  mul_17 = _frozen_param51 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_54: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_25, torch.bfloat16);  add_25 = None
        _frozen_param233: bf16[768] = self._frozen_param233
        view_62: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_54, [4096, 768]);  convert_element_type_54 = None
        _frozen_param234: bf16[768, 768] = self._frozen_param234
        _linear_pointwise_default_54: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_62, _frozen_param234, _frozen_param233, 'none', [], '');  _frozen_param234 = _frozen_param233 = None
        view_63: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_54, [2, 2048, 768]);  _linear_pointwise_default_54 = None
        mul_18: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_63, 0.125);  view_63 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param235: bf16[768] = self._frozen_param235
        _frozen_param236: bf16[768, 768] = self._frozen_param236
        _linear_pointwise_default_53: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_62, _frozen_param236, _frozen_param235, 'none', [], '');  _frozen_param236 = _frozen_param235 = None
        view_65: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_53, [2, 2048, 768]);  _linear_pointwise_default_53 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_66: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_65, [2, -1, 12, 64]);  view_65 = None
        permute_35: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        clone_21: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param237: bf16[768] = self._frozen_param237
        _frozen_param238: bf16[768, 768] = self._frozen_param238
        _linear_pointwise_default_52: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_62, _frozen_param238, _frozen_param237, 'none', [], '');  view_62 = _frozen_param238 = _frozen_param237 = None
        view_68: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_52, [2, 2048, 768]);  _linear_pointwise_default_52 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_69: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_68, [2, -1, 12, 64]);  view_68 = None
        permute_37: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
        clone_22: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_70: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_18, [2, 2048, 12, 64]);  mul_18 = None
        permute_38: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_23: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_71: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_23, [24, -1, 64]);  clone_23 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_72: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_21, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_73: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_22, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_39: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_72, [0, 2, 1]);  view_72 = None
        bmm_6: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_71, permute_39);  view_71 = permute_39 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_74: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_6, [2, 12, 2048, 2048]);  bmm_6 = None
        add_26: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_74, where_2);  view_74 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_3: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_26, full_default_3);  add_26 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_75: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_3, [24, 2048, 2048]);  maximum_3 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_3: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_75, [-1], True)
        sub_12: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_75, amax_3);  view_75 = amax_3 = None
        exp_3: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_4: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_24: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_3);  div_3 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_63: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_24, torch.bfloat16);  clone_24 = None
        bmm_7: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_63, view_73);  convert_element_type_63 = view_73 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_76: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_7, [2, 12, 2048, 64]);  bmm_7 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_40: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_25: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_77: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_25, [2, 2048, 768]);  clone_25 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param239: bf16[768] = self._frozen_param239
        view_78: bf16[4096, 768] = torch.ops.aten.reshape.default(view_77, [4096, 768]);  view_77 = None
        _frozen_param240: bf16[768, 768] = self._frozen_param240
        _linear_pointwise_default_51: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_78, _frozen_param240, _frozen_param239, 'none', [], '');  view_78 = _frozen_param240 = _frozen_param239 = None
        view_79: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_51, [2, 2048, 768]);  _linear_pointwise_default_51 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_26: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_79);  view_79 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_27: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_61, clone_26);  view_61 = clone_26 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_80: f32[4096, 768] = torch.ops.aten.reshape.default(add_27, [-1, 768]);  add_27 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(view_80, [1], correction = 0, keepdim = True)
        getitem_14: f32[4096, 1] = var_mean_7[0]
        getitem_15: f32[4096, 1] = var_mean_7[1];  var_mean_7 = None
        add_28: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
        rsqrt_7: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
        sub_13: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_80, getitem_15);  getitem_15 = None
        mul_19: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_13, rsqrt_7);  sub_13 = rsqrt_7 = None
        mul_20: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_19, _frozen_param60);  mul_19 = _frozen_param60 = None
        add_29: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_20, _frozen_param61);  mul_20 = _frozen_param61 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_66: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16);  add_29 = None
        _frozen_param241: bf16[3072] = self._frozen_param241
        _frozen_param242: bf16[768, 3072] = self._frozen_param242
        _linear_pointwise_default_50: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_66, _frozen_param242, _frozen_param241, 'none', [], '');  convert_element_type_66 = _frozen_param242 = _frozen_param241 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_3: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_50);  _linear_pointwise_default_50 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param243: bf16[768] = self._frozen_param243
        _frozen_param244: bf16[3072, 768] = self._frozen_param244
        _linear_pointwise_default_49: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_3, _frozen_param244, _frozen_param243, 'none', [], '');  relu_3 = _frozen_param244 = _frozen_param243 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_27: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_49);  _linear_pointwise_default_49 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_30: f32[4096, 768] = torch.ops.aten.add.Tensor(view_80, clone_27);  view_80 = clone_27 = None
        view_81: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_30, [2, 2048, 768]);  add_30 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(view_81, [2], correction = 0, keepdim = True)
        getitem_16: f32[2, 2048, 1] = var_mean_8[0]
        getitem_17: f32[2, 2048, 1] = var_mean_8[1];  var_mean_8 = None
        add_31: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_8: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_14: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_81, getitem_17);  getitem_17 = None
        mul_21: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_14, rsqrt_8);  sub_14 = rsqrt_8 = None
        mul_22: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_21, _frozen_param66);  mul_21 = _frozen_param66 = None
        add_32: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_22, _frozen_param67);  mul_22 = _frozen_param67 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_71: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None
        _frozen_param245: bf16[768] = self._frozen_param245
        view_82: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_71, [4096, 768]);  convert_element_type_71 = None
        _frozen_param246: bf16[768, 768] = self._frozen_param246
        _linear_pointwise_default_48: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_82, _frozen_param246, _frozen_param245, 'none', [], '');  _frozen_param246 = _frozen_param245 = None
        view_83: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_48, [2, 2048, 768]);  _linear_pointwise_default_48 = None
        mul_23: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_83, 0.125);  view_83 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param247: bf16[768] = self._frozen_param247
        _frozen_param248: bf16[768, 768] = self._frozen_param248
        _linear_pointwise_default_47: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_82, _frozen_param248, _frozen_param247, 'none', [], '');  _frozen_param248 = _frozen_param247 = None
        view_85: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_47, [2, 2048, 768]);  _linear_pointwise_default_47 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_86: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_85, [2, -1, 12, 64]);  view_85 = None
        permute_46: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        clone_28: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param249: bf16[768] = self._frozen_param249
        _frozen_param250: bf16[768, 768] = self._frozen_param250
        _linear_pointwise_default_46: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_82, _frozen_param250, _frozen_param249, 'none', [], '');  view_82 = _frozen_param250 = _frozen_param249 = None
        view_88: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_46, [2, 2048, 768]);  _linear_pointwise_default_46 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_89: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_88, [2, -1, 12, 64]);  view_88 = None
        permute_48: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
        clone_29: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_90: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_23, [2, 2048, 12, 64]);  mul_23 = None
        permute_49: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_30: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_91: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_30, [24, -1, 64]);  clone_30 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_92: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_28, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_93: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_29, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_50: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
        bmm_8: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_91, permute_50);  view_91 = permute_50 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_94: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_8, [2, 12, 2048, 2048]);  bmm_8 = None
        add_33: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_94, where_2);  view_94 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_4: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_33, full_default_3);  add_33 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_95: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_4, [24, 2048, 2048]);  maximum_4 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_4: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_95, [-1], True)
        sub_15: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_95, amax_4);  view_95 = amax_4 = None
        exp_4: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_5: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_31: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_4);  div_4 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_80: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_31, torch.bfloat16);  clone_31 = None
        bmm_9: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_80, view_93);  convert_element_type_80 = view_93 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_96: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_9, [2, 12, 2048, 64]);  bmm_9 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_51: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_32: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
        view_97: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_32, [2, 2048, 768]);  clone_32 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param251: bf16[768] = self._frozen_param251
        view_98: bf16[4096, 768] = torch.ops.aten.reshape.default(view_97, [4096, 768]);  view_97 = None
        _frozen_param252: bf16[768, 768] = self._frozen_param252
        _linear_pointwise_default_45: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_98, _frozen_param252, _frozen_param251, 'none', [], '');  view_98 = _frozen_param252 = _frozen_param251 = None
        view_99: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_45, [2, 2048, 768]);  _linear_pointwise_default_45 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_33: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_99);  view_99 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_34: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_81, clone_33);  view_81 = clone_33 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_100: f32[4096, 768] = torch.ops.aten.reshape.default(add_34, [-1, 768]);  add_34 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(view_100, [1], correction = 0, keepdim = True)
        getitem_18: f32[4096, 1] = var_mean_9[0]
        getitem_19: f32[4096, 1] = var_mean_9[1];  var_mean_9 = None
        add_35: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_9: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
        sub_16: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_100, getitem_19);  getitem_19 = None
        mul_24: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_16, rsqrt_9);  sub_16 = rsqrt_9 = None
        mul_25: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_24, _frozen_param76);  mul_24 = _frozen_param76 = None
        add_36: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_25, _frozen_param77);  mul_25 = _frozen_param77 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_83: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16);  add_36 = None
        _frozen_param253: bf16[3072] = self._frozen_param253
        _frozen_param254: bf16[768, 3072] = self._frozen_param254
        _linear_pointwise_default_44: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_83, _frozen_param254, _frozen_param253, 'none', [], '');  convert_element_type_83 = _frozen_param254 = _frozen_param253 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_4: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_44);  _linear_pointwise_default_44 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param255: bf16[768] = self._frozen_param255
        _frozen_param256: bf16[3072, 768] = self._frozen_param256
        _linear_pointwise_default_43: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_4, _frozen_param256, _frozen_param255, 'none', [], '');  relu_4 = _frozen_param256 = _frozen_param255 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_34: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_43);  _linear_pointwise_default_43 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_37: f32[4096, 768] = torch.ops.aten.add.Tensor(view_100, clone_34);  view_100 = clone_34 = None
        view_101: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_37, [2, 2048, 768]);  add_37 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(view_101, [2], correction = 0, keepdim = True)
        getitem_20: f32[2, 2048, 1] = var_mean_10[0]
        getitem_21: f32[2, 2048, 1] = var_mean_10[1];  var_mean_10 = None
        add_38: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_10: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_17: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_101, getitem_21);  getitem_21 = None
        mul_26: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_17, rsqrt_10);  sub_17 = rsqrt_10 = None
        mul_27: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_26, _frozen_param82);  mul_26 = _frozen_param82 = None
        add_39: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_27, _frozen_param83);  mul_27 = _frozen_param83 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_88: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16);  add_39 = None
        _frozen_param257: bf16[768] = self._frozen_param257
        view_102: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_88, [4096, 768]);  convert_element_type_88 = None
        _frozen_param258: bf16[768, 768] = self._frozen_param258
        _linear_pointwise_default_42: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_102, _frozen_param258, _frozen_param257, 'none', [], '');  _frozen_param258 = _frozen_param257 = None
        view_103: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_42, [2, 2048, 768]);  _linear_pointwise_default_42 = None
        mul_28: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_103, 0.125);  view_103 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param259: bf16[768] = self._frozen_param259
        _frozen_param260: bf16[768, 768] = self._frozen_param260
        _linear_pointwise_default_41: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_102, _frozen_param260, _frozen_param259, 'none', [], '');  _frozen_param260 = _frozen_param259 = None
        view_105: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_41, [2, 2048, 768]);  _linear_pointwise_default_41 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_106: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_105, [2, -1, 12, 64]);  view_105 = None
        permute_57: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_35: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param261: bf16[768] = self._frozen_param261
        _frozen_param262: bf16[768, 768] = self._frozen_param262
        _linear_pointwise_default_40: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_102, _frozen_param262, _frozen_param261, 'none', [], '');  view_102 = _frozen_param262 = _frozen_param261 = None
        view_108: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_40, [2, 2048, 768]);  _linear_pointwise_default_40 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_109: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_108, [2, -1, 12, 64]);  view_108 = None
        permute_59: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
        clone_36: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_110: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_28, [2, 2048, 12, 64]);  mul_28 = None
        permute_60: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
        clone_37: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_111: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_37, [24, -1, 64]);  clone_37 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_112: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_35, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_113: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_36, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_61: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_112, [0, 2, 1]);  view_112 = None
        bmm_10: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_111, permute_61);  view_111 = permute_61 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_114: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_10, [2, 12, 2048, 2048]);  bmm_10 = None
        add_40: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_114, where_2);  view_114 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_5: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_40, full_default_3);  add_40 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_115: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_5, [24, 2048, 2048]);  maximum_5 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_5: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_115, [-1], True)
        sub_18: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_115, amax_5);  view_115 = amax_5 = None
        exp_5: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_6: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_38: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_5);  div_5 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_97: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_38, torch.bfloat16);  clone_38 = None
        bmm_11: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_97, view_113);  convert_element_type_97 = view_113 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_116: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_11, [2, 12, 2048, 64]);  bmm_11 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_62: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_39: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_117: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_39, [2, 2048, 768]);  clone_39 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param263: bf16[768] = self._frozen_param263
        view_118: bf16[4096, 768] = torch.ops.aten.reshape.default(view_117, [4096, 768]);  view_117 = None
        _frozen_param264: bf16[768, 768] = self._frozen_param264
        _linear_pointwise_default_39: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_118, _frozen_param264, _frozen_param263, 'none', [], '');  view_118 = _frozen_param264 = _frozen_param263 = None
        view_119: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_39, [2, 2048, 768]);  _linear_pointwise_default_39 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_40: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_119);  view_119 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_41: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_101, clone_40);  view_101 = clone_40 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_120: f32[4096, 768] = torch.ops.aten.reshape.default(add_41, [-1, 768]);  add_41 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(view_120, [1], correction = 0, keepdim = True)
        getitem_22: f32[4096, 1] = var_mean_11[0]
        getitem_23: f32[4096, 1] = var_mean_11[1];  var_mean_11 = None
        add_42: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
        rsqrt_11: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_19: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_120, getitem_23);  getitem_23 = None
        mul_29: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_19, rsqrt_11);  sub_19 = rsqrt_11 = None
        mul_30: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_29, _frozen_param92);  mul_29 = _frozen_param92 = None
        add_43: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_30, _frozen_param93);  mul_30 = _frozen_param93 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_100: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16);  add_43 = None
        _frozen_param265: bf16[3072] = self._frozen_param265
        _frozen_param266: bf16[768, 3072] = self._frozen_param266
        _linear_pointwise_default_38: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_100, _frozen_param266, _frozen_param265, 'none', [], '');  convert_element_type_100 = _frozen_param266 = _frozen_param265 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_5: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_38);  _linear_pointwise_default_38 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param267: bf16[768] = self._frozen_param267
        _frozen_param268: bf16[3072, 768] = self._frozen_param268
        _linear_pointwise_default_37: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_5, _frozen_param268, _frozen_param267, 'none', [], '');  relu_5 = _frozen_param268 = _frozen_param267 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_41: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_37);  _linear_pointwise_default_37 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_44: f32[4096, 768] = torch.ops.aten.add.Tensor(view_120, clone_41);  view_120 = clone_41 = None
        view_121: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_44, [2, 2048, 768]);  add_44 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(view_121, [2], correction = 0, keepdim = True)
        getitem_24: f32[2, 2048, 1] = var_mean_12[0]
        getitem_25: f32[2, 2048, 1] = var_mean_12[1];  var_mean_12 = None
        add_45: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
        rsqrt_12: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        sub_20: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_121, getitem_25);  getitem_25 = None
        mul_31: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_20, rsqrt_12);  sub_20 = rsqrt_12 = None
        mul_32: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_31, _frozen_param98);  mul_31 = _frozen_param98 = None
        add_46: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_32, _frozen_param99);  mul_32 = _frozen_param99 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_105: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None
        _frozen_param269: bf16[768] = self._frozen_param269
        view_122: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_105, [4096, 768]);  convert_element_type_105 = None
        _frozen_param270: bf16[768, 768] = self._frozen_param270
        _linear_pointwise_default_36: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_122, _frozen_param270, _frozen_param269, 'none', [], '');  _frozen_param270 = _frozen_param269 = None
        view_123: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_36, [2, 2048, 768]);  _linear_pointwise_default_36 = None
        mul_33: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_123, 0.125);  view_123 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param271: bf16[768] = self._frozen_param271
        _frozen_param272: bf16[768, 768] = self._frozen_param272
        _linear_pointwise_default_35: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_122, _frozen_param272, _frozen_param271, 'none', [], '');  _frozen_param272 = _frozen_param271 = None
        view_125: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_35, [2, 2048, 768]);  _linear_pointwise_default_35 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_126: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_125, [2, -1, 12, 64]);  view_125 = None
        permute_68: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_42: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param273: bf16[768] = self._frozen_param273
        _frozen_param274: bf16[768, 768] = self._frozen_param274
        _linear_pointwise_default_34: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_122, _frozen_param274, _frozen_param273, 'none', [], '');  view_122 = _frozen_param274 = _frozen_param273 = None
        view_128: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_34, [2, 2048, 768]);  _linear_pointwise_default_34 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_129: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_128, [2, -1, 12, 64]);  view_128 = None
        permute_70: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_43: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_130: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_33, [2, 2048, 12, 64]);  mul_33 = None
        permute_71: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        clone_44: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_131: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_44, [24, -1, 64]);  clone_44 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_132: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_42, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_133: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_43, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_72: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
        bmm_12: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_131, permute_72);  view_131 = permute_72 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_134: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_12, [2, 12, 2048, 2048]);  bmm_12 = None
        add_47: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_134, where_2);  view_134 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_6: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_47, full_default_3);  add_47 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_135: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_6, [24, 2048, 2048]);  maximum_6 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_6: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_135, [-1], True)
        sub_21: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_135, amax_6);  view_135 = amax_6 = None
        exp_6: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_7: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_45: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_6);  div_6 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_114: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_45, torch.bfloat16);  clone_45 = None
        bmm_13: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_114, view_133);  convert_element_type_114 = view_133 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_136: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_13, [2, 12, 2048, 64]);  bmm_13 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_73: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_46: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_137: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_46, [2, 2048, 768]);  clone_46 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param275: bf16[768] = self._frozen_param275
        view_138: bf16[4096, 768] = torch.ops.aten.reshape.default(view_137, [4096, 768]);  view_137 = None
        _frozen_param276: bf16[768, 768] = self._frozen_param276
        _linear_pointwise_default_33: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_138, _frozen_param276, _frozen_param275, 'none', [], '');  view_138 = _frozen_param276 = _frozen_param275 = None
        view_139: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_33, [2, 2048, 768]);  _linear_pointwise_default_33 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_47: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_139);  view_139 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_48: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_121, clone_47);  view_121 = clone_47 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_140: f32[4096, 768] = torch.ops.aten.reshape.default(add_48, [-1, 768]);  add_48 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(view_140, [1], correction = 0, keepdim = True)
        getitem_26: f32[4096, 1] = var_mean_13[0]
        getitem_27: f32[4096, 1] = var_mean_13[1];  var_mean_13 = None
        add_49: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_13: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_22: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_140, getitem_27);  getitem_27 = None
        mul_34: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_22, rsqrt_13);  sub_22 = rsqrt_13 = None
        mul_35: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_34, _frozen_param108);  mul_34 = _frozen_param108 = None
        add_50: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_35, _frozen_param109);  mul_35 = _frozen_param109 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_117: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16);  add_50 = None
        _frozen_param277: bf16[3072] = self._frozen_param277
        _frozen_param278: bf16[768, 3072] = self._frozen_param278
        _linear_pointwise_default_32: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_117, _frozen_param278, _frozen_param277, 'none', [], '');  convert_element_type_117 = _frozen_param278 = _frozen_param277 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_6: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_32);  _linear_pointwise_default_32 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param279: bf16[768] = self._frozen_param279
        _frozen_param280: bf16[3072, 768] = self._frozen_param280
        _linear_pointwise_default_31: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_6, _frozen_param280, _frozen_param279, 'none', [], '');  relu_6 = _frozen_param280 = _frozen_param279 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_48: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_31);  _linear_pointwise_default_31 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_51: f32[4096, 768] = torch.ops.aten.add.Tensor(view_140, clone_48);  view_140 = clone_48 = None
        view_141: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_51, [2, 2048, 768]);  add_51 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(view_141, [2], correction = 0, keepdim = True)
        getitem_28: f32[2, 2048, 1] = var_mean_14[0]
        getitem_29: f32[2, 2048, 1] = var_mean_14[1];  var_mean_14 = None
        add_52: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_14: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        sub_23: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_141, getitem_29);  getitem_29 = None
        mul_36: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_23, rsqrt_14);  sub_23 = rsqrt_14 = None
        mul_37: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_36, _frozen_param114);  mul_36 = _frozen_param114 = None
        add_53: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_37, _frozen_param115);  mul_37 = _frozen_param115 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_122: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16);  add_53 = None
        _frozen_param281: bf16[768] = self._frozen_param281
        view_142: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_122, [4096, 768]);  convert_element_type_122 = None
        _frozen_param282: bf16[768, 768] = self._frozen_param282
        _linear_pointwise_default_30: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_142, _frozen_param282, _frozen_param281, 'none', [], '');  _frozen_param282 = _frozen_param281 = None
        view_143: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_30, [2, 2048, 768]);  _linear_pointwise_default_30 = None
        mul_38: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_143, 0.125);  view_143 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param283: bf16[768] = self._frozen_param283
        _frozen_param284: bf16[768, 768] = self._frozen_param284
        _linear_pointwise_default_29: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_142, _frozen_param284, _frozen_param283, 'none', [], '');  _frozen_param284 = _frozen_param283 = None
        view_145: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_29, [2, 2048, 768]);  _linear_pointwise_default_29 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_146: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_145, [2, -1, 12, 64]);  view_145 = None
        permute_79: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_49: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param285: bf16[768] = self._frozen_param285
        _frozen_param286: bf16[768, 768] = self._frozen_param286
        _linear_pointwise_default_28: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_142, _frozen_param286, _frozen_param285, 'none', [], '');  view_142 = _frozen_param286 = _frozen_param285 = None
        view_148: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_28, [2, 2048, 768]);  _linear_pointwise_default_28 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_149: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_148, [2, -1, 12, 64]);  view_148 = None
        permute_81: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
        clone_50: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_150: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_38, [2, 2048, 12, 64]);  mul_38 = None
        permute_82: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
        clone_51: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_151: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_51, [24, -1, 64]);  clone_51 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_152: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_49, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_153: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_50, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_83: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
        bmm_14: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_151, permute_83);  view_151 = permute_83 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_154: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_14, [2, 12, 2048, 2048]);  bmm_14 = None
        add_54: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_154, where_2);  view_154 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_7: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_54, full_default_3);  add_54 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_155: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_7, [24, 2048, 2048]);  maximum_7 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_7: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_155, [-1], True)
        sub_24: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_155, amax_7);  view_155 = amax_7 = None
        exp_7: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_24);  sub_24 = None
        sum_8: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_52: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_7);  div_7 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_131: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_52, torch.bfloat16);  clone_52 = None
        bmm_15: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_131, view_153);  convert_element_type_131 = view_153 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_156: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_15, [2, 12, 2048, 64]);  bmm_15 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_84: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_53: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_157: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_53, [2, 2048, 768]);  clone_53 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param287: bf16[768] = self._frozen_param287
        view_158: bf16[4096, 768] = torch.ops.aten.reshape.default(view_157, [4096, 768]);  view_157 = None
        _frozen_param288: bf16[768, 768] = self._frozen_param288
        _linear_pointwise_default_27: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_158, _frozen_param288, _frozen_param287, 'none', [], '');  view_158 = _frozen_param288 = _frozen_param287 = None
        view_159: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_27, [2, 2048, 768]);  _linear_pointwise_default_27 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_54: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_159);  view_159 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_55: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_141, clone_54);  view_141 = clone_54 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_160: f32[4096, 768] = torch.ops.aten.reshape.default(add_55, [-1, 768]);  add_55 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(view_160, [1], correction = 0, keepdim = True)
        getitem_30: f32[4096, 1] = var_mean_15[0]
        getitem_31: f32[4096, 1] = var_mean_15[1];  var_mean_15 = None
        add_56: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_15: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        sub_25: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_160, getitem_31);  getitem_31 = None
        mul_39: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_25, rsqrt_15);  sub_25 = rsqrt_15 = None
        mul_40: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_39, _frozen_param124);  mul_39 = _frozen_param124 = None
        add_57: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_40, _frozen_param125);  mul_40 = _frozen_param125 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_134: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16);  add_57 = None
        _frozen_param289: bf16[3072] = self._frozen_param289
        _frozen_param290: bf16[768, 3072] = self._frozen_param290
        _linear_pointwise_default_26: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_134, _frozen_param290, _frozen_param289, 'none', [], '');  convert_element_type_134 = _frozen_param290 = _frozen_param289 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_7: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_26);  _linear_pointwise_default_26 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param291: bf16[768] = self._frozen_param291
        _frozen_param292: bf16[3072, 768] = self._frozen_param292
        _linear_pointwise_default_25: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_7, _frozen_param292, _frozen_param291, 'none', [], '');  relu_7 = _frozen_param292 = _frozen_param291 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_55: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_25);  _linear_pointwise_default_25 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_58: f32[4096, 768] = torch.ops.aten.add.Tensor(view_160, clone_55);  view_160 = clone_55 = None
        view_161: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_58, [2, 2048, 768]);  add_58 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(view_161, [2], correction = 0, keepdim = True)
        getitem_32: f32[2, 2048, 1] = var_mean_16[0]
        getitem_33: f32[2, 2048, 1] = var_mean_16[1];  var_mean_16 = None
        add_59: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
        rsqrt_16: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
        sub_26: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_161, getitem_33);  getitem_33 = None
        mul_41: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_26, rsqrt_16);  sub_26 = rsqrt_16 = None
        mul_42: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_41, _frozen_param130);  mul_41 = _frozen_param130 = None
        add_60: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_42, _frozen_param131);  mul_42 = _frozen_param131 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_139: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_60, torch.bfloat16);  add_60 = None
        _frozen_param293: bf16[768] = self._frozen_param293
        view_162: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_139, [4096, 768]);  convert_element_type_139 = None
        _frozen_param294: bf16[768, 768] = self._frozen_param294
        _linear_pointwise_default_24: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_162, _frozen_param294, _frozen_param293, 'none', [], '');  _frozen_param294 = _frozen_param293 = None
        view_163: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_24, [2, 2048, 768]);  _linear_pointwise_default_24 = None
        mul_43: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_163, 0.125);  view_163 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param295: bf16[768] = self._frozen_param295
        _frozen_param296: bf16[768, 768] = self._frozen_param296
        _linear_pointwise_default_23: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_162, _frozen_param296, _frozen_param295, 'none', [], '');  _frozen_param296 = _frozen_param295 = None
        view_165: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_23, [2, 2048, 768]);  _linear_pointwise_default_23 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_166: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_165, [2, -1, 12, 64]);  view_165 = None
        permute_90: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        clone_56: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param297: bf16[768] = self._frozen_param297
        _frozen_param298: bf16[768, 768] = self._frozen_param298
        _linear_pointwise_default_22: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_162, _frozen_param298, _frozen_param297, 'none', [], '');  view_162 = _frozen_param298 = _frozen_param297 = None
        view_168: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_22, [2, 2048, 768]);  _linear_pointwise_default_22 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_169: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_168, [2, -1, 12, 64]);  view_168 = None
        permute_92: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_169, [0, 2, 1, 3]);  view_169 = None
        clone_57: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_170: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_43, [2, 2048, 12, 64]);  mul_43 = None
        permute_93: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_58: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_171: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_58, [24, -1, 64]);  clone_58 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_172: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_56, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_173: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_57, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_94: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
        bmm_16: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_171, permute_94);  view_171 = permute_94 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_174: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_16, [2, 12, 2048, 2048]);  bmm_16 = None
        add_61: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_174, where_2);  view_174 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_8: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_61, full_default_3);  add_61 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_175: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_8, [24, 2048, 2048]);  maximum_8 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_8: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_175, [-1], True)
        sub_27: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_175, amax_8);  view_175 = amax_8 = None
        exp_8: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_9: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_59: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_8);  div_8 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_148: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_59, torch.bfloat16);  clone_59 = None
        bmm_17: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_148, view_173);  convert_element_type_148 = view_173 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_176: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_17, [2, 12, 2048, 64]);  bmm_17 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_95: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_60: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_177: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_60, [2, 2048, 768]);  clone_60 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param299: bf16[768] = self._frozen_param299
        view_178: bf16[4096, 768] = torch.ops.aten.reshape.default(view_177, [4096, 768]);  view_177 = None
        _frozen_param300: bf16[768, 768] = self._frozen_param300
        _linear_pointwise_default_21: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_178, _frozen_param300, _frozen_param299, 'none', [], '');  view_178 = _frozen_param300 = _frozen_param299 = None
        view_179: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_21, [2, 2048, 768]);  _linear_pointwise_default_21 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_61: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_179);  view_179 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_62: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_161, clone_61);  view_161 = clone_61 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_180: f32[4096, 768] = torch.ops.aten.reshape.default(add_62, [-1, 768]);  add_62 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(view_180, [1], correction = 0, keepdim = True)
        getitem_34: f32[4096, 1] = var_mean_17[0]
        getitem_35: f32[4096, 1] = var_mean_17[1];  var_mean_17 = None
        add_63: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
        rsqrt_17: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        sub_28: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_180, getitem_35);  getitem_35 = None
        mul_44: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_28, rsqrt_17);  sub_28 = rsqrt_17 = None
        mul_45: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_44, _frozen_param140);  mul_44 = _frozen_param140 = None
        add_64: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_45, _frozen_param141);  mul_45 = _frozen_param141 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_151: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16);  add_64 = None
        _frozen_param301: bf16[3072] = self._frozen_param301
        _frozen_param302: bf16[768, 3072] = self._frozen_param302
        _linear_pointwise_default_20: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_151, _frozen_param302, _frozen_param301, 'none', [], '');  convert_element_type_151 = _frozen_param302 = _frozen_param301 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_8: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_20);  _linear_pointwise_default_20 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param303: bf16[768] = self._frozen_param303
        _frozen_param304: bf16[3072, 768] = self._frozen_param304
        _linear_pointwise_default_19: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_8, _frozen_param304, _frozen_param303, 'none', [], '');  relu_8 = _frozen_param304 = _frozen_param303 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_62: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_19);  _linear_pointwise_default_19 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_65: f32[4096, 768] = torch.ops.aten.add.Tensor(view_180, clone_62);  view_180 = clone_62 = None
        view_181: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_65, [2, 2048, 768]);  add_65 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(view_181, [2], correction = 0, keepdim = True)
        getitem_36: f32[2, 2048, 1] = var_mean_18[0]
        getitem_37: f32[2, 2048, 1] = var_mean_18[1];  var_mean_18 = None
        add_66: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_18: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
        sub_29: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_181, getitem_37);  getitem_37 = None
        mul_46: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_29, rsqrt_18);  sub_29 = rsqrt_18 = None
        mul_47: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_46, _frozen_param146);  mul_46 = _frozen_param146 = None
        add_67: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_47, _frozen_param147);  mul_47 = _frozen_param147 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_156: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None
        _frozen_param305: bf16[768] = self._frozen_param305
        view_182: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_156, [4096, 768]);  convert_element_type_156 = None
        _frozen_param306: bf16[768, 768] = self._frozen_param306
        _linear_pointwise_default_18: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_182, _frozen_param306, _frozen_param305, 'none', [], '');  _frozen_param306 = _frozen_param305 = None
        view_183: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_18, [2, 2048, 768]);  _linear_pointwise_default_18 = None
        mul_48: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_183, 0.125);  view_183 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param307: bf16[768] = self._frozen_param307
        _frozen_param308: bf16[768, 768] = self._frozen_param308
        _linear_pointwise_default_17: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_182, _frozen_param308, _frozen_param307, 'none', [], '');  _frozen_param308 = _frozen_param307 = None
        view_185: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_17, [2, 2048, 768]);  _linear_pointwise_default_17 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_186: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_185, [2, -1, 12, 64]);  view_185 = None
        permute_101: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
        clone_63: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param309: bf16[768] = self._frozen_param309
        _frozen_param310: bf16[768, 768] = self._frozen_param310
        _linear_pointwise_default_16: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_182, _frozen_param310, _frozen_param309, 'none', [], '');  view_182 = _frozen_param310 = _frozen_param309 = None
        view_188: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_16, [2, 2048, 768]);  _linear_pointwise_default_16 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_189: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_188, [2, -1, 12, 64]);  view_188 = None
        permute_103: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
        clone_64: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_190: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_48, [2, 2048, 12, 64]);  mul_48 = None
        permute_104: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_65: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_191: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_65, [24, -1, 64]);  clone_65 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_192: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_63, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_193: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_64, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_105: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
        bmm_18: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_191, permute_105);  view_191 = permute_105 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_194: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_18, [2, 12, 2048, 2048]);  bmm_18 = None
        add_68: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_194, where_2);  view_194 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_9: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_68, full_default_3);  add_68 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_195: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_9, [24, 2048, 2048]);  maximum_9 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_9: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_195, [-1], True)
        sub_30: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_195, amax_9);  view_195 = amax_9 = None
        exp_9: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        sum_10: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_66: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_9);  div_9 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_165: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_66, torch.bfloat16);  clone_66 = None
        bmm_19: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_165, view_193);  convert_element_type_165 = view_193 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_196: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_19, [2, 12, 2048, 64]);  bmm_19 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_106: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_67: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_197: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_67, [2, 2048, 768]);  clone_67 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param311: bf16[768] = self._frozen_param311
        view_198: bf16[4096, 768] = torch.ops.aten.reshape.default(view_197, [4096, 768]);  view_197 = None
        _frozen_param312: bf16[768, 768] = self._frozen_param312
        _linear_pointwise_default_15: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_198, _frozen_param312, _frozen_param311, 'none', [], '');  view_198 = _frozen_param312 = _frozen_param311 = None
        view_199: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_15, [2, 2048, 768]);  _linear_pointwise_default_15 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_68: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_199);  view_199 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_69: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_181, clone_68);  view_181 = clone_68 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_200: f32[4096, 768] = torch.ops.aten.reshape.default(add_69, [-1, 768]);  add_69 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(view_200, [1], correction = 0, keepdim = True)
        getitem_38: f32[4096, 1] = var_mean_19[0]
        getitem_39: f32[4096, 1] = var_mean_19[1];  var_mean_19 = None
        add_70: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_19: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
        sub_31: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_200, getitem_39);  getitem_39 = None
        mul_49: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_31, rsqrt_19);  sub_31 = rsqrt_19 = None
        mul_50: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_49, _frozen_param156);  mul_49 = _frozen_param156 = None
        add_71: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_50, _frozen_param157);  mul_50 = _frozen_param157 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_168: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16);  add_71 = None
        _frozen_param313: bf16[3072] = self._frozen_param313
        _frozen_param314: bf16[768, 3072] = self._frozen_param314
        _linear_pointwise_default_14: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_168, _frozen_param314, _frozen_param313, 'none', [], '');  convert_element_type_168 = _frozen_param314 = _frozen_param313 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_9: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_14);  _linear_pointwise_default_14 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param315: bf16[768] = self._frozen_param315
        _frozen_param316: bf16[3072, 768] = self._frozen_param316
        _linear_pointwise_default_13: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_9, _frozen_param316, _frozen_param315, 'none', [], '');  relu_9 = _frozen_param316 = _frozen_param315 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_69: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_13);  _linear_pointwise_default_13 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_72: f32[4096, 768] = torch.ops.aten.add.Tensor(view_200, clone_69);  view_200 = clone_69 = None
        view_201: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_72, [2, 2048, 768]);  add_72 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(view_201, [2], correction = 0, keepdim = True)
        getitem_40: f32[2, 2048, 1] = var_mean_20[0]
        getitem_41: f32[2, 2048, 1] = var_mean_20[1];  var_mean_20 = None
        add_73: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_20: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
        sub_32: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_201, getitem_41);  getitem_41 = None
        mul_51: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_32, rsqrt_20);  sub_32 = rsqrt_20 = None
        mul_52: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_51, _frozen_param162);  mul_51 = _frozen_param162 = None
        add_74: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_52, _frozen_param163);  mul_52 = _frozen_param163 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_173: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None
        _frozen_param317: bf16[768] = self._frozen_param317
        view_202: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_173, [4096, 768]);  convert_element_type_173 = None
        _frozen_param318: bf16[768, 768] = self._frozen_param318
        _linear_pointwise_default_12: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_202, _frozen_param318, _frozen_param317, 'none', [], '');  _frozen_param318 = _frozen_param317 = None
        view_203: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_12, [2, 2048, 768]);  _linear_pointwise_default_12 = None
        mul_53: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_203, 0.125);  view_203 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param319: bf16[768] = self._frozen_param319
        _frozen_param320: bf16[768, 768] = self._frozen_param320
        _linear_pointwise_default_11: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_202, _frozen_param320, _frozen_param319, 'none', [], '');  _frozen_param320 = _frozen_param319 = None
        view_205: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_11, [2, 2048, 768]);  _linear_pointwise_default_11 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_206: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_205, [2, -1, 12, 64]);  view_205 = None
        permute_112: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        clone_70: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param321: bf16[768] = self._frozen_param321
        _frozen_param322: bf16[768, 768] = self._frozen_param322
        _linear_pointwise_default_10: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_202, _frozen_param322, _frozen_param321, 'none', [], '');  view_202 = _frozen_param322 = _frozen_param321 = None
        view_208: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_10, [2, 2048, 768]);  _linear_pointwise_default_10 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_209: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_208, [2, -1, 12, 64]);  view_208 = None
        permute_114: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
        clone_71: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_210: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_53, [2, 2048, 12, 64]);  mul_53 = None
        permute_115: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
        clone_72: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_211: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_72, [24, -1, 64]);  clone_72 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_212: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_70, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_213: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_71, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_116: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_212, [0, 2, 1]);  view_212 = None
        bmm_20: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_211, permute_116);  view_211 = permute_116 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_214: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_20, [2, 12, 2048, 2048]);  bmm_20 = None
        add_75: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_214, where_2);  view_214 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_10: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_75, full_default_3);  add_75 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_215: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_10, [24, 2048, 2048]);  maximum_10 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_10: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_215, [-1], True)
        sub_33: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_215, amax_10);  view_215 = amax_10 = None
        exp_10: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_11: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_73: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_10);  div_10 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_182: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_73, torch.bfloat16);  clone_73 = None
        bmm_21: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_182, view_213);  convert_element_type_182 = view_213 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_216: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_21, [2, 12, 2048, 64]);  bmm_21 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_117: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_74: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_217: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_74, [2, 2048, 768]);  clone_74 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param323: bf16[768] = self._frozen_param323
        view_218: bf16[4096, 768] = torch.ops.aten.reshape.default(view_217, [4096, 768]);  view_217 = None
        _frozen_param324: bf16[768, 768] = self._frozen_param324
        _linear_pointwise_default_9: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_218, _frozen_param324, _frozen_param323, 'none', [], '');  view_218 = _frozen_param324 = _frozen_param323 = None
        view_219: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_9, [2, 2048, 768]);  _linear_pointwise_default_9 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_75: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_219);  view_219 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_76: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_201, clone_75);  view_201 = clone_75 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_220: f32[4096, 768] = torch.ops.aten.reshape.default(add_76, [-1, 768]);  add_76 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(view_220, [1], correction = 0, keepdim = True)
        getitem_42: f32[4096, 1] = var_mean_21[0]
        getitem_43: f32[4096, 1] = var_mean_21[1];  var_mean_21 = None
        add_77: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
        rsqrt_21: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
        sub_34: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_220, getitem_43);  getitem_43 = None
        mul_54: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_34, rsqrt_21);  sub_34 = rsqrt_21 = None
        mul_55: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_54, _frozen_param172);  mul_54 = _frozen_param172 = None
        add_78: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_55, _frozen_param173);  mul_55 = _frozen_param173 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_185: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None
        _frozen_param325: bf16[3072] = self._frozen_param325
        _frozen_param326: bf16[768, 3072] = self._frozen_param326
        _linear_pointwise_default_8: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_185, _frozen_param326, _frozen_param325, 'none', [], '');  convert_element_type_185 = _frozen_param326 = _frozen_param325 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_10: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_8);  _linear_pointwise_default_8 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param327: bf16[768] = self._frozen_param327
        _frozen_param328: bf16[3072, 768] = self._frozen_param328
        _linear_pointwise_default_7: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_10, _frozen_param328, _frozen_param327, 'none', [], '');  relu_10 = _frozen_param328 = _frozen_param327 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_76: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_7);  _linear_pointwise_default_7 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_79: f32[4096, 768] = torch.ops.aten.add.Tensor(view_220, clone_76);  view_220 = clone_76 = None
        view_221: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_79, [2, 2048, 768]);  add_79 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:549, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(view_221, [2], correction = 0, keepdim = True)
        getitem_44: f32[2, 2048, 1] = var_mean_22[0]
        getitem_45: f32[2, 2048, 1] = var_mean_22[1];  var_mean_22 = None
        add_80: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
        rsqrt_22: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_35: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_221, getitem_45);  getitem_45 = None
        mul_56: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_35, rsqrt_22);  sub_35 = rsqrt_22 = None
        mul_57: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_56, _frozen_param178);  mul_56 = _frozen_param178 = None
        add_81: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_57, _frozen_param179);  mul_57 = _frozen_param179 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:182, code: query_states = self.q_proj(hidden_states) * self.scaling
        convert_element_type_190: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_81, torch.bfloat16);  add_81 = None
        _frozen_param329: bf16[768] = self._frozen_param329
        view_222: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_190, [4096, 768]);  convert_element_type_190 = None
        _frozen_param330: bf16[768, 768] = self._frozen_param330
        _linear_pointwise_default_6: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_222, _frozen_param330, _frozen_param329, 'none', [], '');  _frozen_param330 = _frozen_param329 = None
        view_223: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_6, [2, 2048, 768]);  _linear_pointwise_default_6 = None
        mul_58: bf16[2, 2048, 768] = torch.ops.aten.mul.Tensor(view_223, 0.125);  view_223 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:200, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        _frozen_param331: bf16[768] = self._frozen_param331
        _frozen_param332: bf16[768, 768] = self._frozen_param332
        _linear_pointwise_default_5: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_222, _frozen_param332, _frozen_param331, 'none', [], '');  _frozen_param332 = _frozen_param331 = None
        view_225: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_5, [2, 2048, 768]);  _linear_pointwise_default_5 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_226: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_225, [2, -1, 12, 64]);  view_225 = None
        permute_123: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_77: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:201, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
        _frozen_param333: bf16[768] = self._frozen_param333
        _frozen_param334: bf16[768, 768] = self._frozen_param334
        _linear_pointwise_default_4: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_222, _frozen_param334, _frozen_param333, 'none', [], '');  view_222 = _frozen_param334 = _frozen_param333 = None
        view_228: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_4, [2, 2048, 768]);  _linear_pointwise_default_4 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_229: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(view_228, [2, -1, 12, 64]);  view_228 = None
        permute_125: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        clone_78: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:162, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        view_230: bf16[2, 2048, 12, 64] = torch.ops.aten.reshape.default(mul_58, [2, 2048, 12, 64]);  mul_58 = None
        permute_126: bf16[2, 12, 2048, 64] = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
        clone_79: bf16[2, 12, 2048, 64] = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:214, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        view_231: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_79, [24, -1, 64]);  clone_79 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:215, code: key_states = key_states.view(*proj_shape)
        view_232: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_77, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:216, code: value_states = value_states.view(*proj_shape)
        view_233: bf16[24, 2048, 64] = torch.ops.aten.reshape.default(clone_78, [24, -1, 64])
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:219, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
        permute_127: bf16[24, 64, 2048] = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
        bmm_22: bf16[24, 2048, 2048] = torch.ops.aten.bmm.default(view_231, permute_127);  view_231 = permute_127 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:232, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
        view_234: bf16[2, 12, 2048, 2048] = torch.ops.aten.reshape.default(bmm_22, [2, 12, 2048, 2048]);  bmm_22 = None
        add_82: f32[2, 12, 2048, 2048] = torch.ops.aten.add.Tensor(view_234, where_2);  view_234 = where_2 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:233, code: attn_weights = torch.max(
        maximum_11: f32[2, 12, 2048, 2048] = torch.ops.aten.maximum.default(add_82, full_default_3);  add_82 = full_default_3 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:236, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        view_235: f32[24, 2048, 2048] = torch.ops.aten.reshape.default(maximum_11, [24, 2048, 2048]);  maximum_11 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:242, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        amax_11: f32[24, 2048, 1] = torch.ops.aten.amax.default(view_235, [-1], True)
        sub_36: f32[24, 2048, 2048] = torch.ops.aten.sub.Tensor(view_235, amax_11);  view_235 = amax_11 = None
        exp_11: f32[24, 2048, 2048] = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        sum_12: f32[24, 2048, 1] = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: f32[24, 2048, 2048] = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:263, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        clone_80: f32[24, 2048, 2048] = torch.ops.aten.clone.default(div_11);  div_11 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:265, code: attn_output = torch.bmm(attn_probs, value_states)
        convert_element_type_199: bf16[24, 2048, 2048] = torch.ops.prims.convert_element_type.default(clone_80, torch.bfloat16);  clone_80 = None
        bmm_23: bf16[24, 2048, 64] = torch.ops.aten.bmm.default(convert_element_type_199, view_233);  convert_element_type_199 = view_233 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:273, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        view_236: bf16[2, 12, 2048, 64] = torch.ops.aten.reshape.default(bmm_23, [2, 12, 2048, 64]);  bmm_23 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:274, code: attn_output = attn_output.transpose(1, 2)
        permute_128: bf16[2, 2048, 12, 64] = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:278, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
        clone_81: bf16[2, 2048, 12, 64] = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_237: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(clone_81, [2, 2048, 768]);  clone_81 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:280, code: attn_output = self.out_proj(attn_output)
        _frozen_param335: bf16[768] = self._frozen_param335
        view_238: bf16[4096, 768] = torch.ops.aten.reshape.default(view_237, [4096, 768]);  view_237 = None
        _frozen_param336: bf16[768, 768] = self._frozen_param336
        _linear_pointwise_default_3: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(view_238, _frozen_param336, _frozen_param335, 'none', [], '');  view_238 = _frozen_param336 = _frozen_param335 = None
        view_239: bf16[2, 2048, 768] = torch.ops.aten.reshape.default(_linear_pointwise_default_3, [2, 2048, 768]);  _linear_pointwise_default_3 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:559, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_82: bf16[2, 2048, 768] = torch.ops.aten.clone.default(view_239);  view_239 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:560, code: hidden_states = residual + hidden_states
        add_83: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(view_221, clone_82);  view_221 = clone_82 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:568, code: hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        view_240: f32[4096, 768] = torch.ops.aten.reshape.default(add_83, [-1, 768]);  add_83 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:573, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(view_240, [1], correction = 0, keepdim = True)
        getitem_46: f32[4096, 1] = var_mean_23[0]
        getitem_47: f32[4096, 1] = var_mean_23[1];  var_mean_23 = None
        add_84: f32[4096, 1] = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_23: f32[4096, 1] = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
        sub_37: f32[4096, 768] = torch.ops.aten.sub.Tensor(view_240, getitem_47);  getitem_47 = None
        mul_59: f32[4096, 768] = torch.ops.aten.mul.Tensor(sub_37, rsqrt_23);  sub_37 = rsqrt_23 = None
        mul_60: f32[4096, 768] = torch.ops.aten.mul.Tensor(mul_59, _frozen_param188);  mul_59 = _frozen_param188 = None
        add_85: f32[4096, 768] = torch.ops.aten.add.Tensor(mul_60, _frozen_param189);  mul_60 = _frozen_param189 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:575, code: hidden_states = self.fc1(hidden_states)
        convert_element_type_202: bf16[4096, 768] = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16);  add_85 = None
        _frozen_param337: bf16[3072] = self._frozen_param337
        _frozen_param338: bf16[768, 3072] = self._frozen_param338
        _linear_pointwise_default_2: bf16[4096, 3072] = torch.ops.torch_ipex._linear_pointwise.default(convert_element_type_202, _frozen_param338, _frozen_param337, 'none', [], '');  convert_element_type_202 = _frozen_param338 = _frozen_param337 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:576, code: hidden_states = self.activation_fn(hidden_states)
        relu_11: bf16[4096, 3072] = torch.ops.aten.relu.default(_linear_pointwise_default_2);  _linear_pointwise_default_2 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:578, code: hidden_states = self.fc2(hidden_states)
        _frozen_param339: bf16[768] = self._frozen_param339
        _frozen_param340: bf16[3072, 768] = self._frozen_param340
        _linear_pointwise_default_1: bf16[4096, 768] = torch.ops.torch_ipex._linear_pointwise.default(relu_11, _frozen_param340, _frozen_param339, 'none', [], '');  relu_11 = _frozen_param340 = _frozen_param339 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:579, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        clone_83: bf16[4096, 768] = torch.ops.aten.clone.default(_linear_pointwise_default_1);  _linear_pointwise_default_1 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:581, code: hidden_states = (residual + hidden_states).view(hidden_states_shape)
        add_86: f32[4096, 768] = torch.ops.aten.add.Tensor(view_240, clone_83);  view_240 = clone_83 = None
        view_241: f32[2, 2048, 768] = torch.ops.aten.reshape.default(add_86, [2, 2048, 768]);  add_86 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:929, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(view_241, [2], correction = 0, keepdim = True)
        getitem_48: f32[2, 2048, 1] = var_mean_24[0]
        getitem_49: f32[2, 2048, 1] = var_mean_24[1];  var_mean_24 = None
        add_87: f32[2, 2048, 1] = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_24: f32[2, 2048, 1] = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
        sub_38: f32[2, 2048, 768] = torch.ops.aten.sub.Tensor(view_241, getitem_49);  view_241 = getitem_49 = None
        mul_61: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(sub_38, rsqrt_24);  sub_38 = rsqrt_24 = None
        mul_62: f32[2, 2048, 768] = torch.ops.aten.mul.Tensor(mul_61, _frozen_param194);  mul_61 = _frozen_param194 = None
        add_88: f32[2, 2048, 768] = torch.ops.aten.add.Tensor(mul_62, _frozen_param195);  mul_62 = _frozen_param195 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:1157, code: logits = self.lm_head(outputs[0]).contiguous()
        convert_element_type_207: bf16[2, 2048, 768] = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16);  add_88 = None
        _frozen_param341: bf16[768, 50272] = self._frozen_param341
        view_242: bf16[4096, 768] = torch.ops.aten.reshape.default(convert_element_type_207, [4096, 768]);  convert_element_type_207 = None
        _linear_pointwise_default: bf16[4096, 50272] = torch.ops.torch_ipex._linear_pointwise.default(view_242, _frozen_param341, None, 'none', [], '');  view_242 = _frozen_param341 = None
        view_243: bf16[2, 2048, 50272] = torch.ops.aten.reshape.default(_linear_pointwise_default, [2, 2048, 50272]);  _linear_pointwise_default = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:1164, code: shift_logits = logits[..., :-1, :].contiguous()
        slice_9: bf16[2, 2047, 50272] = torch.ops.aten.slice.Tensor(view_243, 1, 0, -1)
        slice_10: bf16[2, 2047, 50272] = torch.ops.aten.slice.Tensor(slice_9, 2, 0, 9223372036854775807);  slice_9 = None
        clone_84: bf16[2, 2047, 50272] = torch.ops.aten.clone.default(slice_10, memory_format = torch.contiguous_format);  slice_10 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:1165, code: shift_labels = labels[..., 1:].contiguous()
        slice_11: i64[2, 2047] = torch.ops.aten.slice.Tensor(arg198_1, 1, 1, 9223372036854775807);  arg198_1 = None
        clone_85: i64[2, 2047] = torch.ops.aten.clone.default(slice_11, memory_format = torch.contiguous_format);  slice_11 = None
        
        # File: /home/sdp/miniconda3/envs/chenjunjie-triton/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:1168, code: loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        view_244: bf16[4094, 50272] = torch.ops.aten.reshape.default(clone_84, [-1, 50272]);  clone_84 = None
        view_245: i64[4094] = torch.ops.aten.reshape.default(clone_85, [-1]);  clone_85 = None
        convert_element_type_209: f32[4094, 50272] = torch.ops.prims.convert_element_type.default(view_244, torch.float32);  view_244 = None
        amax_12: f32[4094, 1] = torch.ops.aten.amax.default(convert_element_type_209, [1], True)
        sub_39: f32[4094, 50272] = torch.ops.aten.sub.Tensor(convert_element_type_209, amax_12);  convert_element_type_209 = amax_12 = None
        exp_12: f32[4094, 50272] = torch.ops.aten.exp.default(sub_39)
        sum_13: f32[4094, 1] = torch.ops.aten.sum.dim_IntList(exp_12, [1], True);  exp_12 = None
        log: f32[4094, 1] = torch.ops.aten.log.default(sum_13);  sum_13 = None
        sub_40: f32[4094, 50272] = torch.ops.aten.sub.Tensor(sub_39, log);  sub_39 = log = None
        ne: b8[4094] = torch.ops.aten.ne.Scalar(view_245, -100)
        full_default_17: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='xpu', index=0), pin_memory = False)
        where_3: i64[4094] = torch.ops.aten.where.self(ne, view_245, full_default_17);  view_245 = full_default_17 = None
        unsqueeze_6: i64[4094, 1] = torch.ops.aten.unsqueeze.default(where_3, 1);  where_3 = None
        gather: f32[4094, 1] = torch.ops.aten.gather.default(sub_40, 1, unsqueeze_6);  sub_40 = unsqueeze_6 = None
        squeeze: f32[4094] = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: f32[4094] = torch.ops.aten.neg.default(squeeze);  squeeze = None
        where_4: f32[4094] = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_14: i64[] = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_210: f32[] = torch.ops.prims.convert_element_type.default(sum_14, torch.float32);  sum_14 = None
        sum_15: f32[] = torch.ops.aten.sum.default(where_4);  where_4 = None
        div_12: f32[] = torch.ops.aten.div.Tensor(sum_15, convert_element_type_210);  sum_15 = convert_element_type_210 = None
        return (div_12, view_243, clone, clone_1, clone_7, clone_8, clone_14, clone_15, clone_21, clone_22, clone_28, clone_29, clone_35, clone_36, clone_42, clone_43, clone_49, clone_50, clone_56, clone_57, clone_63, clone_64, clone_70, clone_71, clone_77, clone_78)
        