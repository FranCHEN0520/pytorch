
from ctypes import c_void_p, c_long
import torch
import intel_extension_for_pytorch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from intel_extension_for_pytorch._inductor.xpu.codecache import XPUAsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = XPUAsyncCompile()
constant0 = None  # 4f79991d3954e96b75ee76b253978a2542486b47c08eb26be10e0a04b5a06743
constant1 = None  # ffa1ae389366bb650dc2fa20e30bfbb7bf0eff51aaf13d748816335a80747466
constant2 = None  # 5eff1f9147965ddf5a7ba0ca408c33b2254db0e633a88752d3c1d975b293d8ab
constant3 = None  # 666a010ea22a33fdf5c2b84a8f1135f8f41b0b44f4e2cc23365fbfa3086b9d6c
constant4 = None  # a9140723a0a0a0e9b52c041a6bd61007d8f70a9508e7094f2bad675da7f1bdee
constant5 = None  # ec61ccf9540bdf748e5568a570c99baa9d29eb32c80320e43de86dc84d7a2816
constant6 = None  # 4ecac9c750e1505d9de9b9fc118d7d466cd567afad5e54273cb01221a8359d39
constant7 = None  # 4bcd0de469719e389c6a2b1701f96ff093198322174b880efaab5f60ac69ebc6
constant8 = None  # e07729255a8aad1ce08dcb5599d90c0ccc177a2f922ce1df9bcb5bcd758660e2
constant9 = None  # e8de15e124b3133c1bbcdf24345fc3faa2e52fd5f1805c8cae4e42eb3cc5c5ae
constant10 = None  # 51a12c995a273bea9eac9ea820ab46477274335415df6abdb68d3f82c9cb47e7
constant11 = None  # 0408d9a70ad87d871546dbdc3b38236752caed78fc599002a5baea3dc9ca8d68
constant12 = None  # cf28cff086f0af0efa11d9aa7ea0279544005453bb62b0b8dfdb3d894a00940e
constant13 = None  # 33a58a50d951d2326ae7243b0fcd5c4bbe6ea5c4d39f1679b8a19123ab96b191
constant14 = None  # c29a5d8a5ba7756b0f4bb30340009c9e1ba4a5004d1f22e85b68e3a9c0fc1bb9
constant15 = None  # d7339c1da45cd229cf3567c17178163ce2eda2eba067fa5252e64e2d931cca17
constant16 = None  # 4f7f808ea63096aa99437466d21130f2fd762c4b563dccf4c563d045548190fd
constant17 = None  # 71de35ef8f160ddf6d4e172b02f3af6c35c07b2c861765b73ddf7fd75ef3cbdf
constant18 = None  # 8030279b0d15346138c307c3010fbb26bc3dd9ca05a5f0e2562a9f2c33b26803
constant19 = None  # f1a9e8fbdce88b989db6ec721f5175874d4621902da44a538a9a525d58b918b6
constant20 = None  # 4bd6ecf43c4095a0bc312bff755c599d452b89f51f023e624b5d6d06383d744c
constant21 = None  # 45198ea16499c0ff2e1ee38e64fc05b22055478e6e06608087fbd3a84586b886
constant22 = None  # 0f9f49fbd0f72c65d8ed709aa931bf6b84dd52c9a86fbc90601c70e636e42120
constant23 = None  # 041522f70a14c9366e36eff974223617bed63081b59107be90f97758a27ac609
constant24 = None  # 6a05bdb44b4840108197b0fe7ea567108907a8e7cf2a887d21b435691388ccd8
constant25 = None  # e6b6fa197aea2e68ce12f3d5aaccbe94f5ff52681f09b37a8ef951ff72e23163
constant26 = None  # dffa9254de2d079407dc455b99a2515b1f40a2b975ef501a960832a68327d372
constant27 = None  # d95fc5bf6eee2a426f96e151022a27e506bb5a9a47797613ee3a2adc7e8c88f3
constant28 = None  # 29b31f86ea5314261ac1d4be547320172c885928e69acbe1ab830db4f8e82aae
constant29 = None  # df0fa9cb1d71e3bf6a7fb9857bd561fa5142c70ab5fa2b468e8b6e08630d0bf3
constant30 = None  # db2edb07c357c3b529b8c8a1f8b73720199a9a7b618887ebb669719c6436d385
constant31 = None  # 735e1d5de7d6ac88b5357d4cc43b5593da6e614671c31d8a3a6036dbdb18f706
constant32 = None  # 7c396b9c1252f2d6c4bbd21c9015ca68d7882b133dee0b561324ef5c8d8bec46
constant33 = None  # cf05e63e9f065361f8c0e0e446b6df7eeb63e77dffe66964399053605562dccc
constant34 = None  # cd3e125563b67dd8c2e7ef837f476d7d55620aa80b78be6932276ffee82e2e5d
constant35 = None  # b5557bbfc8a2db5fbf02ddc0063603c49fdcf784890a05cf72b8a1d6dbe0476d
constant36 = None  # ce37b412f9ff057d53c49f261870904e87e4ac041322925ff1e3df8b5d526a23
constant37 = None  # 8de7fb0245bd76a692827db4ac4eba543ee3d1d6a97fee86aec700b3173b3f92
constant38 = None  # 0d0683300a0956b18a33ab7bed4dc6f43ae3aa89921a3031b69d790e1e3046a0
constant39 = None  # cff028c902e362f976f8f087e2dc02e279660fb682263b3583cc04e7e96eb0a4
constant40 = None  # 328092c17d73834d1963792007db271cd31b0ace0240dd1329072afeeb759c3b
constant41 = None  # 49042472d113682b29b006af696c25554e3104282a9041051130635c9f02d3ae
constant42 = None  # b107a8b12931883f16c1a4b29a78ebc7c1dc5fde58d92f66fd02b3cb58e022fc
constant43 = None  # 7ac34c26f8251f1df2b4d75c82041556a33dd50c6752b6cb1de2342dc4e866cf
constant44 = None  # eb8a67145c3d940038b14c3293757fa88e8384b1cdc8b4a61da362996ed42115
constant45 = None  # 481a7e6d779e64f3496517b4cca21f2ff72e9d0ae5c5de039d8cd584eb24b75f
constant46 = None  # ed78dc8b64867d19ac5f4cba4473011b8491e87bee952a1db6cdacde08eabfaa
constant47 = None  # d30bfa3b6277d2d7f5678e076b1547ef92595716bfe176b386f13a3e0eaeeaf7
constant48 = None  # cadd37b44fbad95f8f2591d69418392b00f2ac86bc3d5c426701735cd561015f
constant49 = None  # dd09610cb121145a97a6ae8caed3ae78a622c87d4f67208ada55cebe1a3ff8f2
constant50 = None  # fac94e6b78a6312b48815439cd0ccf4f23f15b0ec4cd7d6cb900fd1516e08190
constant51 = None  # ca0ac5ea42fe42c6e2ee45c6d2553c86b47c8a84568a216fa4990236725e917f
constant52 = None  # b263c813efe1af5a279598fc4a52b28646874760a1e31ec04e3558eb9f2f6e25
constant53 = None  # ba20bf4fe401017764d06c745a8d141226fb393d8a58d90d3989e0b8268e6c05
constant54 = None  # 1f150c447886190ae817a8180912506487ffb20ccf345293f791a843d0a2ea22
constant55 = None  # f7a3fbf48f3ad0ee4c8d5e224e1b933bf0e7b98f302d26cd0201fe9fe676167c
constant56 = None  # 35986da1037741e90361bd5762c0daba450060f19cf337d4c48f6efeb9ed6701
constant57 = None  # 21cc35e86679f1bfbb960fe78f72c7ae313217e7979f52848365b2b2294f82cd
constant58 = None  # 0b7686d92c85a602fd739b31b787b6893f8f4cea923ed09dfd736f769c3a1139
constant59 = None  # 3995369d37837deb159e9c7fc1010a349e9a3ed024692f414108a9c06a146113
constant60 = None  # 80260862e67178c41f702e769fe4b940f1027a1478ab2e5e018e5d5db098ccdd
constant61 = None  # 392f8773f8c2286cdfb4d2b89f7d84cb7d067771e79c5076a142fa41ae8c9b0e
constant62 = None  # fd2601d3b49edd830f97ea54a067a1f1b1dfdfb69d38452af94f84262c47c22f
constant63 = None  # 1c13c4dfb81da878dab81cfccf973afbf25a59c92f70274b931e971b1fce64d0
constant64 = None  # 1c565f60991e92e4ebf2ee3d1ba9ce0a899f7e38f79848e663919730701ae279
constant65 = None  # df1af65c4ff6f38c553135d31ada9dd46c70761b280a4986937089c2affce063
constant66 = None  # 4f81431fc255f7d51a1113551f65ffd8d3d6ef1236e234a925c2af43e015ea5e
constant67 = None  # 9845a955c75df8b56fc18c56e5bbe5a709ae6a8cc2cdfe5052da8052c6710629
constant68 = None  # 1c3d3cd4f932e2b7530d47396ecdc81457350c7f8201bc2666872315563d1989
constant69 = None  # 2de86df6e72eb3998820121331545514d2e763c69ce7ba1c0d66312c2569f63e
constant70 = None  # 9356cdebfd43ed3d4149a9f7abad73077fc471ee12453883bb3dbe7510644eeb
constant71 = None  # b07a4c0bace642bbf0c13a293f828c69658424bb0ab14fca98f2446bb121d38a
constant72 = None  # 75beee128e29957eaade2cdf21dcdee6cd88cdcaf68c6b9a89e6b975a6abcced
constant73 = None  # 05f1fc8db232a3f838d194f5e5ff4a895372f6205138fe065bec636d504061ee
constant74 = None  # 959a3028b6eaccd51b109c0683e585a8e20a147671c637a28f7f67882dc19a25
constant75 = None  # a962f6b708f705c446bafb8ec4bd38b6c4aa7aefe8da7beec7ef2ba36d082f7a
constant76 = None  # 272a481b3c238c8cea4c5be862821ea02b603d4e2e7240f88ad899f3e5162d74
constant77 = None  # 06009c0312563ab3bee712d871f01e2e67e2ec58fe989db0d05d7cb4b4ebe912
constant78 = None  # 899d739c5ffe4f332c988f73843fec9d8ed3ba82228b739ff52db5f8765016e6


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/76/c763tdbojzem3g2cwx4lefg2std5qvfcf3ud2hff2wyj5b46xqut.py
# Source Nodes: [add_1, add_2, embedding, getitem_2, l__self___model_decoder_embed_tokens, l__self___model_decoder_layers_0_self_attn_layer_norm, l__self___model_decoder_layers_0_self_attn_q_proj, sub_1], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.native_layer_norm, aten.slice, aten.sub]
# add_1 => add_1
# add_2 => add_2
# embedding => embedding_1
# getitem_2 => slice_7, slice_8
# l__self___model_decoder_embed_tokens => embedding
# l__self___model_decoder_layers_0_self_attn_layer_norm => add_3, add_4, mul_1, mul_2, rsqrt, sub_2, var_mean
# l__self___model_decoder_layers_0_self_attn_q_proj => convert_element_type_3
# sub_1 => sub_1
triton_red_fused__to_copy_add_embedding_native_layer_norm_slice_sub_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_native_layer_norm_slice_sub_0', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_native_layer_norm_slice_sub_0', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_native_layer_norm_slice_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x3), None, eviction_policy='evict_last')
    x0 = xindex % 2048
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp3 = tl.load(in_ptr2 + (1536 + r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 50272, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 50272), "index out of bounds: 0 <= tmp1 < 50272")
        tmp2 = tl.load(in_ptr1 + (r2 + (768*tmp1)), rmask, eviction_policy='evict_last', other=0)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(in_ptr2 + (1536 + r2 + (768*x0)), rmask, eviction_policy='evict_last', other=0)
        tmp20 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0)
        tmp22 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0)
        tmp9 = tl.where(tmp0 < 0, tmp0 + 50272, tmp0)
        # tl.device_assert((0 <= tmp9) & (tmp9 < 50272), "index out of bounds: 0 <= tmp9 < 50272")
        tmp10 = tl.load(in_ptr1 + (r2 + (768*tmp9)), rmask, other=0)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp12 - tmp6
        tmp14 = 768.0
        tmp15 = tmp7 / tmp14
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = tl.math.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp21 = tmp19 * tmp20
        tmp23 = tmp21 + tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (768*x3)), tmp24, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from intel_extension_for_pytorch._C import _getCurrentRawStream as get_xpu_stream


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/o6/co6iywt66mfvqvowegy3sjwjnbfnyn2lmab2smyq6omihznxqqoi.py
# Source Nodes: [contiguous], Original ATen: [aten.clone]
# contiguous => clone
triton_poi_fused_clone_1 = async_compile.triton('triton_poi_fused_clone_1', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_1', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]})
@triton.jit
def triton_poi_fused_clone_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 2048
    x2 = (xindex // 131072) % 12
    x3 = (xindex // 1572864)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (1572864*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/3s/c3ssuhljevvdhnlootudxhsbfas7ii2lumy6z4qan5gn24ahygfz.py
# Source Nodes: [contiguous_2], Original ATen: [aten.clone]
# contiguous_2 => clone_2
triton_poi_fused_clone_2 = async_compile.triton('triton_poi_fused_clone_2', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_2', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]})
@triton.jit
def triton_poi_fused_clone_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 2048
    x2 = (xindex // 131072) % 12
    x3 = (xindex // 1572864)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (1572864*x3)), None).to(tl.float32)
    tmp1 = 0.125
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x4), tmp2, None)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/fu/cfurmt5xd6fexuwljjpl2fjvgoiq4k6ewebtq43ivagf7kl2fwu5.py
# Source Nodes: [bmm_1, softmax], Original ATen: [aten._softmax, aten._to_copy]
# bmm_1 => convert_element_type_12
# softmax => amax, div, exp, sub_3, sum_1
triton_red_fused__softmax__to_copy_3 = async_compile.triton('triton_red_fused__softmax__to_copy_3', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[65536, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_3', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_3(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 2048
    _tmp13 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = r2
        tmp3 = 1 + x0
        tmp4 = tmp2 < tmp3
        tmp5 = 0.0
        tmp6 = -3.4028234663852886e+38
        tmp7 = tl.where(tmp4, tmp5, tmp6)
        tmp8 = tl.full([1, 1], False, tl.int1)
        tmp9 = tl.where(tmp8, tmp6, tmp7)
        tmp10 = tmp1 + tmp9
        tmp11 = triton_helpers.maximum(tmp10, tmp6)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = triton_helpers.maximum(_tmp13, tmp12)
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp13 = triton_helpers.max2(_tmp13, 1)[:, None]
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp15 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = r2
        tmp18 = 1 + x0
        tmp19 = tmp17 < tmp18
        tmp20 = 0.0
        tmp21 = -3.4028234663852886e+38
        tmp22 = tl.where(tmp19, tmp20, tmp21)
        tmp23 = tl.full([1, 1], False, tl.int1)
        tmp24 = tl.where(tmp23, tmp21, tmp22)
        tmp25 = tmp16 + tmp24
        tmp26 = triton_helpers.maximum(tmp25, tmp21)
        tmp27 = tmp26 - tmp13
        tmp28 = tl.exp(tmp27)
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp32 = tl.load(in_ptr0 + (r2 + (2048*x3)), rmask, other=0).to(tl.float32)
        tmp33 = tmp32.to(tl.float32)
        tmp34 = r2
        tmp35 = 1 + x0
        tmp36 = tmp34 < tmp35
        tmp37 = 0.0
        tmp38 = -3.4028234663852886e+38
        tmp39 = tl.where(tmp36, tmp37, tmp38)
        tmp40 = tl.full([1, 1], False, tl.int1)
        tmp41 = tl.where(tmp40, tmp38, tmp39)
        tmp42 = tmp33 + tmp41
        tmp43 = triton_helpers.maximum(tmp42, tmp38)
        tmp44 = tmp43 - tmp13
        tmp45 = tl.exp(tmp44)
        tmp46 = tmp45 / tmp30
        tmp47 = tmp46.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp47, rmask)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/lk/clkvtmiajkxl65zdxeee3qzfmjw5iuemqo5xzy3fwjsiwr54jown.py
# Source Nodes: [reshape], Original ATen: [aten.clone]
# reshape => clone_4
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]})
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 12
    x2 = (xindex // 768) % 2048
    x3 = (xindex // 1572864)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (131072*x1) + (1572864*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/qr/cqr4gn2h2agnbioa5gd24ai6xwmqwuv32xcxkplysisdjtutswey.py
# Source Nodes: [l__self___model_decoder_layers_0_fc1, l__self___model_decoder_layers_0_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
# l__self___model_decoder_layers_0_fc1 => convert_element_type_15
# l__self___model_decoder_layers_0_final_layer_norm => add_7, add_8, mul_4, mul_5, rsqrt_1, sub_4, var_mean_1
triton_red_fused__to_copy_native_layer_norm_5 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_5', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_5', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp9_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp3 = tl.load(in_ptr2 + (1536 + r1 + (768*(x0 % 2048)) + (768*(tl.where((2 + (x0 % 2048)) >= 0, 0, 2050)))), rmask, eviction_policy='evict_last', other=0)
        tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.where(tmp0 < 0, tmp0 + 50272, tmp0)
        # tl.device_assert((0 <= tmp1) & (tmp1 < 50272), "index out of bounds: 0 <= tmp1 < 50272")
        tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask, eviction_policy='evict_last', other=0)
        tmp4 = tmp2 + tmp3
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 + tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight,
        )
        tmp9_mean = tl.where(rmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(rmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(rmask, tmp9_weight_next, tmp9_weight)
    tmp9_tmp, tmp10_tmp, tmp11_tmp = triton_helpers.welford(
        tmp9_mean, tmp9_m2, tmp9_weight, 1
    )
    tmp9 = tmp9_tmp[:, None]
    tmp10 = tmp10_tmp[:, None]
    tmp11 = tmp11_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr2 + (1536 + r1 + (768*(x0 % 2048)) + (768*(tl.where((2 + (x0 % 2048)) >= 0, 0, 2050)))), rmask, other=0)
        tmp16 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
        tmp26 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp28 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp12 = tl.where(tmp0 < 0, tmp0 + 50272, tmp0)
        # tl.device_assert((0 <= tmp12) & (tmp12 < 50272), "index out of bounds: 0 <= tmp12 < 50272")
        tmp13 = tl.load(in_ptr1 + (r1 + (768*tmp12)), rmask, other=0)
        tmp15 = tmp13 + tmp14
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 + tmp17
        tmp19 = tmp18 - tmp9
        tmp20 = 768.0
        tmp21 = tmp10 / tmp20
        tmp22 = 1e-05
        tmp23 = tmp21 + tmp22
        tmp24 = tl.math.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp27 = tmp25 * tmp26
        tmp29 = tmp27 + tmp28
        tmp30 = tmp29.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp30, rmask)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/me/cmeshlzzwd4pbebjfz7ulwosy254xar77jgckppmz7hg7ruopmez.py
# Source Nodes: [add_5, l__self___model_decoder_layers_1_self_attn_layer_norm, l__self___model_decoder_layers_1_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
# add_5 => add_9
# l__self___model_decoder_layers_1_self_attn_layer_norm => add_10, add_11, mul_6, mul_7, rsqrt_2, sub_5, var_mean_2
# l__self___model_decoder_layers_1_self_attn_q_proj => convert_element_type_20
triton_per_fused__to_copy_add_native_layer_norm_6 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_6', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_6', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (1536 + r1 + (768*(x0 % 2048)) + (768*(tl.where((2 + (x0 % 2048)) >= 0, 0, 2050)))), rmask, other=0)
    tmp5 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0)
    tmp36 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0)
    tmp1 = tl.where(tmp0 < 0, tmp0 + 50272, tmp0)
    # tl.device_assert((0 <= tmp1) & (tmp1 < 50272), "index out of bounds: 0 <= tmp1 < 50272")
    tmp2 = tl.load(in_ptr1 + (r1 + (768*tmp1)), rmask, other=0)
    tmp4 = tmp2 + tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 + tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = tl.full([1], 768, tl.int32)
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 / tmp19
    tmp21 = tmp11 - tmp20
    tmp22 = tmp21 * tmp21
    tmp23 = tl.broadcast_to(tmp22, [RBLOCK])
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = triton_helpers.promote_to_tensor(tl.sum(tmp25, 0))
    tmp27 = tmp10 - tmp20
    tmp28 = 768.0
    tmp29 = tmp26 / tmp28
    tmp30 = 1e-05
    tmp31 = tmp29 + tmp30
    tmp32 = tl.math.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tmp37.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp10, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp38, rmask)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/wb/cwbaqtgnmr463gh5ijog5md2o2pi2topcgzncjxrzyfl5pirnrso.py
# Source Nodes: [l__self___model_decoder_layers_1_fc1, l__self___model_decoder_layers_1_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
# l__self___model_decoder_layers_1_fc1 => convert_element_type_32
# l__self___model_decoder_layers_1_final_layer_norm => add_14, add_15, mul_10, mul_9, rsqrt_3, sub_7, var_mean_3
triton_per_fused__to_copy_native_layer_norm_7 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_7', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_7', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
    tmp27 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0)
    tmp29 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = tl.full([1], 768, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 / tmp12
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tmp3 - tmp13
    tmp21 = 768.0
    tmp22 = tmp19 / tmp21
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = tl.math.rsqrt(tmp24)
    tmp26 = tmp20 * tmp25
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp31, rmask)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/x3/cx3fwvov6fwp6p6wgtesqc62nolyo2uayuqzzfzkrunxy7do354z.py
# Source Nodes: [l__self___model_decoder_layers_2_self_attn_layer_norm, l__self___model_decoder_layers_2_self_attn_q_proj], Original ATen: [aten._to_copy, aten.native_layer_norm]
# l__self___model_decoder_layers_2_self_attn_layer_norm => add_17, add_18, mul_11, mul_12, rsqrt_4, sub_8, var_mean_4
# l__self___model_decoder_layers_2_self_attn_q_proj => convert_element_type_37
triton_per_fused__to_copy_native_layer_norm_8 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_8', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_8', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_8(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*x0)), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = tl.broadcast_to(tmp7, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp14 = tl.full([1], 768, tl.int32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tmp7 - tmp16
    tmp18 = tmp17 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = triton_helpers.promote_to_tensor(tl.sum(tmp21, 0))
    tmp23 = tmp6 - tmp16
    tmp24 = 768.0
    tmp25 = tmp22 / tmp24
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp29 = tmp23 * tmp28
    tmp31 = tmp29 * tmp30
    tmp33 = tmp31 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp34, rmask)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/ll/cll6nitvpxmvie47lgjptwf6zcw5xfphmqsqq7k2n7q6wkoggixb.py
# Source Nodes: [l__self___model_decoder_layers_2_fc1, l__self___model_decoder_layers_2_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
# l__self___model_decoder_layers_2_fc1 => convert_element_type_49
# l__self___model_decoder_layers_2_final_layer_norm => add_21, add_22, mul_14, mul_15, rsqrt_5, sub_10, var_mean_5
triton_red_fused__to_copy_native_layer_norm_9 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_9', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*fp32', 5: '*fp32', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_9', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (768*(x0 % 2048)) + (1572864*((((2048*(x0 // 2048)) + (x0 % 2048)) // 2048) % 2))), rmask, eviction_policy='evict_last', other=0)
        tmp1 = tl.load(in_ptr1 + (r1 + (768*(x0 % 2048)) + (1572864*((((2048*(x0 // 2048)) + (x0 % 2048)) // 2048) % 2))), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tmp0 + tmp2
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 + tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight,
        )
        tmp11_mean = tl.where(rmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(rmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(rmask, tmp11_weight_next, tmp11_weight)
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr0 + (r1 + (768*(x0 % 2048)) + (1572864*((((2048*(x0 // 2048)) + (x0 % 2048)) // 2048) % 2))), rmask, other=0)
        tmp15 = tl.load(in_ptr1 + (r1 + (768*(x0 % 2048)) + (1572864*((((2048*(x0 // 2048)) + (x0 % 2048)) // 2048) % 2))), rmask, other=0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
        tmp21 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = tmp14 + tmp16
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp20 + tmp22
        tmp24 = tmp23 - tmp11
        tmp25 = 768.0
        tmp26 = tmp12 / tmp25
        tmp27 = 1e-05
        tmp28 = tmp26 + tmp27
        tmp29 = tl.math.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp32 = tmp30 * tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp34.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (768*x0)), tmp35, rmask)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/ji/cjinvnw2odrswpysrv5bvbcrhhqoytht5eiiwlguj2j3nrpgltzb.py
# Source Nodes: [add_11, l__self___model_decoder_layers_3_self_attn_layer_norm, l__self___model_decoder_layers_3_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
# add_11 => add_23
# l__self___model_decoder_layers_3_self_attn_layer_norm => add_24, add_25, mul_16, mul_17, rsqrt_6, sub_11, var_mean_6
# l__self___model_decoder_layers_3_self_attn_q_proj => convert_element_type_54
triton_per_fused__to_copy_add_native_layer_norm_10 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_10', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_10', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 768
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (768*(x0 % 2048)) + (1572864*((((2048*(x0 // 2048)) + (x0 % 2048)) // 2048) % 2))), rmask, other=0)
    tmp1 = tl.load(in_ptr1 + (r1 + (768*(x0 % 2048)) + (1572864*((((2048*(x0 // 2048)) + (x0 % 2048)) // 2048) % 2))), rmask, other=0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r1 + (768*x0)), rmask, other=0).to(tl.float32)
    tmp36 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0)
    tmp38 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 + tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = tl.full([1], 768, tl.int32)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 / tmp21
    tmp23 = tmp13 - tmp22
    tmp24 = tmp23 * tmp23
    tmp25 = tl.broadcast_to(tmp24, [RBLOCK])
    tmp27 = tl.where(rmask, tmp25, 0)
    tmp28 = triton_helpers.promote_to_tensor(tl.sum(tmp27, 0))
    tmp29 = tmp12 - tmp22
    tmp30 = 768.0
    tmp31 = tmp28 / tmp30
    tmp32 = 1e-05
    tmp33 = tmp31 + tmp32
    tmp34 = tl.math.rsqrt(tmp33)
    tmp35 = tmp29 * tmp34
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(out_ptr0 + (r1 + (768*x0)), tmp12, rmask)
    tl.store(out_ptr3 + (r1 + (768*x0)), tmp40, rmask)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/d7/cd7kgt5vuizc6q3wsjecrohrhw57brcuzpbyqr5oqlc2z7uf7pmk.py
# Source Nodes: [l__self___lm_head], Original ATen: [aten.view]
# l__self___lm_head => view_243
triton_poi_fused_view_11 = async_compile.triton('triton_poi_fused_view_11', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(size_hints=[268435456], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_11', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]})
@triton.jit
def triton_poi_fused_view_11(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 205914112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/m4/cm47sbkv7wm6slmks6sgksmnfmhc4eyf7x5ttqkh5vo2tudruunl.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax, aten._to_copy]
# cross_entropy => amax_12, convert_element_type_209, exp_12, sub_39, sum_13
triton_red_fused__log_softmax__to_copy_12 = async_compile.triton('triton_red_fused__log_softmax__to_copy_12', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[4096, 65536],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax__to_copy_12', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_red_fused__log_softmax__to_copy_12(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4094
    rnumel = 50272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50272*(x0 % 2047)) + (102957056*(x0 // 2047))), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask & xmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (50272*(x0 % 2047)) + (102957056*(x0 // 2047))), rmask & xmask, other=0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr1 + (x0), tmp10, xmask)
''')


# kernel path: /home/sdp/chenjunjie/pytorch/inductor_log/huggingface/amp_bf16/mz/cmzny7ollhfzyngddz5xn4352pvumjuwjdom5fgrt7dcbwgfxjrz.py
# Source Nodes: [cross_entropy, masked_fill_], Original ATen: [aten.masked_fill, aten.nll_loss_forward]
# cross_entropy => convert_element_type_210, div_12, ne, neg, sum_14, sum_15, where_4
# masked_fill_ => full_default_1
triton_red_fused_masked_fill_nll_loss_forward_13 = async_compile.triton('triton_red_fused_masked_fill_nll_loss_forward_13', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from intel_extension_for_pytorch._inductor.xpu.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*i64', 2: '*bf16', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'xpu', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_masked_fill_nll_loss_forward_13', 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=set(), divisible_by_8=set())]}
)
@triton.jit
def triton_red_fused_masked_fill_nll_loss_forward_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4094
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (1 + (2048*(r0 // 2047)) + (r0 % 2047)), rmask, other=0)
        tmp8 = tl.load(in_ptr2 + (r0), rmask, other=0)
        tmp10 = tl.load(in_ptr3 + (r0), rmask, other=0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tl.full([1, 1], 0, tl.int64)
        tmp4 = tl.where(tmp2, tmp0, tmp3)
        tmp5 = tl.where(tmp4 < 0, tmp4 + 50272, tmp4)
        # tl.device_assert((0 <= tmp5) & (tmp5 < 50272), "index out of bounds: 0 <= tmp5 < 50272")
        tmp6 = tl.load(in_ptr1 + (tmp5 + (50272*(r0 % 2047)) + (102957056*(r0 // 2047))), rmask, other=0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp9 = tmp7 - tmp8
        tmp11 = tl.log(tmp10)
        tmp12 = tmp9 - tmp11
        tmp13 = -tmp12
        tmp14 = 0.0
        tmp15 = tl.where(tmp2, tmp13, tmp14)
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
        tmp19 = tmp2.to(tl.int64)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp23 = tmp21.to(tl.float32)
    tmp24 = tmp17 / tmp23
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp24, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg197_1, arg198_1 = args
    args.clear()
    buf3 = empty_strided((2, 2048, 768), (1572864, 768, 1), device='xpu', dtype=torch.bfloat16)
    # Source Nodes: [add_1, add_2, embedding, getitem_2, l__self___model_decoder_embed_tokens, l__self___model_decoder_layers_0_self_attn_layer_norm, l__self___model_decoder_layers_0_self_attn_q_proj, sub_1], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.native_layer_norm, aten.slice, aten.sub]
    stream0 = get_xpu_stream(0)
    triton_red_fused__to_copy_add_embedding_native_layer_norm_slice_sub_0.run(arg197_1, constant1, constant0, constant2, constant3, buf3, 4096, 768, grid=grid(4096), stream=stream0)
    buf4 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf3, (4096, 768), (768, 1), 0), constant5, constant4, 'none', [-1], '')
    buf5 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf3, (4096, 768), (768, 1), 0), constant6, constant4, 'none', [-1], '')
    buf6 = empty_strided((2, 12, 2048, 64), (1572864, 131072, 64, 1), device='xpu', dtype=torch.bfloat16)
    # Source Nodes: [contiguous], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf5, buf6, 3145728, grid=grid(3145728), stream=stream0)
    buf7 = reinterpret_tensor(buf5, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf5  # reuse
    # Source Nodes: [contiguous_2], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf4, buf7, 3145728, grid=grid(3145728), stream=stream0)
    del buf4
    buf8 = empty_strided((24, 2048, 2048), (4194304, 2048, 1), device='xpu', dtype=torch.bfloat16)
    # Source Nodes: [bmm], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf7, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf6, (24, 64, 2048), (131072, 1, 64), 0), out=buf8)
    buf13 = empty_strided((24, 2048, 2048), (4194304, 2048, 1), device='xpu', dtype=torch.bfloat16)
    # Source Nodes: [bmm_1, softmax], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf8, buf13, 49152, 2048, grid=grid(49152), stream=stream0)
    buf11 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf3, (4096, 768), (768, 1), 0), constant7, constant4, 'none', [-1], '')
    buf12 = reinterpret_tensor(buf3, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf3  # reuse
    # Source Nodes: [contiguous_1], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf11, buf12, 3145728, grid=grid(3145728), stream=stream0)
    buf14 = reinterpret_tensor(buf11, (24, 2048, 64), (131072, 64, 1)); del buf11  # reuse
    # Source Nodes: [bmm_1, softmax], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf13, reinterpret_tensor(buf12, (24, 2048, 64), (131072, 64, 1), 0), out=buf14)
    buf15 = reinterpret_tensor(buf7, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf7  # reuse
    # Source Nodes: [reshape], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf14, buf15, 3145728, grid=grid(3145728), stream=stream0)
    del buf14
    buf16 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf15, (4096, 768), (768, 1), 0), constant8, constant4, 'none', [-1], '')
    buf20 = reinterpret_tensor(buf15, (4096, 768), (768, 1)); del buf15  # reuse
    # Source Nodes: [l__self___model_decoder_layers_0_fc1, l__self___model_decoder_layers_0_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_red_fused__to_copy_native_layer_norm_5.run(arg197_1, constant1, constant0, buf16, constant2, constant3, buf20, 4096, 768, grid=grid(4096), stream=stream0)
    buf21 = torch.ops.torch_ipex._linear_pointwise(buf20, constant10, constant9, 'relu', [-1], '')
    buf22 = torch.ops.torch_ipex._linear_pointwise(buf21, constant11, constant4, 'none', [-1], '')
    del buf21
    buf23 = empty_strided((4096, 768), (768, 1), device='xpu', dtype=torch.float32)
    buf27 = reinterpret_tensor(buf20, (2, 2048, 768), (1572864, 768, 1)); del buf20  # reuse
    # Source Nodes: [add_5, l__self___model_decoder_layers_1_self_attn_layer_norm, l__self___model_decoder_layers_1_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
    triton_per_fused__to_copy_add_native_layer_norm_6.run(arg197_1, constant1, constant0, buf16, buf22, constant2, constant3, buf23, buf27, 4096, 768, grid=grid(4096), stream=stream0)
    del arg197_1
    del buf16
    buf28 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf27, (4096, 768), (768, 1), 0), constant12, constant4, 'none', [-1], '')
    buf29 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf27, (4096, 768), (768, 1), 0), constant13, constant4, 'none', [-1], '')
    buf30 = reinterpret_tensor(buf22, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf22  # reuse
    # Source Nodes: [contiguous_3], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf29, buf30, 3145728, grid=grid(3145728), stream=stream0)
    buf31 = reinterpret_tensor(buf29, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf29  # reuse
    # Source Nodes: [contiguous_5], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf28, buf31, 3145728, grid=grid(3145728), stream=stream0)
    del buf28
    buf32 = buf13; del buf13  # reuse
    # Source Nodes: [bmm_2], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf31, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf30, (24, 64, 2048), (131072, 1, 64), 0), out=buf32)
    buf37 = buf8; del buf8  # reuse
    # Source Nodes: [bmm_3, softmax_1], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf32, buf37, 49152, 2048, grid=grid(49152), stream=stream0)
    buf35 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf27, (4096, 768), (768, 1), 0), constant14, constant4, 'none', [-1], '')
    buf36 = reinterpret_tensor(buf27, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf27  # reuse
    # Source Nodes: [contiguous_4], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf35, buf36, 3145728, grid=grid(3145728), stream=stream0)
    buf38 = reinterpret_tensor(buf35, (24, 2048, 64), (131072, 64, 1)); del buf35  # reuse
    # Source Nodes: [bmm_3, softmax_1], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf37, reinterpret_tensor(buf36, (24, 2048, 64), (131072, 64, 1), 0), out=buf38)
    buf39 = reinterpret_tensor(buf31, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf31  # reuse
    # Source Nodes: [reshape_2], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf38, buf39, 3145728, grid=grid(3145728), stream=stream0)
    buf40 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf39, (4096, 768), (768, 1), 0), constant15, constant4, 'none', [-1], '')
    buf44 = reinterpret_tensor(buf39, (4096, 768), (768, 1)); del buf39  # reuse
    # Source Nodes: [l__self___model_decoder_layers_1_fc1, l__self___model_decoder_layers_1_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_7.run(buf23, buf40, constant2, constant3, buf44, 4096, 768, grid=grid(4096), stream=stream0)
    buf45 = torch.ops.torch_ipex._linear_pointwise(buf44, constant16, constant9, 'relu', [-1], '')
    buf46 = torch.ops.torch_ipex._linear_pointwise(buf45, constant17, constant4, 'none', [-1], '')
    del buf45
    buf50 = reinterpret_tensor(buf44, (2, 2048, 768), (1572864, 768, 1)); del buf44  # reuse
    # Source Nodes: [l__self___model_decoder_layers_2_self_attn_layer_norm, l__self___model_decoder_layers_2_self_attn_q_proj], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_8.run(buf23, buf40, buf46, constant2, constant3, buf50, 4096, 768, grid=grid(4096), stream=stream0)
    buf51 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf50, (4096, 768), (768, 1), 0), constant18, constant4, 'none', [-1], '')
    buf52 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf50, (4096, 768), (768, 1), 0), constant19, constant4, 'none', [-1], '')
    buf53 = reinterpret_tensor(buf38, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf38  # reuse
    # Source Nodes: [contiguous_6], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf52, buf53, 3145728, grid=grid(3145728), stream=stream0)
    buf54 = reinterpret_tensor(buf52, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf52  # reuse
    # Source Nodes: [contiguous_8], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf51, buf54, 3145728, grid=grid(3145728), stream=stream0)
    del buf51
    buf55 = buf37; del buf37  # reuse
    # Source Nodes: [bmm_4], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf54, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf53, (24, 64, 2048), (131072, 1, 64), 0), out=buf55)
    buf60 = buf32; del buf32  # reuse
    # Source Nodes: [bmm_5, softmax_2], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf55, buf60, 49152, 2048, grid=grid(49152), stream=stream0)
    buf58 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf50, (4096, 768), (768, 1), 0), constant20, constant4, 'none', [-1], '')
    buf59 = reinterpret_tensor(buf50, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf50  # reuse
    # Source Nodes: [contiguous_7], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf58, buf59, 3145728, grid=grid(3145728), stream=stream0)
    buf61 = reinterpret_tensor(buf58, (24, 2048, 64), (131072, 64, 1)); del buf58  # reuse
    # Source Nodes: [bmm_5, softmax_2], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf60, reinterpret_tensor(buf59, (24, 2048, 64), (131072, 64, 1), 0), out=buf61)
    buf62 = reinterpret_tensor(buf54, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf54  # reuse
    # Source Nodes: [reshape_4], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf61, buf62, 3145728, grid=grid(3145728), stream=stream0)
    del buf61
    buf63 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf62, (4096, 768), (768, 1), 0), constant21, constant4, 'none', [-1], '')
    buf67 = reinterpret_tensor(buf62, (4096, 768), (768, 1)); del buf62  # reuse
    # Source Nodes: [l__self___model_decoder_layers_2_fc1, l__self___model_decoder_layers_2_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_red_fused__to_copy_native_layer_norm_9.run(buf23, buf40, buf46, buf63, constant2, constant3, buf67, 4096, 768, grid=grid(4096), stream=stream0)
    buf68 = torch.ops.torch_ipex._linear_pointwise(buf67, constant22, constant9, 'relu', [-1], '')
    buf69 = torch.ops.torch_ipex._linear_pointwise(buf68, constant23, constant4, 'none', [-1], '')
    del buf68
    buf70 = empty_strided((4096, 768), (768, 1), device='xpu', dtype=torch.float32)
    buf74 = reinterpret_tensor(buf67, (2, 2048, 768), (1572864, 768, 1)); del buf67  # reuse
    # Source Nodes: [add_11, l__self___model_decoder_layers_3_self_attn_layer_norm, l__self___model_decoder_layers_3_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
    triton_per_fused__to_copy_add_native_layer_norm_10.run(buf23, buf40, buf46, buf63, buf69, constant2, constant3, buf70, buf74, 4096, 768, grid=grid(4096), stream=stream0)
    del buf40
    del buf46
    del buf63
    buf75 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf74, (4096, 768), (768, 1), 0), constant24, constant4, 'none', [-1], '')
    buf76 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf74, (4096, 768), (768, 1), 0), constant25, constant4, 'none', [-1], '')
    buf77 = reinterpret_tensor(buf69, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf69  # reuse
    # Source Nodes: [contiguous_9], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf76, buf77, 3145728, grid=grid(3145728), stream=stream0)
    buf78 = reinterpret_tensor(buf76, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf76  # reuse
    # Source Nodes: [contiguous_11], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf75, buf78, 3145728, grid=grid(3145728), stream=stream0)
    del buf75
    buf79 = buf60; del buf60  # reuse
    # Source Nodes: [bmm_6], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf78, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf77, (24, 64, 2048), (131072, 1, 64), 0), out=buf79)
    buf84 = buf55; del buf55  # reuse
    # Source Nodes: [bmm_7, softmax_3], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf79, buf84, 49152, 2048, grid=grid(49152), stream=stream0)
    buf82 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf74, (4096, 768), (768, 1), 0), constant26, constant4, 'none', [-1], '')
    buf83 = reinterpret_tensor(buf74, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf74  # reuse
    # Source Nodes: [contiguous_10], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf82, buf83, 3145728, grid=grid(3145728), stream=stream0)
    buf85 = reinterpret_tensor(buf82, (24, 2048, 64), (131072, 64, 1)); del buf82  # reuse
    # Source Nodes: [bmm_7, softmax_3], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf84, reinterpret_tensor(buf83, (24, 2048, 64), (131072, 64, 1), 0), out=buf85)
    buf86 = reinterpret_tensor(buf78, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf78  # reuse
    # Source Nodes: [reshape_6], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf85, buf86, 3145728, grid=grid(3145728), stream=stream0)
    buf87 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf86, (4096, 768), (768, 1), 0), constant27, constant4, 'none', [-1], '')
    buf91 = reinterpret_tensor(buf86, (4096, 768), (768, 1)); del buf86  # reuse
    # Source Nodes: [l__self___model_decoder_layers_3_fc1, l__self___model_decoder_layers_3_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_7.run(buf70, buf87, constant2, constant3, buf91, 4096, 768, grid=grid(4096), stream=stream0)
    buf92 = torch.ops.torch_ipex._linear_pointwise(buf91, constant28, constant9, 'relu', [-1], '')
    buf93 = torch.ops.torch_ipex._linear_pointwise(buf92, constant29, constant4, 'none', [-1], '')
    del buf92
    buf97 = reinterpret_tensor(buf91, (2, 2048, 768), (1572864, 768, 1)); del buf91  # reuse
    # Source Nodes: [l__self___model_decoder_layers_4_self_attn_layer_norm, l__self___model_decoder_layers_4_self_attn_q_proj], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_8.run(buf70, buf87, buf93, constant2, constant3, buf97, 4096, 768, grid=grid(4096), stream=stream0)
    buf98 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf97, (4096, 768), (768, 1), 0), constant30, constant4, 'none', [-1], '')
    buf99 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf97, (4096, 768), (768, 1), 0), constant31, constant4, 'none', [-1], '')
    buf100 = reinterpret_tensor(buf85, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf85  # reuse
    # Source Nodes: [contiguous_12], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf99, buf100, 3145728, grid=grid(3145728), stream=stream0)
    buf101 = reinterpret_tensor(buf99, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf99  # reuse
    # Source Nodes: [contiguous_14], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf98, buf101, 3145728, grid=grid(3145728), stream=stream0)
    del buf98
    buf102 = buf84; del buf84  # reuse
    # Source Nodes: [bmm_8], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf101, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf100, (24, 64, 2048), (131072, 1, 64), 0), out=buf102)
    buf107 = buf79; del buf79  # reuse
    # Source Nodes: [bmm_9, softmax_4], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf102, buf107, 49152, 2048, grid=grid(49152), stream=stream0)
    buf105 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf97, (4096, 768), (768, 1), 0), constant32, constant4, 'none', [-1], '')
    buf106 = reinterpret_tensor(buf97, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf97  # reuse
    # Source Nodes: [contiguous_13], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf105, buf106, 3145728, grid=grid(3145728), stream=stream0)
    buf108 = reinterpret_tensor(buf105, (24, 2048, 64), (131072, 64, 1)); del buf105  # reuse
    # Source Nodes: [bmm_9, softmax_4], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf107, reinterpret_tensor(buf106, (24, 2048, 64), (131072, 64, 1), 0), out=buf108)
    buf109 = reinterpret_tensor(buf101, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf101  # reuse
    # Source Nodes: [reshape_8], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf108, buf109, 3145728, grid=grid(3145728), stream=stream0)
    del buf108
    buf110 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf109, (4096, 768), (768, 1), 0), constant33, constant4, 'none', [-1], '')
    buf114 = reinterpret_tensor(buf109, (4096, 768), (768, 1)); del buf109  # reuse
    # Source Nodes: [l__self___model_decoder_layers_4_fc1, l__self___model_decoder_layers_4_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_red_fused__to_copy_native_layer_norm_9.run(buf70, buf87, buf93, buf110, constant2, constant3, buf114, 4096, 768, grid=grid(4096), stream=stream0)
    buf115 = torch.ops.torch_ipex._linear_pointwise(buf114, constant34, constant9, 'relu', [-1], '')
    buf116 = torch.ops.torch_ipex._linear_pointwise(buf115, constant35, constant4, 'none', [-1], '')
    del buf115
    buf117 = buf23; del buf23  # reuse
    buf121 = reinterpret_tensor(buf114, (2, 2048, 768), (1572864, 768, 1)); del buf114  # reuse
    # Source Nodes: [add_17, l__self___model_decoder_layers_5_self_attn_layer_norm, l__self___model_decoder_layers_5_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
    triton_per_fused__to_copy_add_native_layer_norm_10.run(buf70, buf87, buf93, buf110, buf116, constant2, constant3, buf117, buf121, 4096, 768, grid=grid(4096), stream=stream0)
    del buf110
    del buf116
    del buf87
    buf122 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf121, (4096, 768), (768, 1), 0), constant36, constant4, 'none', [-1], '')
    buf123 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf121, (4096, 768), (768, 1), 0), constant37, constant4, 'none', [-1], '')
    buf124 = reinterpret_tensor(buf93, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf93  # reuse
    # Source Nodes: [contiguous_15], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf123, buf124, 3145728, grid=grid(3145728), stream=stream0)
    buf125 = reinterpret_tensor(buf123, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf123  # reuse
    # Source Nodes: [contiguous_17], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf122, buf125, 3145728, grid=grid(3145728), stream=stream0)
    del buf122
    buf126 = buf107; del buf107  # reuse
    # Source Nodes: [bmm_10], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf125, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf124, (24, 64, 2048), (131072, 1, 64), 0), out=buf126)
    buf131 = buf102; del buf102  # reuse
    # Source Nodes: [bmm_11, softmax_5], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf126, buf131, 49152, 2048, grid=grid(49152), stream=stream0)
    buf129 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf121, (4096, 768), (768, 1), 0), constant38, constant4, 'none', [-1], '')
    buf130 = reinterpret_tensor(buf121, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf121  # reuse
    # Source Nodes: [contiguous_16], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf129, buf130, 3145728, grid=grid(3145728), stream=stream0)
    buf132 = reinterpret_tensor(buf129, (24, 2048, 64), (131072, 64, 1)); del buf129  # reuse
    # Source Nodes: [bmm_11, softmax_5], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf131, reinterpret_tensor(buf130, (24, 2048, 64), (131072, 64, 1), 0), out=buf132)
    buf133 = reinterpret_tensor(buf125, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf125  # reuse
    # Source Nodes: [reshape_10], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf132, buf133, 3145728, grid=grid(3145728), stream=stream0)
    buf134 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf133, (4096, 768), (768, 1), 0), constant39, constant4, 'none', [-1], '')
    buf138 = reinterpret_tensor(buf133, (4096, 768), (768, 1)); del buf133  # reuse
    # Source Nodes: [l__self___model_decoder_layers_5_fc1, l__self___model_decoder_layers_5_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_7.run(buf117, buf134, constant2, constant3, buf138, 4096, 768, grid=grid(4096), stream=stream0)
    buf139 = torch.ops.torch_ipex._linear_pointwise(buf138, constant40, constant9, 'relu', [-1], '')
    buf140 = torch.ops.torch_ipex._linear_pointwise(buf139, constant41, constant4, 'none', [-1], '')
    del buf139
    buf144 = reinterpret_tensor(buf138, (2, 2048, 768), (1572864, 768, 1)); del buf138  # reuse
    # Source Nodes: [l__self___model_decoder_layers_6_self_attn_layer_norm, l__self___model_decoder_layers_6_self_attn_q_proj], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_8.run(buf117, buf134, buf140, constant2, constant3, buf144, 4096, 768, grid=grid(4096), stream=stream0)
    buf145 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf144, (4096, 768), (768, 1), 0), constant42, constant4, 'none', [-1], '')
    buf146 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf144, (4096, 768), (768, 1), 0), constant43, constant4, 'none', [-1], '')
    buf147 = reinterpret_tensor(buf132, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf132  # reuse
    # Source Nodes: [contiguous_18], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf146, buf147, 3145728, grid=grid(3145728), stream=stream0)
    buf148 = reinterpret_tensor(buf146, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf146  # reuse
    # Source Nodes: [contiguous_20], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf145, buf148, 3145728, grid=grid(3145728), stream=stream0)
    del buf145
    buf149 = buf131; del buf131  # reuse
    # Source Nodes: [bmm_12], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf148, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf147, (24, 64, 2048), (131072, 1, 64), 0), out=buf149)
    buf154 = buf126; del buf126  # reuse
    # Source Nodes: [bmm_13, softmax_6], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf149, buf154, 49152, 2048, grid=grid(49152), stream=stream0)
    buf152 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf144, (4096, 768), (768, 1), 0), constant44, constant4, 'none', [-1], '')
    buf153 = reinterpret_tensor(buf144, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf144  # reuse
    # Source Nodes: [contiguous_19], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf152, buf153, 3145728, grid=grid(3145728), stream=stream0)
    buf155 = reinterpret_tensor(buf152, (24, 2048, 64), (131072, 64, 1)); del buf152  # reuse
    # Source Nodes: [bmm_13, softmax_6], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf154, reinterpret_tensor(buf153, (24, 2048, 64), (131072, 64, 1), 0), out=buf155)
    buf156 = reinterpret_tensor(buf148, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf148  # reuse
    # Source Nodes: [reshape_12], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf155, buf156, 3145728, grid=grid(3145728), stream=stream0)
    del buf155
    buf157 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf156, (4096, 768), (768, 1), 0), constant45, constant4, 'none', [-1], '')
    buf161 = reinterpret_tensor(buf156, (4096, 768), (768, 1)); del buf156  # reuse
    # Source Nodes: [l__self___model_decoder_layers_6_fc1, l__self___model_decoder_layers_6_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_red_fused__to_copy_native_layer_norm_9.run(buf117, buf134, buf140, buf157, constant2, constant3, buf161, 4096, 768, grid=grid(4096), stream=stream0)
    buf162 = torch.ops.torch_ipex._linear_pointwise(buf161, constant46, constant9, 'relu', [-1], '')
    buf163 = torch.ops.torch_ipex._linear_pointwise(buf162, constant47, constant4, 'none', [-1], '')
    del buf162
    buf164 = buf70; del buf70  # reuse
    buf168 = reinterpret_tensor(buf161, (2, 2048, 768), (1572864, 768, 1)); del buf161  # reuse
    # Source Nodes: [add_23, l__self___model_decoder_layers_7_self_attn_layer_norm, l__self___model_decoder_layers_7_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
    triton_per_fused__to_copy_add_native_layer_norm_10.run(buf117, buf134, buf140, buf157, buf163, constant2, constant3, buf164, buf168, 4096, 768, grid=grid(4096), stream=stream0)
    del buf134
    del buf140
    del buf157
    buf169 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf168, (4096, 768), (768, 1), 0), constant48, constant4, 'none', [-1], '')
    buf170 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf168, (4096, 768), (768, 1), 0), constant49, constant4, 'none', [-1], '')
    buf171 = reinterpret_tensor(buf163, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf163  # reuse
    # Source Nodes: [contiguous_21], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf170, buf171, 3145728, grid=grid(3145728), stream=stream0)
    buf172 = reinterpret_tensor(buf170, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf170  # reuse
    # Source Nodes: [contiguous_23], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf169, buf172, 3145728, grid=grid(3145728), stream=stream0)
    del buf169
    buf173 = buf154; del buf154  # reuse
    # Source Nodes: [bmm_14], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf172, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf171, (24, 64, 2048), (131072, 1, 64), 0), out=buf173)
    buf178 = buf149; del buf149  # reuse
    # Source Nodes: [bmm_15, softmax_7], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf173, buf178, 49152, 2048, grid=grid(49152), stream=stream0)
    buf176 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf168, (4096, 768), (768, 1), 0), constant50, constant4, 'none', [-1], '')
    buf177 = reinterpret_tensor(buf168, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf168  # reuse
    # Source Nodes: [contiguous_22], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf176, buf177, 3145728, grid=grid(3145728), stream=stream0)
    buf179 = reinterpret_tensor(buf176, (24, 2048, 64), (131072, 64, 1)); del buf176  # reuse
    # Source Nodes: [bmm_15, softmax_7], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf178, reinterpret_tensor(buf177, (24, 2048, 64), (131072, 64, 1), 0), out=buf179)
    buf180 = reinterpret_tensor(buf172, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf172  # reuse
    # Source Nodes: [reshape_14], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf179, buf180, 3145728, grid=grid(3145728), stream=stream0)
    buf181 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf180, (4096, 768), (768, 1), 0), constant51, constant4, 'none', [-1], '')
    buf185 = reinterpret_tensor(buf180, (4096, 768), (768, 1)); del buf180  # reuse
    # Source Nodes: [l__self___model_decoder_layers_7_fc1, l__self___model_decoder_layers_7_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_7.run(buf164, buf181, constant2, constant3, buf185, 4096, 768, grid=grid(4096), stream=stream0)
    buf186 = torch.ops.torch_ipex._linear_pointwise(buf185, constant52, constant9, 'relu', [-1], '')
    buf187 = torch.ops.torch_ipex._linear_pointwise(buf186, constant53, constant4, 'none', [-1], '')
    del buf186
    buf191 = reinterpret_tensor(buf185, (2, 2048, 768), (1572864, 768, 1)); del buf185  # reuse
    # Source Nodes: [l__self___model_decoder_layers_8_self_attn_layer_norm, l__self___model_decoder_layers_8_self_attn_q_proj], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_8.run(buf164, buf181, buf187, constant2, constant3, buf191, 4096, 768, grid=grid(4096), stream=stream0)
    buf192 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf191, (4096, 768), (768, 1), 0), constant54, constant4, 'none', [-1], '')
    buf193 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf191, (4096, 768), (768, 1), 0), constant55, constant4, 'none', [-1], '')
    buf194 = reinterpret_tensor(buf179, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf179  # reuse
    # Source Nodes: [contiguous_24], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf193, buf194, 3145728, grid=grid(3145728), stream=stream0)
    buf195 = reinterpret_tensor(buf193, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf193  # reuse
    # Source Nodes: [contiguous_26], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf192, buf195, 3145728, grid=grid(3145728), stream=stream0)
    del buf192
    buf196 = buf178; del buf178  # reuse
    # Source Nodes: [bmm_16], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf195, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf194, (24, 64, 2048), (131072, 1, 64), 0), out=buf196)
    buf201 = buf173; del buf173  # reuse
    # Source Nodes: [bmm_17, softmax_8], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf196, buf201, 49152, 2048, grid=grid(49152), stream=stream0)
    buf199 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf191, (4096, 768), (768, 1), 0), constant56, constant4, 'none', [-1], '')
    buf200 = reinterpret_tensor(buf191, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf191  # reuse
    # Source Nodes: [contiguous_25], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf199, buf200, 3145728, grid=grid(3145728), stream=stream0)
    buf202 = reinterpret_tensor(buf199, (24, 2048, 64), (131072, 64, 1)); del buf199  # reuse
    # Source Nodes: [bmm_17, softmax_8], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf201, reinterpret_tensor(buf200, (24, 2048, 64), (131072, 64, 1), 0), out=buf202)
    buf203 = reinterpret_tensor(buf195, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf195  # reuse
    # Source Nodes: [reshape_16], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf202, buf203, 3145728, grid=grid(3145728), stream=stream0)
    del buf202
    buf204 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf203, (4096, 768), (768, 1), 0), constant57, constant4, 'none', [-1], '')
    buf208 = reinterpret_tensor(buf203, (4096, 768), (768, 1)); del buf203  # reuse
    # Source Nodes: [l__self___model_decoder_layers_8_fc1, l__self___model_decoder_layers_8_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_red_fused__to_copy_native_layer_norm_9.run(buf164, buf181, buf187, buf204, constant2, constant3, buf208, 4096, 768, grid=grid(4096), stream=stream0)
    buf209 = torch.ops.torch_ipex._linear_pointwise(buf208, constant58, constant9, 'relu', [-1], '')
    buf210 = torch.ops.torch_ipex._linear_pointwise(buf209, constant59, constant4, 'none', [-1], '')
    del buf209
    buf211 = buf117; del buf117  # reuse
    buf215 = reinterpret_tensor(buf208, (2, 2048, 768), (1572864, 768, 1)); del buf208  # reuse
    # Source Nodes: [add_29, l__self___model_decoder_layers_9_self_attn_layer_norm, l__self___model_decoder_layers_9_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
    triton_per_fused__to_copy_add_native_layer_norm_10.run(buf164, buf181, buf187, buf204, buf210, constant2, constant3, buf211, buf215, 4096, 768, grid=grid(4096), stream=stream0)
    del buf181
    del buf187
    del buf204
    buf216 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf215, (4096, 768), (768, 1), 0), constant60, constant4, 'none', [-1], '')
    buf217 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf215, (4096, 768), (768, 1), 0), constant61, constant4, 'none', [-1], '')
    buf218 = reinterpret_tensor(buf210, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf210  # reuse
    # Source Nodes: [contiguous_27], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf217, buf218, 3145728, grid=grid(3145728), stream=stream0)
    buf219 = reinterpret_tensor(buf217, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf217  # reuse
    # Source Nodes: [contiguous_29], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf216, buf219, 3145728, grid=grid(3145728), stream=stream0)
    del buf216
    buf220 = buf201; del buf201  # reuse
    # Source Nodes: [bmm_18], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf219, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf218, (24, 64, 2048), (131072, 1, 64), 0), out=buf220)
    buf225 = buf196; del buf196  # reuse
    # Source Nodes: [bmm_19, softmax_9], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf220, buf225, 49152, 2048, grid=grid(49152), stream=stream0)
    buf223 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf215, (4096, 768), (768, 1), 0), constant62, constant4, 'none', [-1], '')
    buf224 = reinterpret_tensor(buf215, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf215  # reuse
    # Source Nodes: [contiguous_28], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf223, buf224, 3145728, grid=grid(3145728), stream=stream0)
    buf226 = reinterpret_tensor(buf223, (24, 2048, 64), (131072, 64, 1)); del buf223  # reuse
    # Source Nodes: [bmm_19, softmax_9], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf225, reinterpret_tensor(buf224, (24, 2048, 64), (131072, 64, 1), 0), out=buf226)
    buf227 = reinterpret_tensor(buf219, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf219  # reuse
    # Source Nodes: [reshape_18], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf226, buf227, 3145728, grid=grid(3145728), stream=stream0)
    buf228 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf227, (4096, 768), (768, 1), 0), constant63, constant4, 'none', [-1], '')
    buf232 = reinterpret_tensor(buf227, (4096, 768), (768, 1)); del buf227  # reuse
    # Source Nodes: [l__self___model_decoder_layers_9_fc1, l__self___model_decoder_layers_9_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_7.run(buf211, buf228, constant2, constant3, buf232, 4096, 768, grid=grid(4096), stream=stream0)
    buf233 = torch.ops.torch_ipex._linear_pointwise(buf232, constant64, constant9, 'relu', [-1], '')
    buf234 = torch.ops.torch_ipex._linear_pointwise(buf233, constant65, constant4, 'none', [-1], '')
    del buf233
    buf238 = reinterpret_tensor(buf232, (2, 2048, 768), (1572864, 768, 1)); del buf232  # reuse
    # Source Nodes: [l__self___model_decoder_layers_10_self_attn_layer_norm, l__self___model_decoder_layers_10_self_attn_q_proj], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_8.run(buf211, buf228, buf234, constant2, constant3, buf238, 4096, 768, grid=grid(4096), stream=stream0)
    buf239 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf238, (4096, 768), (768, 1), 0), constant66, constant4, 'none', [-1], '')
    buf240 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf238, (4096, 768), (768, 1), 0), constant67, constant4, 'none', [-1], '')
    buf241 = reinterpret_tensor(buf226, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf226  # reuse
    # Source Nodes: [contiguous_30], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf240, buf241, 3145728, grid=grid(3145728), stream=stream0)
    buf242 = reinterpret_tensor(buf240, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf240  # reuse
    # Source Nodes: [contiguous_32], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf239, buf242, 3145728, grid=grid(3145728), stream=stream0)
    del buf239
    buf243 = buf225; del buf225  # reuse
    # Source Nodes: [bmm_20], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf242, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf241, (24, 64, 2048), (131072, 1, 64), 0), out=buf243)
    buf248 = buf220; del buf220  # reuse
    # Source Nodes: [bmm_21, softmax_10], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf243, buf248, 49152, 2048, grid=grid(49152), stream=stream0)
    buf246 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf238, (4096, 768), (768, 1), 0), constant68, constant4, 'none', [-1], '')
    buf247 = reinterpret_tensor(buf238, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf238  # reuse
    # Source Nodes: [contiguous_31], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf246, buf247, 3145728, grid=grid(3145728), stream=stream0)
    buf249 = reinterpret_tensor(buf246, (24, 2048, 64), (131072, 64, 1)); del buf246  # reuse
    # Source Nodes: [bmm_21, softmax_10], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf248, reinterpret_tensor(buf247, (24, 2048, 64), (131072, 64, 1), 0), out=buf249)
    buf250 = reinterpret_tensor(buf242, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf242  # reuse
    # Source Nodes: [reshape_20], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf249, buf250, 3145728, grid=grid(3145728), stream=stream0)
    del buf249
    buf251 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf250, (4096, 768), (768, 1), 0), constant69, constant4, 'none', [-1], '')
    buf255 = reinterpret_tensor(buf250, (4096, 768), (768, 1)); del buf250  # reuse
    # Source Nodes: [l__self___model_decoder_layers_10_fc1, l__self___model_decoder_layers_10_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_red_fused__to_copy_native_layer_norm_9.run(buf211, buf228, buf234, buf251, constant2, constant3, buf255, 4096, 768, grid=grid(4096), stream=stream0)
    buf256 = torch.ops.torch_ipex._linear_pointwise(buf255, constant70, constant9, 'relu', [-1], '')
    buf257 = torch.ops.torch_ipex._linear_pointwise(buf256, constant71, constant4, 'none', [-1], '')
    del buf256
    buf258 = buf164; del buf164  # reuse
    buf262 = reinterpret_tensor(buf255, (2, 2048, 768), (1572864, 768, 1)); del buf255  # reuse
    # Source Nodes: [add_35, l__self___model_decoder_layers_11_self_attn_layer_norm, l__self___model_decoder_layers_11_self_attn_q_proj], Original ATen: [aten._to_copy, aten.add, aten.native_layer_norm]
    triton_per_fused__to_copy_add_native_layer_norm_10.run(buf211, buf228, buf234, buf251, buf257, constant2, constant3, buf258, buf262, 4096, 768, grid=grid(4096), stream=stream0)
    del buf211
    del buf228
    del buf234
    del buf251
    buf263 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf262, (4096, 768), (768, 1), 0), constant72, constant4, 'none', [-1], '')
    buf264 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf262, (4096, 768), (768, 1), 0), constant73, constant4, 'none', [-1], '')
    buf265 = reinterpret_tensor(buf257, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf257  # reuse
    # Source Nodes: [contiguous_33], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf264, buf265, 3145728, grid=grid(3145728), stream=stream0)
    buf266 = reinterpret_tensor(buf264, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf264  # reuse
    # Source Nodes: [contiguous_35], Original ATen: [aten.clone]
    triton_poi_fused_clone_2.run(buf263, buf266, 3145728, grid=grid(3145728), stream=stream0)
    del buf263
    buf267 = buf248; del buf248  # reuse
    # Source Nodes: [bmm_22], Original ATen: [aten.bmm]
    extern_kernels.bmm(reinterpret_tensor(buf266, (24, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf265, (24, 64, 2048), (131072, 1, 64), 0), out=buf267)
    buf272 = buf243; del buf243  # reuse
    # Source Nodes: [bmm_23, softmax_11], Original ATen: [aten._softmax, aten._to_copy]
    triton_red_fused__softmax__to_copy_3.run(buf267, buf272, 49152, 2048, grid=grid(49152), stream=stream0)
    del buf267
    buf270 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf262, (4096, 768), (768, 1), 0), constant74, constant4, 'none', [-1], '')
    buf271 = reinterpret_tensor(buf262, (2, 12, 2048, 64), (1572864, 131072, 64, 1)); del buf262  # reuse
    # Source Nodes: [contiguous_34], Original ATen: [aten.clone]
    triton_poi_fused_clone_1.run(buf270, buf271, 3145728, grid=grid(3145728), stream=stream0)
    buf273 = reinterpret_tensor(buf270, (24, 2048, 64), (131072, 64, 1)); del buf270  # reuse
    # Source Nodes: [bmm_23, softmax_11], Original ATen: [aten._softmax, aten._to_copy, aten.bmm]
    extern_kernels.bmm(buf272, reinterpret_tensor(buf271, (24, 2048, 64), (131072, 64, 1), 0), out=buf273)
    del buf272
    buf274 = reinterpret_tensor(buf266, (2, 2048, 12, 64), (1572864, 768, 64, 1)); del buf266  # reuse
    # Source Nodes: [reshape_22], Original ATen: [aten.clone]
    triton_poi_fused_clone_4.run(buf273, buf274, 3145728, grid=grid(3145728), stream=stream0)
    del buf273
    buf275 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf274, (4096, 768), (768, 1), 0), constant75, constant4, 'none', [-1], '')
    buf279 = reinterpret_tensor(buf274, (4096, 768), (768, 1)); del buf274  # reuse
    # Source Nodes: [l__self___model_decoder_layers_11_fc1, l__self___model_decoder_layers_11_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_7.run(buf258, buf275, constant2, constant3, buf279, 4096, 768, grid=grid(4096), stream=stream0)
    buf280 = torch.ops.torch_ipex._linear_pointwise(buf279, constant76, constant9, 'relu', [-1], '')
    buf281 = torch.ops.torch_ipex._linear_pointwise(buf280, constant77, constant4, 'none', [-1], '')
    del buf280
    buf285 = reinterpret_tensor(buf279, (2, 2048, 768), (1572864, 768, 1)); del buf279  # reuse
    # Source Nodes: [l__self___lm_head, l__self___model_decoder_final_layer_norm], Original ATen: [aten._to_copy, aten.native_layer_norm]
    triton_per_fused__to_copy_native_layer_norm_8.run(buf258, buf275, buf281, constant2, constant3, buf285, 4096, 768, grid=grid(4096), stream=stream0)
    del buf258
    del buf275
    del buf281
    buf286 = torch.ops.torch_ipex._linear_pointwise(reinterpret_tensor(buf285, (4096, 768), (768, 1), 0), constant78, None, 'none', [-1], '')
    del buf285
    buf287 = reinterpret_tensor(buf286, (2, 2048, 50272), (102957056, 50272, 1)); del buf286  # reuse
    # Source Nodes: [l__self___lm_head], Original ATen: [aten.view]
    triton_poi_fused_view_11.run(buf287, 205914112, grid=grid(205914112), stream=stream0)
    buf288 = empty_strided((4094, 1), (1, 4094), device='xpu', dtype=torch.float32)
    buf289 = empty_strided((4094, 1), (1, 4094), device='xpu', dtype=torch.float32)
    # Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax, aten._to_copy]
    triton_red_fused__log_softmax__to_copy_12.run(buf287, buf288, buf289, 4094, 50272, grid=grid(4094), stream=stream0)
    buf290 = empty_strided((), (), device='xpu', dtype=torch.float32)
    buf292 = buf290; del buf290  # reuse
    # Source Nodes: [cross_entropy, masked_fill_], Original ATen: [aten.masked_fill, aten.nll_loss_forward]
    triton_red_fused_masked_fill_nll_loss_forward_13.run(buf292, arg198_1, buf287, buf288, buf289, 1, 4094, grid=grid(1), stream=stream0)
    del arg198_1
    return (buf292, buf287, buf6, buf12, buf30, buf36, buf53, buf59, buf77, buf83, buf100, buf106, buf124, buf130, buf147, buf153, buf171, buf177, buf194, buf200, buf218, buf224, buf241, buf247, buf265, buf271, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    global constant0
    constant0 = rand_strided((2050, 768), (768, 1), device='xpu:0', dtype=torch.float32)
    global constant1
    constant1 = rand_strided((50272, 768), (768, 1), device='xpu:0', dtype=torch.float32)
    global constant2
    constant2 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global constant3
    constant3 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.float32)
    global constant4
    constant4 = rand_strided((768, ), (1, ), device='xpu:0', dtype=torch.bfloat16)
    global constant5
    constant5 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant6
    constant6 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant7
    constant7 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant8
    constant8 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant9
    constant9 = rand_strided((3072, ), (1, ), device='xpu:0', dtype=torch.bfloat16)
    global constant10
    constant10 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant11
    constant11 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant12
    constant12 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant13
    constant13 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant14
    constant14 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant15
    constant15 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant16
    constant16 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant17
    constant17 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant18
    constant18 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant19
    constant19 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant20
    constant20 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant21
    constant21 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant22
    constant22 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant23
    constant23 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant24
    constant24 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant25
    constant25 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant26
    constant26 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant27
    constant27 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant28
    constant28 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant29
    constant29 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant30
    constant30 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant31
    constant31 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant32
    constant32 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant33
    constant33 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant34
    constant34 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant35
    constant35 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant36
    constant36 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant37
    constant37 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant38
    constant38 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant39
    constant39 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant40
    constant40 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant41
    constant41 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant42
    constant42 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant43
    constant43 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant44
    constant44 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant45
    constant45 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant46
    constant46 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant47
    constant47 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant48
    constant48 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant49
    constant49 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant50
    constant50 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant51
    constant51 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant52
    constant52 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant53
    constant53 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant54
    constant54 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant55
    constant55 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant56
    constant56 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant57
    constant57 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant58
    constant58 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant59
    constant59 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant60
    constant60 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant61
    constant61 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant62
    constant62 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant63
    constant63 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant64
    constant64 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant65
    constant65 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant66
    constant66 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant67
    constant67 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant68
    constant68 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant69
    constant69 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant70
    constant70 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant71
    constant71 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant72
    constant72 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant73
    constant73 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant74
    constant74 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant75
    constant75 = rand_strided((768, 768), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant76
    constant76 = rand_strided((768, 3072), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    global constant77
    constant77 = rand_strided((3072, 768), (1, 3072), device='xpu:0', dtype=torch.bfloat16)
    global constant78
    constant78 = rand_strided((768, 50272), (1, 768), device='xpu:0', dtype=torch.bfloat16)
    arg197_1 = rand_strided((2, 2048), (2048, 1), device='xpu:0', dtype=torch.int64)
    arg198_1 = rand_strided((2, 2048), (2048, 1), device='xpu:0', dtype=torch.int64)
    return print_performance(lambda: call([arg197_1, arg198_1]), times=times, repeat=repeat, device='xpu')


if __name__ == "__main__":
    from intel_extension_for_pytorch._inductor.xpu.wrapper_benchmark import compiled_module_main
    compiled_module_main('OPTForCausalLM', benchmark_compiled_module)
