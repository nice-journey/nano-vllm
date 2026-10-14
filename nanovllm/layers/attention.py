import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,    # 每个 token 的 K（或 V）元素数 = 本卡 KV 头数 * head_dim，编译期常量
):
    """把这一批新算出来的 K/V 写进分页 cache 的指定槽位。
    一个 program 负责一个 token，写入位置由 slot_mapping 查表得到——
    这正是“分页”的关键：token 在 cache 里的位置可以是任意不连续的。"""
    # @triton.jit 将函数编译为 GPU kernel；ptr 参数是设备内存指针，偏移以元素为单位。
    # 一个 program 可协作处理多个元素，不等价于一个 CUDA 线程。
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    # -1 是 CUDA Graph padding 出来的假样本，直接跳过不写
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # 来自 Attention.forward，每层、每次 forward 调用；直接写已有 GPU 缓存，不经过 CPU。
    # key/value: CUDA 模型 dtype [N,K,Dh]，K=本卡 KV 头数，Dh=head_dim。
    # k_cache/v_cache: 同 dtype/device [物理页数,S,K,Dh]，S=每页 token 数；
    # slot_mapping: CUDA int32 [N]，N=本轮输入 token 数。
    # cache 的前两维可按槽位理解为 [物理页数*S,K*Dh]，kernel 直接用地址计算，不另建张量。
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # stride 表示沿某维前进一步要跨过多少个元素。这里只要求一个 token 内的 K*Dh 连续，
    # 不要求整份 key/value 连续：split 后 token 之间可能有空隙，用 stride(0) 跨过去。
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # grid 开 N 个 program，每个处理一个 token
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):
    """注意力层。本身不含参数（q/k/v/o 的投影在外面的 Qwen3Attention 里），
    只负责“写 KV cache + 调 FlashAttention”这两件事。"""

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 占位，真正的 cache 由 ModelRunner.allocate_kv_cache() 在运行时绑进来
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # Qwen3Attention -> 本函数；投影/norm/RoPE 已在外层完成。
        # q: CUDA 模型 dtype [T,Q_heads,D]，k/v: [T,KV_heads,D]；头数均为本卡局部值。
        # T=本轮 token 数，D=head_dim；Q_heads/KV_heads 决定多少个 query heads 共用一组 KV。
        # 数据流：新 K/V -> 按 slot_mapping 写 cache -> 按页表读历史 -> attention 输出。
        # 数学上做 softmax(QK^T/sqrt(D) + 因果掩码)V；FlashAttention 分块执行，
        # 避免把完整注意力分数矩阵写回显存。各序列的边界由 Context 提供。
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # numel 为 0 说明还没绑 cache（warmup 阶段），跳过写入
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                # 前缀命中或 chunk 续算时，历史 KV 不在本次传入的 k/v 里，而在 cache 中，
                # 所以改成直接读整块 cache，由 block_table 指引该读哪些页
                k, v = k_cache, v_cache
            # varlen 版本处理首尾相接的变长批；causal=True 加因果掩码
            # 有缓存时单条序列 Q_len < K_len，query 对应已有上下文末尾的本轮 token；
            # 因果掩码按该位置对齐，允许读历史，不允许看本轮中位于自己之后的 token。
            # 输出 o: [T,Q_heads,D]，dtype/device 与 q 相同。
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            # decode 每条序列只有 1 个 query token。unsqueeze(1) 把 [bs, heads, dim]
            # 变成 [bs, 1, heads, dim]，符合该 kernel 要求的 seqlen 维度
            # unsqueeze 是共享数据的视图；输出也保留长度为 1 的轴：[B,1,Q_heads,D]，B=序列数。
            # context_lens 包含本轮刚写的 KV；历史 q 不缓存，因为只需计算新位置的输出。
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True)
        return o
