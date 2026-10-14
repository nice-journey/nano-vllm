from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """把向量按位置旋转。等价于把配对维度看成复数乘以 e^{i*m*θ}。

    这里用的是“前后对半配对”（x1 配 x2，即第 i 维和第 i+d/2 维），
    和 HuggingFace 的实现一致；另一种流派是相邻两维配对，两者权重布局不同。"""
    # x: CUDA 模型 dtype [T,heads,D]；T=本轮 token 数，heads=本卡头数，D=head_dim。
    # cos/sin: CUDA float32 [T,1,D/2]。
    # float() 转 float32（已有该 dtype 时可能不拷贝）；chunk 得到前后半的视图。
    # [T,1,D/2] 广播到各 head，使同一个 token 的所有 head 用相同位置角度。
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    # 标准的二维旋转矩阵
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # cat 新建 [T,heads,D] 存储，最后恢复输入 dtype；全程在 GPU，无 CPU 往返。
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE 旋转位置编码。

    核心思想：不给输入加位置向量，而是按位置旋转 q/k。
    因为旋转后的内积只和两者的相对角度（即相对位置）有关，
    所以天然编码了相对位置信息。"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        # 本实现不支持部分维度旋转（partial rotary）
        assert rotary_dim == head_size
        # 每一对维度用不同的频率：低维转得快、高维转得慢，形成多尺度的位置表示
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积：freqs[位置, 维度对] = 位置 * 该维度对的频率
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 预先把所有位置的 cos/sin 算好查表，推理时只需索引。
        # unsqueeze(1) 补出 head 维，方便后面和 [token, head, dim] 广播
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # shape [max_position,1,D]，float32；初始化在 ModelRunner 设置的 CUDA device 上。
        # register_buffer 注册非可训练状态，使其跟随 Module 的设备迁移，不作为 Parameter。
        # persistent=False：这是推导出来的常量，不需要存进 state_dict
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # @torch.compile 包装此函数，调用时由 PyTorch 编译/优化张量运算；
        # 与 CUDA Graph 的“记录固定 GPU 工作并重放”是不同层次的机制。
        # 按每个 token 的绝对位置去查表。positions 由 ModelRunner 根据 Sequence 进度构造，
        # 所以 chunked prefill / 带缓存前缀的场景下位置依然连续正确
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # 只转 q 和 k，v 不需要转（位置信息只参与注意力分数的计算）
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    """带缓存的工厂函数：所有层参数都一样，就让它们共用同一个实例，
    cos_sin_cache 这张大表只需要建一份。"""
    # 装饰器最多缓存一组参数及返回对象；相同参数再次调用直接返回同一实例，
    # 每个进程有自己的缓存，不会跨 TP 进程共享 GPU 对象。
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
