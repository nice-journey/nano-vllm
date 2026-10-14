import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU 激活：把融合后的 gate_up 输出一劈两半，silu(gate) * up。

    上游的 gate_up_proj 把两个投影合成了一次矩阵乘，所以这里拿到的 x
    是 [..., 2 * intermediate_size / tp_size]，前半是本卡 gate、后半是本卡 up。
    torch.compile 可融合这些逐元素操作以减少中间读写，实际 kernel 数取决于编译结果。"""

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # T=本轮 token 数，I=intermediate_size，p=张量并行卡数。
        # Qwen3MLP 调用；CUDA 模型 dtype [T,2*I/p] -> 两个共享存储的视图 [T,I/p]。
        # dim=-1 表示最后一维。silu(gate)=gate*sigmoid(gate)，与 up 相乘后 shape 为 [T,I/p]。
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
