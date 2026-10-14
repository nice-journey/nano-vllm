import torch
from torch import nn


class RMSNorm(nn.Module):
    """RMSNorm：相比 LayerNorm 去掉了减均值和 bias，按均方根缩放，再乘可学习权重。

    这里额外支持“融合残差”：把 x + residual 和归一化放入同一个可编译函数，
    让 torch.compile 有机会融合运算，减少中间读写；实际 kernel 数取决于编译结果。"""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x: CUDA 模型 dtype [...,D]，D 可能是 hidden_size，也可能是 QK-Norm 的 head_dim。
        # 沿最后一维独立归一化，输出 shape/dtype 与 x 相同，不混合 token 或 head。
        # 升到 float32 再算：bf16/fp16 下平方和容易溢出或掉精度
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # var 实际是均方 E[x²]（不是减均值后的方差），shape [...,1]，供最后一维广播。
        # rsqrt 是倒数平方根；mul_ 的下划线表示原地修改接收者。
        # float() 只有需要转换 dtype 时才产生新数据；输入已是 float32 时可能共享原存储。
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 输入 x/residual 同 shape/device/dtype。返回 (归一化结果, 加完但尚未归一化的残差)。
        # 这是两个用途不同的值：前者送入 attention/MLP，后者沿残差通路继续向下传。
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        # 注意：返回的新 residual 是“加完还没归一化”的值，
        # 它会被带到下一层继续累加，这就是残差流
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 在 Python 层按 residual 是否存在分派到两个编译函数；
        # 不应据此推断 if 本身一定造成 graph break，编译行为还取决于输入和 PyTorch 版本。
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
