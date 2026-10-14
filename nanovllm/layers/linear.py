import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    """整除并断言能整除——张量并行切分时维度必须能被卡数整除。"""
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """所有并行线性层的基类。

    张量并行的两种基本切法（权重形状是 [output_size, input_size]）：
      - 列并行（tp_dim=0）：按输出维切。每卡算出一部分输出，不需要通信，
        但输出是“残缺”的，得由下一层来收尾
      - 行并行（tp_dim=1）：按输入维切。每卡拿部分输入算出完整形状的部分和，
        最后 all_reduce 求和才是正确结果
    Transformer 里把 qkv/gate_up 做成列并行、o_proj/down_proj 做成行并行，
    TP>1 时一个 DecoderLayer 内 attention 和 MLP 各一次 all_reduce，共两次。"""

    # 命名来自 Y=X@A 的数学矩阵 A；PyTorch 保存的是转置权重 W=[out,in]，
    # F.linear(x,W,b)=x@W.T+b，所以“列并行”在这里反而切 W 的第 0 维（行）。
    # T=token 数、p=TP 卡数：
    # column: x[T,in] × W_local[out/p,in] -> y[T,out/p]，保留局部输出。
    # row:    x_local[T,in/p] × W_local[out,in/p] -> 局部和[T,out] -> all_reduce。
    # 这些激活/权重均在当前 rank 的 CUDA device，浮点 dtype 由模型配置决定。

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # Parameter 是注册到 nn.Module 的 Tensor；empty 只分配空间，loader 随后填权重。
        # 把加载函数挂在 Parameter 上，loader.py 会按名字取出来用。
        # 这样“怎么切权重”的知识就留在定义它的层里，加载器不需要知道细节
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 注册成 None，保证 state_dict 和属性访问行为一致
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """不切分，每张卡存一份完整权重。本项目里暂未使用，留作参考。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """列并行：按输出维切。每卡只持有 output_size / tp_size 行权重。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        # 从完整权重里 narrow 出属于本 rank 的那一段
        # narrow(dim,start,length) 是 Tensor 视图；loaded_weight 来自 CPU safetensors，
        # param_data 是 GPU 目标存储，copy_ 才真正把这一分片拷到卡上（并按目标 dtype 转换）。
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 不需要通信，输出就是部分输出维
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """把多个输出维度拼在一起的列并行层，用于 gate_proj + up_proj 融合成 gate_up_proj。
    融合的好处：原本两次 GEMM 变一次，kernel launch 和访存都更划算。"""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # loaded_shard_id 指明这次传进来的是第几段（0=gate, 1=up）。
        # 要做两件事：定位它在合并参数里的区间，以及在该区间内再做张量并行切分
        param_data = param.data
        # 注意除以 tp_size：每卡持有的是切过的，偏移量也要相应缩小
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # chunk(p,dim) 沿指定轴分片，再取当前 rank；这里各维已保证可整除。
        # 与 narrow 一样共享源数据，最终 copy_ 才写入融合参数的目标区间。
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """把 q/k/v 三个投影融合成一个大矩阵的列并行层。
    GQA 下 k/v 的头数比 q 少，所以三段长度不一样，加载时要分别算偏移。"""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        # 每卡分到的头数。注意力的头之间互不相干，所以按头切是天然的并行维度
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        # 合并后的布局是 [q | k | v]，按这个顺序算每段的起点和长度（都是本卡的分片大小）
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """行并行：按输入维切。配合前面的列并行层使用——
    上一层每卡算出部分输出，正好作为这一层每卡的部分输入，中间不用通信。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        # bias 是一维的，对应输出维，而输出维没被切，所以整份拷贝
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每卡得到的是“部分和”。bias 只让 rank0 加，否则 all_reduce 后会被加 tp_size 次
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # all_reduce 原地对各卡 y 逐元素求和，并让每卡都得到结果；不是拼接。
            dist.all_reduce(y)
        return y
