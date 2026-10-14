import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


# T=本轮 token 数，B=序列数，V=词表大小，H=hidden_size，p=张量并行卡数。

class VocabParallelEmbedding(nn.Module):
    """按词表维切分的 embedding。词表动辄十几万，切开能省不少显存。

    每张卡只持有词表的一段，查不到的 token 本卡输出 0，
    最后 all_reduce 一求和，各卡贡献的非零部分拼成完整结果。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        # 本卡负责的 token id 区间 [start, end)
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        # Qwen3Model 每轮调用：x 是 CUDA int64 [T] 的 token id。
        # weight: CUDA 模型 dtype [V/p,H]；输出 y: 同 dtype [T,H]，每卡得到完整特征。
        # 多卡 mask 是 bool [T]；& 是 Tensor 逐元素“与”，不能用 Python 的 and 代替。
        if self.tp_size > 1:
            # 不属于本卡的 token 先被 mask 成 0 号下标（避免越界），
            # 属于本卡的则减去起点转成本地下标
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            # 把“查错的”那些行清零，再求和——每个 token 只会有一张卡贡献非零向量
            # mask.unsqueeze(1): [T] -> [T,1]，广播到 y 的 H 列，一次清掉整行。
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """输出层。权重形状和 embedding 一样（可以共享），但计算方向相反：
    hidden_state -> 词表 logits。"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # ModelRunner.run_model -> compute_logits -> 本函数；每轮一次。
        # 输入是 CUDA 模型 dtype hidden states；prefill [T,H]，decode [B,H]。
        # 输出 rank0 为 CUDA 模型 dtype logits [B,V]；TP 的其它 rank 返回 None。
        context = get_context()
        if context.is_prefill:
            # 关键优化：prefill 时只有每条序列的最后一个位置需要预测下一个 token，
            # 中间位置的 hidden_state 不需要送入 LMHead（但主干仍需计算它们的 KV）。把一个
            # [总token数, vocab] 的巨大矩阵乘缩小成 [batch, vocab]
            last_indices = context.cu_seqlens_q[1:] - 1
            # 若 cu_seqlens_q=[0,3,5]，则 last_indices=[2,4]，挑出各分块最后一行。
            # Tensor 索引 x[last_indices] 是高级索引，会生成新张量 [B,H]；
            # contiguous() 保证紧凑布局，已连续时直接返回，不一定再复制。
            # 中间 chunk 也会算候选 logits，但 Scheduler 不会接收它生成的 token。
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        # 本卡 logits [B,V/p] 是词表片段的分数；Sampler 随后才做温度缩放和 softmax。
        if self.tp_size > 1:
            # 每卡算出词表的一段 logits。这里用 gather 而不是 all_gather，
            # 因为只有 rank0 需要采样，避免在其它卡也收集完整词表
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            # cat 沿词表轴分配并拼接 [B,V/p] 的张量为 [B,V]，与 all_reduce 的求和不同。
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
