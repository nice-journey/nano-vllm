import torch
from torch import nn


class Sampler(nn.Module):
    """从 logits 采样下一个 token。用的是 Gumbel-max 技巧而不是 torch.multinomial。

    原理：对每个类别算 prob_i / E_i（E_i 是独立的指数分布随机数），取 argmax，
    在理想连续分布下结果服从 prob 指定的分类分布（代码中 clamp 是数值保护）。
    它等价于给 log(prob) 加 Gumbel 噪声再取最大值，适合在 GPU 上按 batch 并行。"""

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # ModelRunner.run 在 rank0 每轮调用；本实现只做温度采样，没有 top-k/top-p 筛选。
        # 输入 logits: CUDA 模型 dtype [B,V]；temperatures: CUDA float32 [B]，B=序列数，V=词表大小。
        # unsqueeze(1) -> [B,1]，使每条请求的温度作用于自己整行词表分数。
        # probs: CUDA float32 [B,V]，每行 softmax 后和为 1。
        # 输出 sample_tokens: CUDA int64 [B]，每项是一个词表 id，随后由 run().tolist() 回 CPU。
        # 升 float32 保证数值稳定；unsqueeze 让每条序列用自己的温度做广播除法
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # clamp_min 防止指数随机数取到 0 导致除零
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
