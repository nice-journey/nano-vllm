from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """每条请求的采样参数。"""

    # generate 可为所有请求共用一份，也可传 list[SamplingParams] 逐条指定。
    # Sequence 在入队时读取字段；temperature 交给 Sampler，另两项由 Scheduler 判断终止。
    # 温度，越大越随机。会在 Sampler 里用来除 logits
    temperature: float = 1.0
    # 最多生成多少个 token（不含 prompt）
    max_tokens: int = 64
    # True = 遇到 eos 也不停，一直生成到 max_tokens（压测时常用，保证每条长度可控）
    ignore_eos: bool = False

    def __post_init__(self):
        # 本项目的 Sampler 用的是 Gumbel-max 技巧，实现里要除以 temperature，
        # 所以不允许 temperature 为 0（即不支持纯 greedy），否则会除零
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
