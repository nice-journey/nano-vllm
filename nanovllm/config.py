import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    """引擎的全局配置。slots=True 省内存、禁止动态加字段（写错字段名会直接报错）。"""

    # @dataclass 根据下面的字段生成构造函数，近似 Config(model, max_num_batched_tokens=16384, ...)。
    # 类型标注帮助阅读/静态检查，并不自动在运行时校验所有值；具体限制看 __post_init__。
    # 模型目录（本地路径，必须是已经下载好的 HuggingFace 格式目录）
    model: str
    # 一次 forward 最多塞多少 token。prefill 阶段用它限制 batch 的总长度，
    # 是显存占用和计算量的主要闸门（chunked prefill 的"块大小"也由它决定）
    max_num_batched_tokens: int = 16384
    # 一个 batch 最多同时跑多少条序列（并发度上限）
    max_num_seqs: int = 512
    # 长度配置，会和模型位置编码上限取 min，用于 warmup 和 CUDA Graph 页表容量；
    # 注意当前 add_request/postprocess 并未用它校验或截断每条请求的 prompt+生成总长度
    max_model_len: int = 4096
    # 显存预算比例，不是 KV cache 本身固定占比。缓存预算扣除已用量及前向临时峰值，
    # 具体为 total*比例 - used - (peak-current)，见 ModelRunner.allocate_kv_cache
    gpu_memory_utilization: float = 0.9
    # 张量并行度（几张卡切模型）。1 表示单卡
    tensor_parallel_size: int = 1
    # True = 跳过 CUDA Graph；False = 对 decode 捕获 Graph。True 不会禁用层上的 torch.compile
    enforce_eager: bool = False
    # HuggingFace 的模型结构配置，在 __post_init__ 里从磁盘读出来
    hf_config: AutoConfig | None = None
    # eos token id，由 LLMEngine 拿到 tokenizer 后回填
    eos: int = -1
    # KV cache 分页的页大小（一个 block 存多少个 token 的 KV）
    kvcache_block_size: int = 256
    # KV cache 总共能分多少个 block，由 ModelRunner 根据实际剩余显存算出来后回填
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        # dataclass 构造完自动调用，这里做参数校验 + 派生字段填充
        assert os.path.isdir(self.model)
        # block_size 必须是 256 的倍数：FlashAttention 的 paged kernel 对页大小有对齐要求
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 用户给的 max_model_len 不能超过模型本身支持的位置编码长度
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
