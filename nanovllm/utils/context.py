from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    """一次 forward 所需的“元信息”。

    这些信息主要供底层 Attention 使用，LMHead 也需要其中的 prefill 边界。
    全局 context 省去在 Model/DecoderLayer 间逐层传参，保持主干接口简洁；
    使用全局变量并不是 CUDA Graph 捕获的必要条件。"""

    # @dataclass 自动生成 __init__ 等方法；slots=True 限制为这些预定义字段，省去实例 __dict__。
    # Context 对象在 CPU/Python 侧，字段引用的 Tensor 在当前 rank 的 CUDA device。
    # 各进程有自己的 _CONTEXT；同一进程内是全局可变状态，不支持任意并发重叠的 forward。
    # ModelRunner.prepare_* 设置 -> Attention/LMHead 读取 -> run 完成后 reset。
    # 以下整数 Tensor 均为 int32：cu_seqlens [B+1]、slot_mapping [T]、
    # context_lens [B]、block_tables [B,P]；B=序列数，T=输入 token 数，P=页表补齐宽度。

    is_prefill: bool = False
    # 下面四个只在 prefill 用：变长批的边界信息，给 flash_attn_varlen_func
    cu_seqlens_q: torch.Tensor | None = None    # query 长度的前缀和，shape [batch+1]
    cu_seqlens_k: torch.Tensor | None = None    # key 长度的前缀和，shape [batch+1]
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 本次每个 token 的 KV 要写进 cache 的槽位号，prefill / decode 都用
    slot_mapping: torch.Tensor | None = None
    # 只在 decode 用：每条序列当前的总长度，kernel 据此知道要读多少历史 KV
    context_lens: torch.Tensor | None = None
    # 页表，paged attention 靠它把逻辑位置翻译成物理页
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    # global 表示重新绑定模块级变量；否则赋值会创建函数局部变量。
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    # 每次 run 后换成空 Context，移除这份对临时元数据的引用，避免下一轮误用。
    # 不清空真正的 KV cache；仍被 CUDA Graph 缓冲区等对象引用的 Tensor 也不会被释放。
    global _CONTEXT
    _CONTEXT = Context()
