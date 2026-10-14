from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """一条序列的生命周期状态。"""
    # Enum 用有名字的成员代替裸整数；auto() 自动分配值，不依赖具体数值。
    WAITING = auto()    # 在等待队列里，prompt 还没（完全）算完
    RUNNING = auto()    # 已被调度到最后一个 prefill 分块，或正在 decode
    FINISHED = auto()   # 已经生成完（遇到 eos 或达到 max_tokens）


class Sequence:
    """一条请求对应的全部运行时状态：token、调度进度、以及它占用的 KV cache 页表。"""

    # 类级别变量：KV cache 的页大小。LLMEngine 启动时会按 Config 覆写它，
    # 这样所有 Sequence 实例共享同一个值，不用每条都存一份
    block_size = 256
    # 全局自增 id 生成器
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # 默认参数对象在定义函数时创建一次（不像 C++ 每次调用求值）。
        # 这里仅读取 SamplingParams；若修改这个默认对象，会影响后续省略该参数的调用。
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # 浅拷贝创建新 list；元素是不可变 int，无需深拷贝。引擎 append 也不会改调用方列表。
        self.token_ids = copy(token_ids)
        # 缓存最后一个 token：decode 阶段每步只需要喂这一个 token 进模型
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)       # 当前总长度（prompt + 已生成）
        self.num_prompt_tokens = len(token_ids)     # prompt 长度，之后不变

        # 下面两个是 chunked prefill / prefix cache 的核心计数器：
        # num_cached_tokens：前多少个 token 的 KV 已经算好并写进 cache 了
        self.num_cached_tokens = 0
        # num_scheduled_tokens：本次 step 调度器分配给它要算的 token 数
        self.num_scheduled_tokens = 0

        self.is_prefill = True
        # is_prefill 表示下一次执行/序列化的模式，不等价于 status == WAITING。
        # 调度到最后一个 prefill 分块时，status 已是 RUNNING，模式仍为 prefill。
        # 页表：block_table[i] 是这条序列第 i 个逻辑 block 对应的物理 block id
        self.block_table = []

        # 把采样参数摊平进来，省得每次都要间接访问
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        # Python 魔术方法：len(seq) 等价于 seq.__len__()，返回 token 总数。
        return self.num_tokens

    def __getitem__(self, key):
        # 支持 seq[a:b] 这种切片取 token，prepare_prefill 里会用到
        return self.token_ids[key]

    @property
    def is_finished(self):
        # @property 让方法像只读属性一样访问：seq.is_finished，而非 seq.is_finished()。
        # 以下各 property 都是访问时计算，不是另外保存一份容易过期的状态。
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        # list 切片 [start:] 会创建新列表；它只包含生成部分，不含 prompt。
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        # 向上取整：最后一个 block 没装满也要占一整页
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        # 最后一个 block 里实际装了多少 token（1 ~ block_size）
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """取第 i 个逻辑 block 对应的 token 列表，用来算 hash 做 prefix cache 匹配。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        # 采样得到 token id 不等于已算出它的 KV。
        # 正常继续 decode 时：num_tokens = num_cached_tokens + 1；
        # 最后这一个 token 要在下一次 forward 才被投影为 K/V 并写入 cache。
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # __getstate__ / __setstate__ 控制 pickle 的序列化内容。
    # 张量并行时 rank0 要把 seqs 通过共享内存发给其它 rank，
    # 这里只传必要字段来压缩数据量：
    #   - prefill 阶段其它 rank 需要完整 token_ids 来构造输入
    #   - decode 阶段每步只用到 last_token，就只传一个 int
    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    def __setstate__(self, state):
        # pickle 恢复对象时走这里，不重新执行 __init__。
        # worker 只得到推理需要的字段，不是完整调度对象：没有 status/temperature 等；
        # worker 不调用调度/采样逻辑，也不能用 len(token_ids) 代替 num_tokens。
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        # 用类型来区分传过来的是整串 token 还是单个 last_token
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
