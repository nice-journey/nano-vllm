from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """调度器：每一步决定这一轮把哪些序列、各算多少 token 丢给 GPU。
    两条队列 + 一个策略：prefill 优先，prefill 没活干了才做 decode。"""

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # waiting：还没跑完 prefill 的（含被抢占退回来的）
        self.waiting: deque[Sequence] = deque()
        # running：已安排最后一个 prefill 分块，或正在逐 token 解码的
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """返回 (本轮要跑的序列, 是否是 prefill 批)。
        本实现把 prefill 和 decode 分开调度、走不同输入准备及 attention 路径。"""
        # LLMEngine.step 每轮调用一次；只操作 CPU 元数据，不执行模型。
        # waiting/running -> 挑选 seqs + 分配 KV 页 + 标记本轮 token 数 -> ModelRunner。
        # chunked prefill 的 chunk 是本轮计算量，KV block 是缓存分配单位，二者不同；
        # 当前实现切分计算，却仍在首次 allocate 时为整条现有序列预留 KV 页。
        scheduled_seqs = []
        num_batched_tokens = 0

        # ---------- 阶段一：prefill ----------
        # 新请求优先，让它们尽快进入 running，提高并发度
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                # 还没分配过页 —— 第一次调度它，先试分配
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    # 页不够就停止尝试队首后的请求；这是队首阻塞，不保证所有请求不饥饿
                    break
                # prefix cache 命中的部分不用重算
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 已经有页表 —— 是上一轮 chunked prefill 没算完的，接着往下算
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                # 装不下且批里已经有别的序列了：先发车，这条下一轮再说。
                # 只允许批里第一条做 chunked prefill，是为了让切块逻辑保持简单——
                # 批里至多有一条序列是“半截”的
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            # 算得下就全算，算不下就只算 remaining 这么多（chunked prefill）
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 预计本轮能算完现有 token，先转入 running；真正的 GPU 计算还没发生
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 否则留在 waiting 队首，下一轮继续算剩下的块
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # ---------- 阶段二：decode ----------
        # 本轮没有选出 prefill 请求（可能是 waiting 为空，也可能页不足），尝试 decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # while / else：只有 while 条件正常变为 False（即腾出了页）才走 else
            while not self.block_manager.can_append(seq):
                if self.running:
                    # 页不够：抢占 running 队尾；这是简单策略，没有比较各请求的重算成本
                    self.preempt(self.running.pop())
                else:
                    # 已经没人可牺牲了，只能抢占自己，break 跳过 else 分支
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1    # decode 每步固定只算 1 个 token
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        # 上面是 popleft 取出来的，这里原序放回队首。
        # extendleft 会反转顺序，所以先 reversed 抵消掉
        # 若 scheduled_seqs == [A, B]，等价于先 appendleft(B)，再 appendleft(A)。
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """抢占：把序列的 KV cache 全部释放、退回等待队列重跑 prefill。
        此实现选择重算，没有实现 KV 在 CPU/GPU 间换入换出；哪种更快取决于负载。
        appendleft 放队首，让它优先被尝试，能否调度仍取决于资源。"""
        # token_ids 不丢失：恢复时处理的是 prompt + 已生成 token。
        # deallocate 释放引用并清页表，不擦除 GPU 内容；未被覆盖的满页仍可能命中缓存。
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """模型跑完后更新状态：登记 hash、推进进度、接收新 token、判断结束。"""
        # 由 step 在 ModelRunner.run 返回后调用，每轮一次；zip 按对应位置配对。
        # 例如已有 4 个 KV，本轮计算 3 个 token：先登记覆盖到的完整页，再推进到 7。
        # 若现有 prompt 长度是 10，则不接收候选 token；若已算完才 append 新 token。
        for seq, token_id in zip(seqs, token_ids):
            # 把这轮新算满的页登记进 prefix cache 索引
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                # chunked prefill 的中间块：prompt 还没算完，这一步吐出来的 token 不作数
                continue
            seq.append_token(token_id)
            # 两种结束条件：撞到 eos（且没设 ignore_eos），或生成够了 max_tokens
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
