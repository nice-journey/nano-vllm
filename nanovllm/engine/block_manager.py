from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """KV cache 里的一个物理页。注意它本身不存 KV 张量，只存元数据；
    真正的 KV 数据在 ModelRunner 里一整块大张量上，用 block_id 做下标定位。"""

    def __init__(self, block_id):
        self.block_id = block_id
        # 引用计数：同一个 block 可能被多条共享前缀的序列同时引用
        self.ref_count = 0
        # 这一页内容的 hash（含前缀链），-1 表示还没算过 / 无效
        self.hash = -1
        # 这一页对应的 token 列表，用来在 hash 碰撞时做二次校验
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """页被填满后登记它的 hash，之后才能被别的序列复用。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重新分配给新序列时调用：引用计数置 1，清掉旧的 hash 信息。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """分页式 KV cache 的分配器，相当于显存上的一个“内存管理单元”。
    核心能力有两个：分页（每条序列无需占连续物理页）、前缀复用（prefix cache）。
    分页减少外部碎片；最后一页未填满仍有内部碎片，单条序列最多浪费 block_size-1 个槽位。"""

    # 注意这里的分配/释放只是 CPU 元数据管理，GPU 大张量在启动时已由 ModelRunner 分配。
    # 一项 block_id 同时对应所有 Transformer 层各自的 K/V 页，页内存本卡的 KV heads。
    # 例：block_size=256，seq.block_table=[7, 2]，逻辑位置 260 -> 第 1 页偏移 4
    #    -> 物理页 2 -> slot=2*256+4。Attention 用 slot 写入，用页表读取历史 KV。

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash -> block_id 的索引，prefix cache 靠它做 O(1) 查找
        self.hash_to_block_id: dict[int, int] = dict()
        # 空闲页队列。用 deque 从左边取、右边放，近似 FIFO，
        # 这样刚释放的页不会马上被抢走，它身上的 hash 还能多留一会儿被复用
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """算一页的 hash。关键点是把上一页的 hash 作为 prefix 混进来，
        形成一条 hash 链，让缓存键同时依赖本页及之前的上下文。
        hash 不是无碰撞证明；can_allocate 还会比对本页 token，但不会完整比对历史前缀。"""
        # @classmethod 自动传入类 cls，而不是实例 self；本方法没有用到实例状态。
        # 必须包含前缀：同一 token 块接在不同历史之后，经 attention 得到的 KV 会不同。
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        """从空闲队列里取一页出来用。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 这一页身上可能还挂着上一任主人的 hash 索引，现在要被覆写了，
        # 必须先把索引删掉，否则后面有人按 hash 查过来会读到脏数据。
        # 判断 get(hash) == block_id 是因为可能已有新页顶替了这个 hash 条目
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """把页还回空闲队列。注意这里不清 hash，让它保留着好被 prefix cache 命中。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """试算这条序列能不能分配下来。
        返回 -1 表示空闲页不够；否则返回能从 prefix cache 命中复用的页数。"""
        h = -1
        num_cached_blocks = 0
        # 当前序列所需的总页数；命中正在使用的共享页时才减少对空闲页的需求
        num_new_blocks = seq.num_blocks
        # 本实现总是排除最后一页，即使它恰好装满；因此至少重算一个 token，
        # 才能拿到最后位置的 hidden state/logits。KV cache 本身没有保存这两个结果。
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 两个条件缺一不可：hash 要查得到，且 token 要逐个对得上（防 hash 碰撞）。
            # 本实现只复用连续前缀：遇到缺页就停止，不去尝试后续零散命中
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            # 命中的页如果还在使用中，说明它还活着，直接加引用即可，不消耗新页；
            # 如果它已经躺在 free 队列里，仍然要从 free 里取出来，所以算作要占一页
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        """真正建页表。num_cached_blocks 是上面 can_allocate 算出来的命中页数。"""
        assert not seq.block_table
        h = -1
        # 前 num_cached_blocks 页直接复用已有物理页
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                # 还在用 —— 多一个使用者，加引用计数
                block.ref_count += 1
            else:
                # 在空闲队列里“复活”它；deque.remove 要查找该元素，时间复杂度为 O(n)
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        # 剩下的页只能新分配
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        # 命中的这些 token 的 KV 已经在 cache 里了，不需要重算，直接跳过
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """序列结束或被抢占时释放它的所有页。
        倒序释放是为了让靠前的（更可能被别人复用的）前缀页更晚进入空闲队列尾部。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            # 只有引用计数归零才真的还回去，共享前缀的页不会被误释放
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """decode 一步前检查页够不够。
        len(seq) % block_size == 1 意味着刚追加到 token 列表的 token 正好开了新的一页
        （比如 block_size=256，长度变成 257 时第 2 页才刚起头），这时才需要一页新的。
        布尔值在比较里会当成 1/0 用，所以其它情况只要求空闲数 >= 0，恒成立。"""
        # 这个最新 token 尚无 KV；may_append 为它预留槽位，随后 forward 才写入。
        # 等价展开：
        # needed = 1 if len(seq) % self.block_size == 1 else 0
        # return len(self.free_block_ids) >= needed
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """与 can_append 同样的判据，跨页时补一页进页表。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """本次 step 算完后，把新填满的页登记进 hash 索引，供后续请求复用。"""
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        # 整除的写法保证只处理“刚好填满”的页，没填满的页不会被登记
        if start == end: return
        # 接着上一页的 hash 往下链；start==0 时没有前缀，用 -1
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
