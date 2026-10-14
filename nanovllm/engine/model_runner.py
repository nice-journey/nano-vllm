import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:
    """真正和 GPU 打交道的一层：加载模型、分配 KV cache、把调度结果转成张量喂给模型。
    张量并行时每个 rank 一个实例，rank 0 是主控，其余 rank 在 loop() 里等命令。"""

    # 下面统一用 B=本轮序列数，T=本轮实际输入 token 总数，H=hidden_size，V=词表大小。
    # prefill: T=sum(seq.num_scheduled_tokens)；decode: T=B。
    # 运行时主线：Sequence/CPU -> prepare_* / CUDA Tensor -> Transformer
    #            -> hidden_states [T,H] -> LMHead / logits [B,V] -> 采样 -> Python token 列表。
    # tensor parallel 的每个 rank 处理相同的 token batch，但持有不同权重/KV 分片；
    # 它不是每卡分配不同请求的数据并行，collective 通信必须由所有 rank 同序参与。

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        # rank 0 拿到的是所有子进程 event 的列表（用于广播）；子进程拿到的是自己那一个
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        # 临时把默认 dtype/device 改成模型的 dtype 和 cuda，
        # 这样下面构造模型时所有参数直接就建在 GPU 上、且 dtype 正确，不用再 .to()
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 顺序很重要：先 warmup 跑一次峰值负载，才能测出模型本身要占多少显存，
        # 然后在 gpu_memory_utilization 预算内为 KV cache 分配空间
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        # 恢复原 dtype，并把默认 device 设为 CPU（这里并没有保存/恢复调用方原 device）
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            # 用一块共享内存在进程间传命令。barrier 保证 rank0 先建好，其它 rank 再 attach
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                # 子进程到这里就不返回了，一直在循环里听命令
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            # unlink 只能由创建者做一次，否则会报错
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """子进程主循环：读命令 -> 执行 -> 直到收到 exit。"""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        # 阻塞等 rank0 把数据写好并 set event
        self.event.wait()
        # 前 4 字节是长度头，后面是 pickle 过的 [方法名, *参数]
        n = int.from_bytes(self.shm.buf[0:4], "little")
        # 星号解包收集剩余项：若消息为 ["run", seqs, True]，
        # 等价于 message=pickle.loads(...); method_name=message[0]; args=message[1:]。
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        # 挨个唤醒子进程
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """统一的调用入口。rank0 调用时会先把同样的命令广播出去，
        这样所有 rank 执行同一个方法，保持 SPMD 的步调一致。"""
        # 定义里的 *args 收集位置参数为 tuple；调用时 *args 把它们展开。
        # call("run", seqs, True) 中 getattr(self, "run") 取得绑定了 self 的方法，
        # method(*args) 等价于 self.run(seqs, True)。共享内存传 CPU 元数据，不传 GPU KV。
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """用最大负载空跑一次，目的有二：触发 torch.compile / kernel 的初始化，
        以及测出模型前向的显存峰值，供 allocate_kv_cache 扣除。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        # 假序列：内容全 0，且没有 block_table（prepare_prefill 里会据此跳过 slot_mapping）
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """按剩余显存算出能开多少个 block，一次性分配一整块大张量。"""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 张量并行下 KV head 也被切了，每个 rank 只存自己那份
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 一页的字节数：2(K和V) * 层数 * 页内 token 数 * KV头数 * head_dim * 每元素字节
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        # 预算 = 总显存*利用率 - 已用 - (峰值 - 当前)。
        # 减 (peak - current) 是给前向过程中的临时激活留出空间，否则跑起来会 OOM
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # CUDA Tensor，dtype=hf_config.dtype（通常 fp16/bf16，取决于加载的模型配置）。
        # shape [2, L, N, S, K, D]：K/V、层数、物理页数、每页 token 数、本卡 KV 头数、头维度。
        # torch.empty 不初始化内容；只有已写入且被有效页表/长度覆盖的区域才能读取。
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # 把大张量按层切开，分别绑到每层 Attention 模块上。
        # 这是视图不是拷贝，所以各层写自己的 k_cache 等于写进这块总 cache
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """把各条序列长短不一的页表 pad 成一个矩形张量，-1 表示无效位。
        pin_memory + non_blocking 允许异步 H2D 拷贝；不保证本代码中一定与计算重叠。"""
        # 输出：CUDA int32 [B, P]，P=本批最长页表长度，各元素是物理页编号。
        # 补齐的是页表元数据，不是把各条 prompt 的 token 也补成一样长。
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """构造 prefill 的输入。多条序列的 token 首尾相接拼成一维（变长批），
        用 cu_seqlens（前缀和）告诉 FlashAttention 每条序列的边界在哪。"""
        # run 每个 prefill step 调用；包含首轮、chunk 续算、抢占后的重新计算。
        # 例：两条序列本轮分别算 3、2 个 token -> T=5，cu_seqlens_q=[0,3,5]。
        # 若第一条已有 4 个缓存 token，cu_seqlens_k=[0,7,9]：两条可见 KV 长度为 7、2。
        # positions 保存各序列自己的绝对位置，如 [4,5,6,0,1]，不是扁平 batch 的索引。
        # 边界元数据让 kernel 知道哪些 token 属于同一请求，避免不同请求互相 attention。
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens        # 已经算过的部分，跳过
            seqlen_q = seq.num_scheduled_tokens  # 这轮要算的 query 长度
            end = start + seqlen_q
            # k 的长度是从 0 开始的全部历史（含缓存部分），因为 attention 要看到前面所有 token
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            # 算 slot_mapping：这轮每个 token 的 KV 该写到整块 cache 的哪个槽位。
            # 槽位号 = 物理 block_id * block_size + 页内偏移
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    # 第一页可能是从中间接着写的（chunked prefill / prefix cache）
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    # 中间页写满
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 最后一页只写到 end 为止
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            # k 比 q 长说明有历史 KV：可能是 prefix cache，也可能是 chunked prefill 续算
            block_tables = self.prepare_block_tables(seqs)
        # ------------------------------------------------------------
        # CPU 列表 -> pinned CPU Tensor -> 当前 rank 的 CUDA Tensor（发生数据拷贝）。
        # input_ids/positions: int64 [T]，分别是词表 id / 序列内绝对位置。
        # cu_seqlens_q/k: int32 [B+1]；slot_mapping: int32 [T]，存 KV 写入槽位。
        # warmup 没有页表，slot_mapping 为空，Attention 也会跳过 KV 写入。
        # pin_memory 固定 CPU 内存页，支持异步 DMA；non_blocking 避免每次拷贝后 CPU 等待。
        # 后续同一 CUDA stream 上的算子仍按顺序等待拷贝完成，不会读取半份数据。
        # ------------------------------------------------------------
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # 这些元信息通过全局 context 传给深处的 Attention 层，避免逐层透传参数
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """构造 decode 的输入。每条序列只喂 1 个 token，所以是规整的 [batch] 形状。"""
        # 输入是上一轮采样出来、尚未计算 KV 的 last_token；位置为 len(seq)-1。
        # 例：prompt 长 5，prefill 生成第 6 个 token；此轮输入它，position=5，
        # context_len=6（历史 5 个 + 本轮刚写入的 1 个），模型据此预测第 7 个 token。
        # 返回 input_ids/positions: CUDA int64 [B]；Context 中 slot_mapping/context_lens
        # 为 CUDA int32 [B]，block_tables 为 CUDA int32 [B,P]。CPU->GPU 拷贝与 prefill 相同。
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            # 新 token 写在最后一页的最后一个已用位置
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        # 仅 rank 0 调用；输出 CUDA float32 [B]，每条请求一个温度，与 logits 的行一一对应。
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """跑一次前向。decode 且 batch 不大时走 CUDA Graph，省掉大量 kernel launch 开销。"""
        # 装饰器在调用期间关闭 autograd 记录等训练开销；推理不需要 backward。
        # model(...) 走 nn.Module.__call__，再调用 forward，而不是重新构造模型。
        # 主干输出 CUDA 模型 dtype [T,H]；compute_logits 输出 rank0 的 [B,V]，
        # 其他 rank 在 TP gather 后返回 None。logits 是未归一化的词表分数，不是概率。
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 本实现没有捕获 prefill 的 graph，走普通执行路径；并非 prefill 原理上不能捕获
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # 挑一个 >= 当前 batch 的、最小的已捕获尺寸（向上取整到预设档位）
            # next(generator) 惰性取首个匹配值。等价于：
            # for size in self.graph_bs:
            #     if size >= bs:
            #         graph = self.graphs[size]
            #         break
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # CUDA Graph 回放的是固定地址上的张量，所以必须把本次数据拷进这些占位张量里
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # padding 部分要清干净：slot_mapping 填 -1 让 kernel 跳过写入，
            # context_lens 清零让这些假样本不去读任何 KV
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            # lm_head 在 graph 外面算：只取 bs 这部分，避免给 padding 也算一遍大矩阵乘
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # LLMEngine.step -> call("run", ...) -> 本函数；worker 也会执行同一轮。
        # 条件表达式选准备路径，再用元组解包取得两个 Tensor：
        # if is_prefill: input_ids, positions = self.prepare_prefill(seqs)
        # else:          input_ids, positions = self.prepare_decode(seqs)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 只有 rank0 做采样（它是唯一持有完整 logits 的），其它 rank 返回 None
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        # Sampler 返回 CUDA int64 [B]；tolist() 将结果传回 CPU、等待结果可用，
        # 变成 list[int] 交给 Scheduler.postprocess。worker 实际返回 None（签名未表达）。
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """为若干个预设的 batch size 各抓一张 CUDA Graph。
        decode 阶段每步的 kernel 又多又小，launch 开销占比很高，用 graph 能显著提速。"""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 这些张量是 graph 的固定输入输出缓冲区，地址必须一直不变
        # 此时仍在 __init__ 的默认 CUDA device 下创建；整数 dtype 显式指定，
        # outputs 是模型 dtype [max_bs,H]。它保存 hidden states，不是词表 logits。
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        # 小 batch 密一点、大 batch 按 16 递增，兼顾 padding 浪费和 graph 数量
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小抓：第一张（最大的）建立的内存池能被后面所有小 graph 复用，省显存
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            # with 是上下文管理器：进入时开始捕获，退出时结束捕获。
            # 捕获记录 GPU 工作及地址，之后 replay 不再逐层执行这段 Python；
            # 所以运行时必须 copy 到原缓冲区，不能用新 Tensor 替换字典中的对象。
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
