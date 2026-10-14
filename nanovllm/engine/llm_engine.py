import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """对外的总入口。把 tokenizer、调度器、模型执行器粘在一起，
    对外暴露 generate()，内部是经典的 “while not finished: step()” 循环。"""

    # 阅读主线（LLM 直接继承本类）：
    # generate -> add_request -> Sequence 入队
    #          -> step -> Scheduler.schedule -> ModelRunner.run
    #                  -> Scheduler.postprocess -> 收集完成结果 -> tokenizer.decode
    # 调度和 token 列表在 CPU/Python 侧；模型权重、激活和真正的 KV cache 在 GPU。

    def __init__(self, model, **kwargs):
        # **kwargs 在函数定义里收集额外关键字参数，类型为 dict[str, object]。
        # fields(Config) 返回 dataclass 的字段描述；下面分别是 set / dict comprehension。
        # config_kwargs 的展开写法：
        # config_kwargs = {}
        # for k, v in kwargs.items():
        #     if k in config_fields:
        #         config_kwargs[k] = v
        # 只挑 Config 里真实存在的字段，其余 kwargs 静默忽略（方便和 vllm 的接口兼容）
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 调用时 ** 则反过来展开字典：如 Config(model, tensor_parallel_size=2)。
        config = Config(model, **config_kwargs)
        # 把页大小同步到 Sequence 的类变量上，全局生效
        Sequence.block_size = config.kvcache_block_size

        self.ps = []
        self.events = []
        # 张量并行：rank 1..N-1 各起一个子进程。
        # 必须用 spawn（不是 fork）——CUDA 上下文不能跨 fork 继承
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            # 注意 target 直接就是 ModelRunner 类：子进程里构造它时，
            # 构造函数末尾会进入 loop() 一直阻塞等命令，所以进程不会退出
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        # 本进程是 rank 0（主控），持有所有子进程的 event 用来广播命令
        self.model_runner = ModelRunner(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # eos 要等 tokenizer 加载完才知道，这里回填进 config 再传给调度器
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        # 注册正常解释器退出时的清理回调；强制终止进程时不保证执行
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """塞一条请求进等待队列。prompt 可以是字符串，也可以是已经编码好的 token id。"""
        # str | list[int] 是类型提示（两种类型均可），Python 不会自动按它校验输入。
        # encode 后 prompt 是 list[int]：词表编号，尚不是 Tensor 或 embedding 向量。
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """推进一步：调度 -> 跑模型 -> 后处理。返回 (本步完成的序列, 本步 token 数)。"""
        # generate 每轮调用一次。seqs: list[Sequence]，is_prefill: bool（元组解包）。
        # Sequence 对象是队列中同一对象的引用，后处理会直接更新它们的状态。
        seqs, is_prefill = self.scheduler.schedule()
        # 用正负号编码阶段：正数 = prefill 处理的 token 数，负数 = decode 的序列数。
        # 这样下面统计吞吐时一个变量就够了，省得再传一个 flag
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        # 上面的 sum 接收 generator expression，逐项累加而不创建中间列表。
        # 计数必须在 postprocess 前取：后处理会把 num_scheduled_tokens 清零。
        # token_ids: list[int]，顺序与 seqs 一致，每条返回一个候选 token；
        # 未完成的 prefill 分块还不能续写，postprocess 会丢弃其候选结果。
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        # 只返回本轮刚完成的请求，不是流式返回每轮的新 token。等价展开：
        # outputs = []
        # for seq in seqs:
        #     if seq.is_finished:
        #         outputs.append((seq.seq_id, seq.completion_token_ids))
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """批量生成。所有 prompt 一次性入队，然后不停 step 直到全部跑完。"""
        # 实际返回 list[dict]，每项为 {"text": str, "token_ids": list[int]}；
        # 上方遗留的 -> list[str] 标注不符合实际值，调用方应按下面的字典结构取结果。
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 允许所有请求共用一份采样参数
        if not isinstance(sampling_params, list):
            # 列表乘法重复同一对象的引用，不会创建多份 SamplingParams。
            # 此处 Sequence 会读取其中的标量字段，所以共用配置即可。
            sampling_params = [sampling_params] * len(prompts)
        # zip 按位置配对；两边长度不同会在较短的一边结束，本函数没有长度校验。
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            # 靠 num_tokens 的正负区分这步是 prefill 还是 decode，分别统计吞吐
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()

        # 完成顺序是乱的（谁先生成完谁先进字典），按 seq_id 排回输入顺序
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
