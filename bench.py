import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# 把上面那行注释掉、换成下面这行，就能用完全一样的代码去跑 vllm 做对比
# from vllm import LLM, SamplingParams


def main():
    seed(0)    # 固定 Python random 的输入/长度负载；不固定 PyTorch 的 GPU 采样结果
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    # 此压测启用 CUDA Graph（enforce_eager=False），衡量该执行模式下的性能
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 直接造随机 token id，不走 tokenizer：一来省时间，二来随机 token
    # 几乎不会有完整页的公共前缀，减少 prefix cache 命中带来的影响；仍包含调度/通信开销
    # 嵌套 list comprehension 展开：
    # prompt_token_ids = []
    # for _ in range(num_seqs):
    #     ids = []
    #     for _ in range(randint(100, max_input_len)):
    #         ids.append(randint(0, 10000))
    #     prompt_token_ids.append(ids)
    # 最终是 list[list[int]]，每个内部列表是一条长度可能不同的请求。
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]

    # ignore_eos=True 很关键：保证每条都老老实实生成满 max_tokens，
    # 否则提前遇到 eos 停下来，总 token 数就对不上了
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 先跑一条热身，把 torch.compile 编译、显存分配等一次性开销排除在计时之外
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    # 分子只数输出 token，但计时覆盖整个 generate（含 prefill），
    # 所以这是端到端输出吞吐，不是只对 decode 时间单独计算的吞吐
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
