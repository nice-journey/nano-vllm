import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    # enforce_eager=True 跳过 CUDA Graph 捕获，启动快、方便调试
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    # 套上对话模板（加 <|im_start|>user 之类的标记），
    # 对话模型通常应使用与训练一致的格式。tokenize=False 是因为
    # LLM.generate 内部会自己编码
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,    # 结尾补上 assistant 起始标记，引导模型开始答
        )
        for prompt in prompts
    ]
    # 两条请求会被引擎自动组成一个 batch 跑，不是串行
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        # output 是 {"text": str, "token_ids": list[int]}；f-string 中 !r 使用 repr，
        # 会显示引号及换行转义，便于检查实际文本，而非额外的模型控制符。
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    # 仅直接运行此文件时执行 main；spawn 子进程导入模块时不会再次创建引擎/子进程。
    main()
