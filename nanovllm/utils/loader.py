import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """没有自定义 loader 的参数（比如各种 norm 的 weight）走这里，直接整份拷贝。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """从 safetensors 文件加载权重。

    难点在于本项目为了性能做了两类改造，权重名字和 HF 原始的对不上：
      1. 算子融合：q/k/v 三个 proj 合成一个 qkv_proj，gate/up 合成 gate_up_proj
      2. 张量并行：每个 rank 只该拿权重的一个分片
    因而这里不直接调用 load_state_dict，而是通过各参数的 weight_loader 完成转换。"""
    # ModelRunner 初始化时每个 rank 执行一次，不在每 token 推理的热路径中。
    # 文件 -> CPU loaded_weight -> 本 rank 对应的切片 -> copy_ 到 CUDA Parameter。
    # loaded_weight dtype 来自 checkpoint；目标参数 dtype 由 hf_config.dtype 决定。
    # 形如 {"q_proj": ("qkv_proj", "q"), ...}：原始名 -> (融合后的名字, 分片标识)
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        # safe_open 是懒加载，用到哪个张量才读哪个，不会一次性把文件读进内存
        # with 调用上下文管理器，退出时（包括异常退出）负责关闭文件资源。
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        # 命中融合映射：改名后找到合并参数，再按 shard_id 写进对应的那一段
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        # Parameter 上挂的是绑定方法，调用时 self 已绑定；param 是显式传入的权重。
                        # 例如 checkpoint 的 q_proj.weight -> qkv_proj.weight 中本卡的 Q 区段。
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    # for-else：循环没被 break，说明不是融合权重，按普通权重加载。
                    # 这里的 weight_loader 由各 Linear 层自己挂上，内部会做张量并行切分
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
