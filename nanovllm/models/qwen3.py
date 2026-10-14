import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


# 本文件的 forward 属于 ModelRunner.run_model 的 GPU 主干，不负责调度请求。
# 记 T=本轮输入 token 数，H=hidden_size，D=head_dim，p=TP 卡数，I=intermediate_size。
# hidden_states 是 CUDA 模型 dtype 的 [T,H]，每行表示一个 token 的特征，而不是 token id。
# prefill 的 T 是各条本轮分块长度之和；decode 的 T 等于序列数。

class Qwen3Attention(nn.Module):
    """Qwen3 的注意力块。相比标准实现有两个特点：
    GQA（k/v 头数少于 q 头数，省 KV cache），以及 QK-Norm（对 q/k 逐头做 RMSNorm）。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        # total_* 是全局的头数，num_* 是本卡分到的头数
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 本卡 q/kv 在拼接张量里各占多宽，forward 里按它切分
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 1/sqrt(d)，防止点积随维度增大而方差过大
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        # q/k/v 融合成一次矩阵乘
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 输出投影用行并行：输入维正好是上面列并行切出来的那一段，衔接上不用通信
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # 本实现按 qkv_bias 开关选择是否启用 QK-Norm；这不是两者在数学上必须互斥
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # positions: CUDA int64 [T]；hidden_states: CUDA 模型 dtype [T,H]。
        # Q 表示当前 token 要查询什么，K 表示供匹配的特征，V 表示要聚合的信息。
        # 每卡 qkv 输出 [T, (本卡 Q 头数 + 2*本卡 KV 头数)*D]。
        qkv = self.qkv_proj(hidden_states)
        # 一次矩阵乘的结果按 [q | k | v] 切开
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 拆出 head 维。注意这里是 [token数, head数, head_dim]，
        # 没有 batch 维——变长批已经被拍平成一维序列了
        # split 返回共享存储的视图；view(-1, heads, D) 让 -1 自动推导 T，
        # 只把最后一维拆成 heads 和 D，不复制数据。view 要求 stride 与目标布局兼容。
        # q: [T,Q_heads,D]，k/v: [T,KV_heads,D]；这里的头数都是当前 rank 的局部头数。
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            # 在最后一维（head_dim）上做 norm，即逐头归一化
            q = self.q_norm(q)
            k = self.k_norm(k)
        # 顺序要求：先 norm 再 RoPE，最后才写 KV cache（在 self.attn 内部），
        # 这样缓存里存的是已经带好位置信息的 k
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        # Attention 输出：prefill [T,Q_heads,D]，decode [B,1,Q_heads,D]。
        # flatten(1,-1) 把第 1 维到末维合并为 [T,Q_heads*D]，交给输出投影；
        # 布局兼容时是 view，否则 flatten 可复制。o_proj all_reduce 后恢复 [T,H]。
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen3MLP(nn.Module):
    """SwiGLU 前馈网络。gate 和 up 两个投影被融合成一个 gate_up_proj。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # 列并行 -> 激活 -> 行并行，整个 MLP 只在 down_proj 末尾做一次 all_reduce
        # CUDA 模型 dtype：[T,H] -> gate_up [T,2*I/p] -> SwiGLU [T,I/p]
        # -> down_proj 的局部和 [T,H] -> all_reduce 得到完整 [T,H]。
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """一层 Transformer block：注意力 + MLP，都是 Pre-Norm + 残差。"""

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 残差加法延迟到紧随其后的 norm：attention 残差在本层 post_attention_layernorm 加，
        # MLP 残差在下一层 input_layernorm（或最终 norm）加。避免单独启动一次加法。
        # hidden_states/residual 均为 CUDA 模型 dtype [T,H]；第一次 residual 为 None。
        # 理解计算关系可写成下面的数学展开（不是替换当前实现）：
        # a = 输入残差流
        # b = a + Attention(RMSNorm(a))
        # c = b + MLP(RMSNorm(b))
        # 当前层返回 (MLP 的输出, b)，下一次 norm 再把两者相加为 c。
        if residual is None:
            # 多重赋值先求右侧全部值再赋给左侧，因此 residual 接到的是归一化前的输入。
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        # 这一步同时完成了“加上注意力的残差”和“MLP 前的归一化”
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # ModuleList 会向 PyTorch 注册子模块，使 modules()/get_parameter() 能找到层及权重；
        # 普通 Python list 不会自动注册。推导式中的 _ 表示不使用循环下标。
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # int64 token ids [T] -> embedding [T,H] -> 多层 attention/MLP -> 最终 [T,H]。
        # embedding 查表产生浮点特征；positions 只传给各 attention 的 RoPE，不直接相加。
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后这次 norm 顺手把最后一层遗留的残差加上
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    # 告诉 loader：HF 权重里的这些名字要映射到融合后的参数上。
    # 值是 (目标参数名, 分片标识)——qkv 用字符串标识，gate_up 用下标
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        # 权重绑定：小模型常让输入 embedding 和输出层共用一份权重，省参数量。
        # 这里保留两个 Parameter 对象，但令它们的 .data 共享存储，节省权重显存；
        # 不能因此认为 state_dict 会自动删除其中一个参数名
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # 注意 forward 只算到 hidden_states 为止，不含 lm_head。
        # 这是为了配合 CUDA Graph：graph 里只捕获主干，lm_head 在外面按真实 batch 算
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
