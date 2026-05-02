import math
import types
from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    bsz, q_len, _ = hidden_states.size()

    # ==========================================================
    # 1. 计算 Q, K, V
    # ==========================================================
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, getattr(self, "num_key_value_heads", self.num_heads), self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, getattr(self, "num_key_value_heads", self.num_heads), self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if getattr(self, "layer_idx", None) is None:
            raise ValueError(
                f"The cache structure has changed. Please initialize the attention class with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # ==========================================================
    # 2. 旋转位置编码 (RoPE) 和 KV 缓存更新
    # ==========================================================
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # 为了兼容新版 transformers 中的 GQA
    if hasattr(self, "num_key_value_groups") and self.num_key_value_groups > 1:
        from transformers.models.llama.modeling_llama import repeat_kv
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    # ==========================================================
    # 3. 基础注意力权重计算
    # ==========================================================
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

    # =========================================================================
    # 🌟 核心创新：基于 VTACR 诊断与 mask_R 引导的动态因果干预
    # =========================================================================
    if hasattr(self, "use_intervention") and self.use_intervention:
        img_start = getattr(self, "img_start_idx", -1)
        img_end = getattr(self, "img_end_idx", -1)
        text_start = getattr(self, "text_start_idx", -1)
        text_end = getattr(self, "text_end_idx", -1)
        
        # 【极其关键】 q_len == 1 表示当前处于逐词推理阶段，此时才需要进行幻觉干预！
        if q_len == 1 and img_start != -1 and img_end != -1 and text_start != -1 and text_end != -1:
            # 确保序列长度对齐，防止索引越界
            if img_end <= kv_seq_len and text_end <= kv_seq_len:
                with torch.no_grad():
                    # 算一下临时概率，仅用作诊断
                    attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
                    
                    v_score = attn_probs[:, :, -1, img_start:img_end].mean()
                    t_score = attn_probs[:, :, -1, text_start:text_end].mean()
                    
                    # 诊断指标 VTACR
                    vtacr = v_score / (t_score + 1e-9)

                # 诊断：如果看图太少 (产生幻觉风险)
                if vtacr < self.threshold:
                    # 拿到训练时得到的纯净物体掩码 mask_R [B, 1, 1, 576]
                    object_mask = getattr(self.master_model, 'current_object_mask', None)
                    
                    # a. 图像区域靶向放大
                    if object_mask is not None and object_mask.shape[-1] == (img_end - img_start):
                        # 空间因果干预：原权重 + 绝对值 * 放大系数 * 物体掩码权重
                        attn_weights[:, :, -1, img_start:img_end] += (
                            attn_weights[:, :, -1, img_start:img_end].abs() * self.alpha * object_mask
                        )
                    else:
                        # 兜底：退化为无差别干预
                        attn_weights[:, :, -1, img_start:img_end] += (
                            attn_weights[:, :, -1, img_start:img_end].abs() * self.alpha
                        )
                    
                    # b. 文本区域整体抑制
                    attn_weights[:, :, -1, text_start:text_end] -= (
                        attn_weights[:, :, -1, text_start:text_end].abs() * self.b
                    )

    # ==========================================================
    # 4. Softmax 与最终输出
    # ==========================================================
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def apply_mask_guided_intervention(model, start_layer=0, end_layer=32, threshold=1.0, alpha=0.4, b=0.2):
    """
    在推理前调用此函数，将注意力机制替换为我们的靶向干预版本。
    """
    for i in range(start_layer, end_layer):
        layer = model.model.layers[i].self_attn
        layer.use_intervention = True
        layer.threshold = threshold
        layer.alpha = alpha
        layer.b = b
        
        # 让 attention 层能拿到全局的 model 实例，从而获取 current_object_mask
        layer.master_model = model.model 
        
        # 替换前向传播
        layer.forward = types.MethodType(llama_new_forward, layer)
