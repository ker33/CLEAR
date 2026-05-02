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
    # 1. 基础 Q, K, V 与位置编码
    # ==========================================================
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, getattr(self, "num_key_value_heads", self.num_heads), self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, getattr(self, "num_key_value_heads", self.num_heads), self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, getattr(self, "layer_idx", 0))

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, getattr(self, "layer_idx", 0), {"sin": sin, "cos": cos})

    if hasattr(self, "num_key_value_groups") and self.num_key_value_groups > 1:
        from transformers.models.llama.modeling_llama import repeat_kv
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

    # =========================================================================
    # 🌟 V7 概率空间重整化 (Probability-Space Intervention, PSI)
    # 💡 我们将 Softmax 提前！在真实的概率域上进行降维打击！
    # =========================================================================
    attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

    if hasattr(self, "use_intervention") and self.use_intervention:
        img_s = getattr(self, "img_start_idx", -1)
        img_e = getattr(self, "img_end_idx", -1)
        
        # 仅在生成答案阶段（q_len==1）且图片存在时启动
        if q_len == 1 and img_s != -1 and img_e != -1 and img_e <= kv_seq_len:
            with torch.no_grad():
                v_mass = attn_probs[:, :, -1, img_s:img_e].sum()
                t_mass = attn_probs[:, :, -1, :].sum() - v_mass
                v_ratio = v_mass / (v_mass + t_mass + 1e-9)

            # 诊断：模型看图的注意力是否跌破阈值
            if v_ratio < self.threshold:
                object_mask = getattr(self.master_model, 'current_object_mask', None)

                # 克隆一份用于修改，防止原地操作报错
                new_probs = attn_probs.clone()

                # 💡 魔法1：构建平滑的缩放矩阵 (基底为 1.0，即不干预)
                scale_factor = torch.ones_like(new_probs[:, :, -1, img_s:img_e])

                # 💡 魔法2：全局无损底色提升 (拯救 MME)
                # 所有视觉 Token 统一放大 10% (alpha * 0.1) 的概率
                # 这逼迫模型多看图，且完全不破坏 OCR、空间、数量的底层结构！
                scale_factor += self.alpha * 0.1

                # 💡 魔法3：高亮目标精确放大 (拯救 POPE & CHAIR)
                if object_mask is not None and object_mask.shape[-1] == (img_e - img_s):
                    if object_mask.max() > 0.3:
                        # 对于明确的物体，概率进一步放大 up to 90% (alpha * 0.9)
                        scale_factor += self.alpha * 0.9 * object_mask

                # 在概率域进行线性乘法
                new_probs[:, :, -1, img_s:img_e] = new_probs[:, :, -1, img_s:img_e] * scale_factor

                # 💡 魔法4：概率流重整化 (Re-normalization)
                # 这会像弹簧一样，平滑、无损地将多出来的概率从“文本 (先验幻觉)”那边挤压过来
                # 彻底消除负值/绝对值带来的畸变！
                new_probs[:, :, -1, :] = new_probs[:, :, -1, :] / new_probs[:, :, -1, :].sum(dim=-1, keepdim=True)

                attn_probs = new_probs

    # ==========================================================
    # 最终输出 (直接用修改后的 probs 乘 Value)
    # ==========================================================
    attn_probs = attn_probs.to(query_states.dtype)
    attn_output = torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_probs if output_attentions else None, past_key_value


def apply_mask_guided_intervention(model, start_layer=18, end_layer=28, threshold=0.50, alpha=1.0):
    """
    V7 终极参数：
    threshold=0.20 (视觉注意力低于 20% 唤醒)
    alpha=1.0 (概率最高放大 2 倍，柔和且极其稳定)
    """
    for i in range(start_layer, end_layer):
        layer = model.model.layers[i].self_attn
        layer.use_intervention = True
        layer.threshold = threshold
        layer.alpha = alpha
        layer.master_model = model.model 
        layer.forward = types.MethodType(llama_new_forward, layer)