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
    # 1. 计算 Q, K, V 与位置编码
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

    # 基础注意力权重计算
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

    # =========================================================================
    # 🌟 进阶优化：精确区间定位 + 自适应强度 + 软掩码空间干预
    # =========================================================================
    if hasattr(self, "use_intervention") and self.use_intervention:
        img_s = getattr(self, "img_start_idx", -1)
        img_e = getattr(self, "img_end_idx", -1)
        
        # q_len == 1 表示生成答案阶段，且图片存在
        if q_len == 1 and img_s != -1 and img_e != -1 and img_e <= kv_seq_len:
            
            # 💡 优化1：完美切分三个区间，彻底避免交叉重叠！
            t1_s, t1_e = 0, img_s             # 区间1：图前文本 (系统提示词等)
            t2_s, t2_e = img_e, kv_seq_len    # 区间2：图后文本 (用户问题)

            with torch.no_grad():
                attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
                
                v_score = attn_probs[:, :, -1, img_s:img_e].mean()
                
                # 计算文本总得分 (前后文本加权平均)
                t_sum = 0.0
                t_count = 0
                if t1_e > t1_s:
                    t_sum += attn_probs[:, :, -1, t1_s:t1_e].sum()
                    t_count += (t1_e - t1_s)
                if t2_e > t2_s:
                    t_sum += attn_probs[:, :, -1, t2_s:t2_e].sum()
                    t_count += (t2_e - t2_s)
                
                t_score = t_sum / max(t_count, 1)
                vtacr = v_score / (t_score + 1e-9)

            # 触发诊断阈值
            if vtacr < self.threshold:
                # 💡 优化2：自适应干预强度 (病情越重，系数放大越多，上限1.5倍)
                deficit = max(0, self.threshold - vtacr)
                scale = min(1.0 + deficit, 1.5) 
                
                cur_alpha = self.alpha * scale
                cur_b = self.b * scale

                object_mask = getattr(self.master_model, 'current_object_mask', None)

                # --- 靶向增强图片 ---
                if object_mask is not None and object_mask.shape[-1] == (img_e - img_s):
                    # 💡 优化3：软掩码引导 (30%全局保底注意力 + 70%高亮物体精准注意力)
                    spatial_weight = 0.3 + 0.7 * object_mask
                    attn_weights[:, :, -1, img_s:img_e] += (
                        attn_weights[:, :, -1, img_s:img_e].abs() * cur_alpha * spatial_weight
                    )
                else:
                    attn_weights[:, :, -1, img_s:img_e] += attn_weights[:, :, -1, img_s:img_e].abs() * cur_alpha

                # --- 靶向抑制文本 ---
                if t1_e > t1_s:
                    attn_weights[:, :, -1, t1_s:t1_e] -= attn_weights[:, :, -1, t1_s:t1_e].abs() * cur_b
                if t2_e > t2_s:
                    attn_weights[:, :, -1, t2_s:t2_e] -= attn_weights[:, :, -1, t2_s:t2_e].abs() * cur_b

    # ==========================================================
    # 最终输出
    # ==========================================================
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights if output_attentions else None, past_key_value


def apply_mask_guided_intervention(model, start_layer=8, end_layer=30, threshold=1.0, alpha=0.4, b=0.2):
    """
    注意：默认参数调整为 8~30层，避免干扰最后两层的分类预测能力
    """
    for i in range(start_layer, end_layer):
        layer = model.model.layers[i].self_attn
        layer.use_intervention = True
        layer.threshold = threshold
        layer.alpha = alpha
        layer.b = b
        layer.master_model = model.model 
        layer.forward = types.MethodType(llama_new_forward, layer)