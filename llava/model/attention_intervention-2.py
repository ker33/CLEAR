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
    # 🌟 V4 认知无损干预 (Cognition-Preserved Intervention)
    # =========================================================================
    if hasattr(self, "use_intervention") and self.use_intervention:
        img_s = getattr(self, "img_start_idx", -1)
        img_e = getattr(self, "img_end_idx", -1)
        
        # 仅在生成答案阶段（逐词推理）触发
        if q_len == 1 and img_s != -1 and img_e != -1 and img_e <= kv_seq_len:
            t1_s, t1_e = 0, img_s             # 图前文本
            t2_s, t2_e = img_e, kv_seq_len    # 图后文本 + 已生成的历史答案

            with torch.no_grad():
                attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
                
                # 💡 核心修正1：使用 sum 计算全局注意力占比，而非 mean！
                v_mass = attn_probs[:, :, -1, img_s:img_e].sum()
                
                t_mass = 0.0
                if t1_e > t1_s: t_mass += attn_probs[:, :, -1, t1_s:t1_e].sum()
                if t2_e > t2_s: t_mass += attn_probs[:, :, -1, t2_s:t2_e].sum()
                
                # 计算模型看图的总精力百分比
                v_ratio = v_mass / (v_mass + t_mass + 1e-9)

            # 💡 核心修正2：只有当看图占比严重不足（阈值设为比如 15%）时，才判定为幻觉
            if v_ratio < self.threshold:
                object_mask = getattr(self.master_model, 'current_object_mask', None)

                # --- 靶向增强图片 (不再打压文本！) ---
                if object_mask is not None and object_mask.shape[-1] == (img_e - img_s):
                    # 💡 核心修正3：掩码置信度门控 (Confidence Gating)
                    # 如果 MME 是 OCR/数学题，mask_R 提取不到焦点，最大值可能很低。
                    mask_max = object_mask.max()
                    
                    if mask_max > 0.3: 
                        # 找到了明确焦点物体：启动强空间干预
                        # 自适应放大：偏离阈值越多，补救力度越大
                        deficit = self.threshold - v_ratio.item()
                        cur_alpha = self.alpha * (1.0 + deficit * 5.0) 
                        
                        # 保留 20% 全局视野，防止变成管状视野，80% 压在物体上
                        spatial_weight = 0.2 + 0.8 * (object_mask ** 2)
                        attn_weights[:, :, -1, img_s:img_e] += (
                            attn_weights[:, :, -1, img_s:img_e].abs() * cur_alpha * spatial_weight
                        )
                    else:
                        # 没找到明确焦点 (如代码/数学题)：进行轻微的均匀增强，绝不误导视线
                        attn_weights[:, :, -1, img_s:img_e] += attn_weights[:, :, -1, img_s:img_e].abs() * (self.alpha * 0.3)
                else:
                    # 兜底
                    attn_weights[:, :, -1, img_s:img_e] += attn_weights[:, :, -1, img_s:img_e].abs() * (self.alpha * 0.5)

                # 💡 核心修正4：删除了所有针对文本 (t1, t2) 的减法！
                # 保护 LLM 的逻辑推理链条与语言流利度！

    # ==========================================================
    # 最终输出
    # ==========================================================
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights if output_attentions else None, past_key_value


def apply_mask_guided_intervention(model, start_layer=15, end_layer=30, threshold=0.15, alpha=0.3):
    """
    默认参数已全面优化为认知保护模式。删除了破坏性的 b 参数。
    """
    for i in range(start_layer, end_layer):
        layer = model.model.layers[i].self_attn
        layer.use_intervention = True
        layer.threshold = threshold
        layer.alpha = alpha
        layer.master_model = model.model 
        layer.forward = types.MethodType(llama_new_forward, layer)