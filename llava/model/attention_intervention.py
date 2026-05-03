import math
import types
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.generation.logits_process import LogitsProcessor

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

    attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

    if hasattr(self, "use_intervention") and self.use_intervention:
        img_s = getattr(self, "img_start_idx", -1)
        img_e = getattr(self, "img_end_idx", -1)
        
        if img_s != -1 and img_e != -1 and img_e <= kv_seq_len:
            
            mode = getattr(self, "intervention_mode", "positive")
            
            with torch.no_grad():
                v_mass = attn_probs[:, :, -1, img_s:img_e].sum()
                t_mass = attn_probs[:, :, -1, :].sum() - v_mass
                v_ratio = v_mass / (v_mass + t_mass + 1e-9)

            if v_ratio < self.threshold or mode == "negative":
                object_mask = getattr(self.master_model, 'current_object_mask', None)
                new_probs = attn_probs.clone()
                scale_factor = torch.ones_like(new_probs[:, :, -1, img_s:img_e])

                _mask = None
                if object_mask is not None:
                    _mask = object_mask.view(bsz, 1, -1).to(new_probs.device)

                if mode == "positive":
                    scale_factor = scale_factor + self.alpha * 0.1
                    if _mask is not None and _mask.shape[-1] == (img_e - img_s):
                        if _mask.max() > 0.3:
                            scale_factor = scale_factor + self.alpha * 0.9 * _mask

                elif mode == "negative":
                    if _mask is not None and _mask.shape[-1] == (img_e - img_s):
                        scale_factor = scale_factor * (1.0 - _mask)
                    else:
                        scale_factor = scale_factor * 0.1

                new_probs[:, :, -1, img_s:img_e] = new_probs[:, :, -1, img_s:img_e] * scale_factor
                new_probs[:, :, -1, :] = new_probs[:, :, -1, :] / new_probs[:, :, -1, :].sum(dim=-1, keepdim=True)
                attn_probs = new_probs

    attn_probs = attn_probs.to(query_states.dtype)
    attn_output = torch.matmul(attn_probs, value_states).transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_probs if output_attentions else None, past_key_value


def apply_mask_guided_intervention(model, start_layer=8, end_layer=28, threshold=0.50, alpha=0.50):
    for i in range(start_layer, end_layer):
        layer = model.model.layers[i].self_attn
        layer.use_intervention = True
        layer.threshold = threshold
        layer.alpha = alpha
        layer.master_model = model.model 
        layer.forward = types.MethodType(llama_new_forward, layer)

class MaskGuidedCDProcessor(LogitsProcessor):
    def __init__(self, model, images, image_sizes, penalty_alpha=0.4):
        self.model = model
        self.images = images
        self.image_sizes = image_sizes
        self.penalty_alpha = penalty_alpha
        self.neg_past_key_values = None

    def set_mode(self, mode):
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, 'use_intervention'):
                layer.self_attn.intervention_mode = mode

    def __call__(self, input_ids, scores):
        pos_logits = F.log_softmax(scores, dim=-1)

        self.set_mode("negative")
        with torch.no_grad():
            if self.neg_past_key_values is None:
                out_neg = self.model(
                    input_ids=input_ids,
                    images=self.images,
                    image_sizes=self.image_sizes,
                    use_cache=True
                )
            else:
                out_neg = self.model(
                    input_ids=input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=self.neg_past_key_values
                )
        self.neg_past_key_values = out_neg.past_key_values
        neg_logits = F.log_softmax(out_neg.logits[:, -1, :], dim=-1)

        self.set_mode("positive")

        cutoff = torch.log(torch.tensor(0.1, device=scores.device)) + pos_logits.max(dim=-1, keepdim=True).values
        cd_logits = (1 + self.penalty_alpha) * pos_logits - self.penalty_alpha * neg_logits
        cd_logits = cd_logits.masked_fill(pos_logits < cutoff, -float("inf"))

        return cd_logits
