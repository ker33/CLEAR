import torch
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessor

class MaskGuidedCDProcessor(LogitsProcessor):
    def __init__(self, model, input_ids, images, image_sizes, penalty_alpha=0.3):
        """
        penalty_alpha: 对比解码的惩罚力度（类似 OWL 的 lambda），建议 0.2 ~ 0.5
        """
        self.model = model
        self.images = images
        self.image_sizes = image_sizes
        self.penalty_alpha = penalty_alpha
        
        # 缓存第一条路径（正向路径）的 past_key_values，加速推理
        self.past_key_values_pos = None
        self.past_key_values_neg = None

    def __call__(self, input_ids, scores):
        # scores 是模型正常前向传播（作为正向路径）吐出来的 Logits
        # 这里为了保持纯洁性，我们假设 model.generate() 跑的就是正向路径
        
        # 1. 获取正向路径的概率分布 (已经通过 Attention 增强了目标物体)
        pos_logits = F.log_softmax(scores, dim=-1)
        
        # 2. 跑第二条路径（反向路径 / 幻觉路径）
        # 此时我们需要通知底层的 Attention 层：“现在是反向路径，给我抑制物体，只看背景和文本！”
        self.set_attention_mode(mode="negative")
        
        with torch.no_grad():
            # 取出最后一个 token 送进去算反向路径
            out_neg = self.model(
                input_ids=input_ids[:, -1:],
                images=self.images if self.past_key_values_neg is None else None, # 只有第一步需要图像
                use_cache=True,
                past_key_values=self.past_key_values_neg,
            )
            self.past_key_values_neg = out_neg.past_key_values
            neg_logits = F.log_softmax(out_neg.logits[:, -1, :], dim=-1)
            
        # 恢复正向模式，以备生成下一个词
        self.set_attention_mode(mode="positive")

        # 3. 对比解码核心公式 (Logits 相减)
        # 动态截断：防止惩罚过度导致选出乱码词汇（OWL 里的 Adaptive Plausibility Constraint）
        cutoff = torch.log(torch.tensor(0.1, device=scores.device)) + pos_logits.max(dim=-1, keepdim=True).values
        
        cd_logits = (1 + self.penalty_alpha) * pos_logits - self.penalty_alpha * neg_logits
        
        # 把原本概率就极低的词屏蔽掉，防止减法算出奇怪的低频词
        cd_logits = cd_logits.masked_fill(pos_logits < cutoff, -float("inf"))

        return cd_logits

    def set_attention_mode(self, mode="positive"):
        """通知所有注意力层当前的模式"""
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, 'use_intervention'):
                layer.self_attn.intervention_mode = mode