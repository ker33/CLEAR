#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

# 引入我们在 builder.py 里写好的解耦器
from ..multimodal_projector.builder import TextGuidedDisentangler


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        # ==========================================
        # 创新注入：挂载我们的文本引导解耦器 (增加 mm_projector 前缀以避开 PEFT Bug)
        # ==========================================
        self.mm_projector_disentangler = TextGuidedDisentangler(config.hidden_size)

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # =========================================================================
        # 核心创新：训练阶段的 2 分支平滑因果干预 (取代原有的 3 分支)
        # =========================================================================
        if self.training and images is not None and inputs_embeds is None:
            
            # 🌟 1. 构造防作弊面具：提取出只包含 Prompt 的位置 (LLaVA 规定 User prompt 的 label 是 -100)
            # text_mask = (labels == -100).float()
            
            safe_input_ids = input_ids.clone()
            safe_input_ids[safe_input_ids < 0] = 0 
            text_embeds_guide = self.get_model().embed_tokens(safe_input_ids)
            
            # 🌟 修复: 强制让 text_mask 跟随 text_embeds_guide 的数据类型 (bfloat16) 和设备 (GPU)
            text_mask = (labels == -100).to(dtype=text_embeds_guide.dtype, device=text_embeds_guide.device)
            
            # 🌟 2. 获取解耦特征 (此时解耦器绝对看不到答案)
            base_image_features = self.encode_images(images) 
            Z_obj, Z_bg, _ = self.mm_projector_disentangler(base_image_features, text_embeds_guide, text_mask)
            
            # 备份原始的 encode_images 方法
            original_encode_images = self.encode_images
            
            # ==========================================================
            # 分支 1：正常因果学习 (基于解耦出的纯净物体 Z_obj)
            # ==========================================================
            self.encode_images = lambda _: Z_obj
            inputs_1 = self.prepare_inputs_labels_for_multimodal(
                input_ids.clone(), position_ids, attention_mask.clone() if attention_mask is not None else None, 
                past_key_values, labels.clone() if labels is not None else None, images, image_sizes
            )
            out_1 = super().forward(
                input_ids=inputs_1[0], position_ids=inputs_1[1], attention_mask=inputs_1[2], 
                past_key_values=inputs_1[3], inputs_embeds=inputs_1[4], labels=inputs_1[5], 
                use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict
            )
            causal_loss = out_1.loss
            
            # ==========================================================
            # 🌟 分支 2：图内背景打乱的抗干扰学习 (取代危险的跨图相加)
            # ==========================================================
            # 逻辑：把当前图片的背景特征 Z_bg 在空间维度(序列长度)上打乱，
            # 破坏其具象语义(比如墙壁的形状)，但保留了当前光照和色调特征流形！
            batch_size, seq_len, dim = Z_bg.shape
            idx = torch.randperm(seq_len, device=Z_bg.device)
            Z_bg_shuffled = Z_bg[:, idx, :] 
            
            # 干预特征 = 纯净物体 + 破碎的本地环境光影
            Z_intervene = Z_obj + Z_bg_shuffled
            
            self.encode_images = lambda _: Z_intervene
            inputs_intervene = self.prepare_inputs_labels_for_multimodal(
                input_ids.clone(), position_ids, attention_mask.clone() if attention_mask is not None else None, 
                past_key_values, labels.clone() if labels is not None else None, images, image_sizes
            )
            out_intervene = super().forward(
                input_ids=inputs_intervene[0], position_ids=inputs_intervene[1], attention_mask=inputs_intervene[2], 
                past_key_values=inputs_intervene[3], inputs_embeds=inputs_intervene[4], labels=inputs_intervene[5], 
                use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict
            )
            intervene_loss = out_intervene.loss
            del out_intervene # 节省显存
            
            # ==========================================================
            # 🌟 分支 3：隐空间正交约束 (取代危险的负向 CE Loss)
            # ==========================================================
            # 强制物体特征和背景特征在多维空间中绝对正交 (相关性为 0)
            # cos_sim = torch.nn.functional.cosine_similarity(Z_obj, Z_bg, dim=-1)
            # 🌟 修复：强行加入 eps=1e-5，防止在 fp16 下由于分母过小导致除以 0 产生 NaN 梯度！
            cos_sim = torch.nn.functional.cosine_similarity(Z_obj, Z_bg, dim=-1, eps=1e-5)
            ortho_loss = torch.mean(torch.abs(cos_sim))
            
            # 恢复挂载的原始方法 (极其重要，防越权崩溃)
            self.encode_images = original_encode_images
            
            # ==========================================================
            # 最终 Loss 大一统聚合
            # ==========================================================
            # 基础 Loss 保底 + 0.5倍干预压迫抗干扰 + 0.1倍特征分离强制力
            total_loss = causal_loss + 0.5 * intervene_loss + 0.1 * ortho_loss
            
            # 将总 Loss 塞回 out_1 中返回，欺骗 Trainer 让它进行统一步骤的反向传播
            out_1.loss = total_loss
            return out_1

        # =========================================================================
        # 修改 llava_llama.py 中的推理阶段前向传播 (Inference / Generation)
        # =========================================================================
        elif inputs_embeds is None:
            if images is not None and not self.training:
                safe_input_ids = input_ids.clone()
                safe_input_ids[safe_input_ids < 0] = 0
                text_embeds_guide = self.get_model().embed_tokens(safe_input_ids)
                
                original_encode_images = self.encode_images
                base_image_features = self.encode_images(images)
                
                if hasattr(self, 'mm_projector_disentangler'):
                    # 🌟 改进点：接收 mask_R
                    Z_obj, _, mask_R = self.mm_projector_disentangler(base_image_features, text_embeds_guide)
                    # 🌟 创新注入：将物体掩码挂载到模型上，形状从[B, 576, 1] 转换为 [B, 1, 1, 576] 以对齐注意力矩阵
                    self.get_model().current_object_mask = mask_R.transpose(1, 2).unsqueeze(1).detach()
                else:
                    Z_obj = base_image_features
                    
                self.encode_images = lambda _: Z_obj
                
                (
                    input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes
                )
                self.encode_images = original_encode_images
            else:
                (
                    input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes
                )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
