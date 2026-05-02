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
        # 核心创新：训练阶段的 3 分支因果干预 (Training Phase Intervention)
        # =========================================================================
        if self.training and images is not None and inputs_embeds is None:
            
            # 1. 获取文本语义引导 (Text Guidance)
            safe_input_ids = input_ids.clone()
            safe_input_ids[safe_input_ids < 0] = 0 
            text_embeds_guide = self.get_model().embed_tokens(safe_input_ids)
            
            # 2. 提取原始视觉特征 (Base Visual Features: 576 x 4096)
            base_image_features = self.encode_images(images) 
            
            # 3. 文本引导解耦！分离出物体本质(Z_obj)和虚假背景(Z_bg)
            Z_obj, Z_bg = self.mm_projector_disentangler(base_image_features, text_embeds_guide)
            
            # 4. 跨图抓取背景构造反事实 (Cross-Image Background Intervention)
            batch_size = Z_obj.shape[0]
            if batch_size > 1:
                perm = torch.randperm(batch_size)
                Z_bg_other = Z_bg[perm] 
            else:
                Z_bg_other = Z_bg
                
            visual_branch_1 = Z_obj                      
            visual_branch_2 = Z_obj + Z_bg_other         
            visual_branch_3 = Z_bg                       
            
            # 5. 使用 Python 动态拦截技巧 (Mocking) 算 3 个分支的 Loss
            original_encode_images = self.encode_images
            
            # --- 分支 1 传播 ---
            self.encode_images = lambda _: visual_branch_1
            inputs_1 = self.prepare_inputs_labels_for_multimodal(
                input_ids.clone(), position_ids, attention_mask.clone() if attention_mask is not None else None, 
                past_key_values, labels.clone() if labels is not None else None, images, image_sizes
            )
            out_1 = super().forward(
                input_ids=inputs_1[0], position_ids=inputs_1[1], attention_mask=inputs_1[2], 
                past_key_values=inputs_1[3], inputs_embeds=inputs_1[4], labels=inputs_1[5], 
                use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict
            )
            loss_1 = out_1.loss
            
            # --- 分支 2 传播 ---
            self.encode_images = lambda _: visual_branch_2
            inputs_2 = self.prepare_inputs_labels_for_multimodal(
                input_ids.clone(), position_ids, attention_mask.clone() if attention_mask is not None else None, 
                past_key_values, labels.clone() if labels is not None else None, images, image_sizes
            )
            out_2 = super().forward(
                input_ids=inputs_2[0], position_ids=inputs_2[1], attention_mask=inputs_2[2], 
                past_key_values=inputs_2[3], inputs_embeds=inputs_2[4], labels=inputs_2[5], 
                use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict
            )
            loss_2 = out_2.loss
            del out_2  # 【救命魔法】：立刻释放分支2巨大 logits 占用的 500MB 显存！
            
            # --- 分支 3 传播 ---
            self.encode_images = lambda _: visual_branch_3
            inputs_3 = self.prepare_inputs_labels_for_multimodal(
                input_ids.clone(), position_ids, attention_mask.clone() if attention_mask is not None else None, 
                past_key_values, labels.clone() if labels is not None else None, images, image_sizes
            )
            out_3 = super().forward(
                input_ids=inputs_3[0], position_ids=inputs_3[1], attention_mask=inputs_3[2], 
                past_key_values=inputs_3[3], inputs_embeds=inputs_3[4], labels=inputs_3[5], 
                use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict
            )
            loss_3 = out_3.loss
            del out_3  # 【救命魔法】：立刻释放分支3的 500MB 显存！
            
            self.encode_images = original_encode_images
            
            # --- 6. 聚合 Loss ---
            # 删除了导致 FP16 崩溃的负惩罚，回归最稳健的相加！
            # 取平均！维持梯度原本的量级，防止 FP16 溢出
            # total_loss = (out_1.loss + out_2.loss) / 2.0
            
            # 策略：前两个分支要求算得准，第三个分支要求它“不知道”，因此给第三个分支施加 -0.1 倍的惩罚。
            # clamp(max=5.0) 是即使在 BF16 下，依然保留的稳健性保护机制，防止惩罚力度把正确答案推得太远。
            penalty_loss = -0.1 * torch.clamp(loss_3, max=5.0)
            
            # 除以 2.0 是为了维持主梯度量级稳定
            total_loss = (loss_1 + loss_2) / 2.0 + penalty_loss
            
            out_1.loss = total_loss
            return out_1

        # =========================================================================
        # 推理阶段的正常前向传播 (Inference / Generation)
        # =========================================================================
        elif inputs_embeds is None:
            if images is not None and not self.training:
                safe_input_ids = input_ids.clone()
                safe_input_ids[safe_input_ids < 0] = 0
                text_embeds_guide = self.get_model().embed_tokens(safe_input_ids)
                
                original_encode_images = self.encode_images
                base_image_features = self.encode_images(images)
                
                if hasattr(self, 'mm_projector_disentangler'):
                    Z_obj, _ = self.mm_projector_disentangler(base_image_features, text_embeds_guide)
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
