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

        if self.training and images is not None and inputs_embeds is None:
            safe_input_ids = input_ids.clone()
            safe_input_ids[safe_input_ids < 0] = 0 
            text_embeds_guide = self.get_model().embed_tokens(safe_input_ids)
            
            text_mask = (labels == -100).to(dtype=text_embeds_guide.dtype, device=text_embeds_guide.device)
            
            base_image_features = self.encode_images(images) 
            Z_obj, Z_bg, _ = self.mm_projector_disentangler(base_image_features, text_embeds_guide, text_mask)
            
            original_encode_images = self.encode_images
            
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
            
            batch_size, seq_len, dim = Z_bg.shape
            idx = torch.randperm(seq_len, device=Z_bg.device)
            Z_bg_shuffled = Z_bg[:, idx, :] 
            
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
            del out_intervene 
            
            cos_sim = torch.nn.functional.cosine_similarity(Z_obj, Z_bg, dim=-1, eps=1e-5)
            ortho_loss = torch.mean(torch.abs(cos_sim))
            
            self.encode_images = original_encode_images
            
            total_loss = causal_loss + 0.5 * intervene_loss + 0.1 * ortho_loss
            
            out_1.loss = total_loss
            return out_1

        elif inputs_embeds is None:
            if images is not None and not self.training:
                safe_input_ids = input_ids.clone()
                safe_input_ids[safe_input_ids < 0] = 0
                text_embeds_guide = self.get_model().embed_tokens(safe_input_ids)
                
                original_encode_images = self.encode_images
                base_image_features = self.encode_images(images)
                
                if hasattr(self, 'mm_projector_disentangler'):
                    Z_obj, _, mask_R = self.mm_projector_disentangler(base_image_features, text_embeds_guide)
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
