import torch
import torch.nn as nn
import torch.nn.functional as F
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

class TextGuidedDisentangler(nn.Module):
    def __init__(self, hidden_size=4096):
        super().__init__()
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.vis_proj = nn.Linear(hidden_size, hidden_size)
        
        self.score_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(), 
            nn.Linear(hidden_size // 2, 1) 
        )
        
        self.norm_obj = nn.LayerNorm(hidden_size)
        self.norm_bg = nn.LayerNorm(hidden_size)

    def forward(self, visual_feats, text_embeds, text_mask=None):
        if text_mask is None:
            text_mask = torch.ones((text_embeds.shape[0], text_embeds.shape[1]), 
                                   device=text_embeds.device, dtype=text_embeds.dtype)
            
        text_mask_expanded = text_mask.unsqueeze(-1)
        masked_text_embeds = text_embeds * text_mask_expanded
        
        valid_lengths = text_mask.sum(dim=1, keepdim=True).clamp(min=1)
        text_pooled = masked_text_embeds.sum(dim=1) / valid_lengths
        text_pooled = text_pooled.unsqueeze(1)
        
        t_feat = self.text_proj(text_pooled)
        v_feat = self.vis_proj(visual_feats)
        fusion_feat = v_feat * t_feat
        
        M = self.score_net(fusion_feat)
        mask_R = torch.sigmoid(M)
        mask_I = 1.0 - mask_R
        
        Z_obj = self.norm_obj(mask_R * visual_feats) 
        Z_bg  = self.norm_bg(mask_I * visual_feats)
        
        return Z_obj, Z_bg, mask_R
