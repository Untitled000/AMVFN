
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops



class ChannelAttentionModule(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        return self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3) * x
    

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y
    

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttentionModule(channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x
    

class simAM(nn.Module):
    def __init__(self, l=1e-5):
        super().__init__()
        self.l = l
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        # Calculate the mean of the input tensor in the channel dimension
        mean = torch.mean(x, dim=[-2, -1], keepdim=True)
        # Calculate the variance of the input tensor in the channel dimension
        std = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        e_t = torch.pow((x - mean), 2) / (4 * (std + self.l)) + 0.5
        out = x * self.sigmoid(e_t)

        return out


class AttentionMultiViewFusionNet(nn.Module):
    def __init__(self, arch, num_classes, n, pretrained_weights=False):
        
        super().__init__()

        self.n = n
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights

        self.sa = SpatialAttentionModule()
        self.ca = ChannelAttentionModule(3)
        self.cbam = CBAMBlock(3)
        self.simAM = simAM()

        drop_rate = .0 if 'tiny' in arch else .1
        state_dict = torch.load('/home/wcy/Smoke_dt/multi-view-hybrid-main/vit_tiny_r_s16_p8_224.bin')
        # state_dict = torch.load('/home/wcy/Smoke_dt/multi-view-hybrid-main/vit_small_r26_s32_224.bin')
        self.model = timm.create_model(arch, pretrained=self.pretrained_weights, drop_rate=drop_rate)
        self.model.load_state_dict(state_dict)
        self.model.reset_classifier(self.num_classes)
        for block in self.model.blocks:
            block.attn.fused_attn = False
        
        self.embed_dim = self.model.embed_dim
        # self.target_layer = self.model.blocks[-1].norm1

        for block in self.model.blocks:
            block.attn.proj_drop = nn.Dropout(p=0.0)

        self.img_embed_matrix = nn.Parameter(torch.zeros(1, n, self.embed_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.img_embed_matrix)

        nn.init.zeros_(self.model.head.weight)
        nn.init.zeros_(self.model.head.bias)

    def format_multi_image_tokens(self, x, batch_size, tokens_per_image):

        x = einops.rearrange(x, '(b n) s c -> b (n s) c', b=batch_size, n=self.n)
        first_img_token_idx = 0
        if self.model.cls_token is not None:
            # Need to remove all excess CLS tokens
            for i in range(1, self.n):
                excess_cls_index = i * tokens_per_image + 1
                x = torch.cat((x[:, :excess_cls_index], x[:, excess_cls_index + 1:]), dim=1)
            first_img_token_idx = 1

        image_embeddings = F.normalize(self.img_embed_matrix, dim=-1)
        x[:, first_img_token_idx:] += torch.repeat_interleave(image_embeddings, tokens_per_image, dim=1)
        return x

    def forward(self, x):
        batch_size = len(x)
        output_dict = {'single': {}}
        if self.n > 1:
            output_dict['mv_collection'] = {}
            # Take VI and IR images separately
            x_vi = x[:, 0, :, :, :]
            x_ir = x[:, 1, :, :, :]
            
            '''Attention Module'''
            # sa & ca
            # x_vi = self.sa(x_vi).unsqueeze(1)
            # x_ir = self.ca(x_ir).unsqueeze(1)
            # CBAM
            # x_vi = self.cbam(x_vi).unsqueeze(1)
            # x_ir = self.cbam(x_ir).unsqueeze(1)
            # simAM
            # x_vi = self.simAM(x_vi).unsqueeze(1)
            # x_ir = self.simAM(x_ir).unsqueeze(1)
            # None
            x_vi = x_vi.unsqueeze(1)
            x_ir = x_ir.unsqueeze(1)

            x = torch.cat([x_vi, x_ir], dim=1)
        
        x = einops.rearrange(x, 'b n c h w -> (b n) c h w')
        x = self.model.patch_embed(x)

        tokens_per_image = x.shape[1]
        x = self.model._pos_embed(x)

        # Corresponding to single image input and multiple image fusion input respectively
        for view_type in output_dict:
            tokens = x.clone()
            # If there is more than one input image, fusion processing is performed
            if view_type == 'mv_collection':
                tokens = self.format_multi_image_tokens(tokens, batch_size, tokens_per_image)
            tokens = self.model.blocks(tokens)
            tokens = self.model.norm(tokens)
            # Output a dictionary containing the results of individual images and the results of the fused images
            output_dict[view_type]['logits'] = self.model.forward_head(tokens)

        return output_dict