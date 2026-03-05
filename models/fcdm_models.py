import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

#################################################################################
#                                 Core DiU Model                                #
#################################################################################

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block with adaptive LayerNorm-Zero (adaLN-Zero) conditioning.
    Combines depthwise conv and pointwise MLP (as in ConvNeXt) with adaLN modulation on the channel dimension.
    """
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()        
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.norm = LayerNorm2d(dim, affine=False, eps=1e-6) #nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, int(dim * mlp_ratio), 1) #nn.Linear(dim, int(dim * mlp_ratio))
        self.act = nn.GELU()
        self.grn = GRN(int(dim * mlp_ratio))
        self.pwconv2 = nn.Conv2d(int(dim * mlp_ratio), dim, 1) #nn.Linear(int(dim * mlp_ratio), dim)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 3 * dim, bias=True)
        )

    def forward(self, x, c):
        """
        x: tensor of shape (B, C, H, W)
        c: conditioning tensor of shape (B, C)
        """
        # Depthwise conv
        h = self.dwconv(x)  # (B, C, H, W)
        # Compute adaLN parameters
        shift, scale, gate = self.adaLN_modulation(c).unsqueeze(2).unsqueeze(3).chunk(3, dim=1)
        # Apply adaptive LayerNorm-Zero: norm -> scale & shift
        h = self.norm(h)
        h = torch.addcmul(shift, h, scale + 1) #h * (scale[:, None, None, :] + 1) + shift[:, None, None, :]
        # Pointwise MLP
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.grn(h)
        h = self.pwconv2(h)
        # Apply gate
        h = h * gate
        # Residual
        return x + h

class ConvFinalLayer(nn.Module):
    """
    Conv-style final layer
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))#nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # self.up   = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))#nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        # x = self.up(x)
        return self.conv(x)


class FCDM(nn.Module):
    """
    Fully Convolutional Diffusion Models.
    """
    def __init__(
        self,
        in_channels=4,
        hidden_size=1152,
        depth=[2,5,8,5,2],
        mlp_ratio=3,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        **kwargs
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        
        self.t_embedder_1 = TimestepEmbedder(hidden_size)
        self.y_embedder_1 = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.t_embedder_2 = TimestepEmbedder(hidden_size*2)
        self.y_embedder_2 = LabelEmbedder(num_classes, hidden_size*2, class_dropout_prob)

        self.t_embedder_3 = TimestepEmbedder(hidden_size*4)
        self.y_embedder_3 = LabelEmbedder(num_classes, hidden_size*4, class_dropout_prob)

        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1) #nn.Conv2d(in_channels, hidden_size, kernel_size=1)

        # encoder-1
        self.encoder_level_1 = nn.ModuleList()
        for _ in range(depth[0]):
            self.encoder_level_1.append(ConvNeXtBlock(hidden_size, mlp_ratio=mlp_ratio))
        self.down1_2 = Downsample(hidden_size, hidden_size*2)

        # encoder-2
        self.encoder_level_2 = nn.ModuleList()
        for _ in range(depth[1]):
            self.encoder_level_2.append(ConvNeXtBlock(hidden_size*2, mlp_ratio=mlp_ratio))
        self.down2_3 = Downsample(hidden_size*2, hidden_size*4)

        # latent
        self.latent = nn.ModuleList()
        for _ in range(depth[2]):
            self.latent.append(ConvNeXtBlock(hidden_size * 4, mlp_ratio=mlp_ratio))

        # decoder-2
        self.up3_2 = Upsample(hidden_size*4, hidden_size*2)
        self.reduce_chans_2 = nn.Conv2d(hidden_size*4, hidden_size*2, kernel_size=1)
        self.decoder_level_2 = nn.ModuleList()
        for _ in range(depth[3]):
            self.decoder_level_2.append(ConvNeXtBlock(hidden_size*2, mlp_ratio=mlp_ratio))

        # decoder-1
        self.up2_1 = Upsample(hidden_size*2, hidden_size)
        self.reduce_chans_1 = nn.Conv2d(hidden_size*2, hidden_size, kernel_size=1)
        self.decoder_level_1 = nn.ModuleList([
            ConvNeXtBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(depth[4])
        ])

        self.output_layer = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
        self.final_layer = ConvFinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder_1.embedding_table.weight, std=0.02)
        nn.init.normal_(self.y_embedder_2.embedding_table.weight, std=0.02)
        nn.init.normal_(self.y_embedder_3.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder_1.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_1.mlp[2].weight, std=0.02)

        nn.init.normal_(self.t_embedder_2.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_2.mlp[2].weight, std=0.02)

        nn.init.normal_(self.t_embedder_3.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_3.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in blocks:
        blocks = self.encoder_level_1 + self.encoder_level_2 + self.latent + self.decoder_level_2 + self.decoder_level_1
        for block in blocks:
            if hasattr(block, 'adaLN_modulation'):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward

    def forward(self, x, t, y):
        """
        Forward pass of U-DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x_emb = self.x_embedder(x)
        t1 = self.t_embedder_1(t)
        y1 = self.y_embedder_1(y, self.training)
        c1 = t1 + y1

        t2 = self.t_embedder_2(t)
        y2 = self.y_embedder_2(y, self.training)
        c2 = t2 + y2

        t3 = self.t_embedder_3(t)
        y3 = self.y_embedder_3(y, self.training)
        c3 = t3 + y3

        # encoder_1
        out_enc_level1 = x_emb
        for block in self.encoder_level_1:
            out_enc_level1 = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), out_enc_level1, c1, use_reentrant=False)
        inp_enc_level2 = self.down1_2(out_enc_level1)

        # encoder_2
        out_enc_level2 = inp_enc_level2
        for block in self.encoder_level_2:
            out_enc_level2 = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), out_enc_level2, c2, use_reentrant=False)
        inp_enc_level3 = self.down2_3(out_enc_level2)

        # latent
        latent = inp_enc_level3
        for block in self.latent:
            latent = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), latent, c3, use_reentrant=False)

        # decoder_2
        inp_dec_level2 = self.up3_2(latent)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        out_dec_level2 = self.reduce_chans_2(inp_dec_level2)
        for block in self.decoder_level_2:
            out_dec_level2 = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), out_dec_level2, c2, use_reentrant=False)

        # decoder_1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.reduce_chans_1(inp_dec_level1)
        for block in self.decoder_level_1:
            out_dec_level1 = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), out_dec_level1, c1, use_reentrant=False)
        
        # output
        x = self.output_layer(out_dec_level1)
        x = self.final_layer(x, c1)

        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#             FCDM (Fully Convolutional Diffusion Models) Configs               #
#################################################################################

def FCDM_S(**kwargs):
    return FCDM(hidden_size=128, depth=[2,4,8,4,2], **kwargs)

def FCDM_B(**kwargs):
    return FCDM(hidden_size=256, depth=[2,4,8,4,2], **kwargs)

def FCDM_L(**kwargs):
    return FCDM(hidden_size=512, depth=[2,4,8,4,2], **kwargs)

def FCDM_XL(**kwargs):
    return FCDM(hidden_size=512, depth=[3,6,12,6,3], **kwargs)

FCDM_models = {
    'FCDM-S': FCDM_S,
    'FCDM-B': FCDM_B,
    'FCDM-L': FCDM_L,
    'FCDM-XL': FCDM_XL
}