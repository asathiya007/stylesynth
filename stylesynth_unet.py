from einops.layers.torch import Rearrange
import math
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    '''
    Convolution block, for use in StyleSynth U-net
    '''
    def __init__(self, in_chs, out_chs, group_size, num_hidden_layers=1,
                 pool=True):
        super().__init__()

        # define layers
        init_layer = [
            nn.Conv2d(in_chs, out_chs, 3, 1, 1),
            nn.GroupNorm(group_size, out_chs),
            nn.GELU()
        ]
        hidden_layers = [
            nn.Conv2d(out_chs, out_chs, 3, 1, 1),
            nn.GroupNorm(group_size, out_chs),
            nn.GELU()
        ] * (num_hidden_layers)
        if pool:
            rearrange_layers = [
                # cuts image into p1 groups of horizontal strips (adjacent
                # strips are in different groups), then cuts each group of
                # horizontal strips into p2 groups of vertical strips (adjacent
                # strips are in different groups), then stacks the groups of
                # vertical strips (pertaining to the same group of horizontal
                # strips) along the channels dimension, then stacks the groups
                # of horizontal strips along the channels dimension
                Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
                # convolution to produce feature map with desired number of
                # output dimensions
                nn.Conv2d(4 * out_chs, out_chs, 3, 1, 1)
            ]
        else:
            rearrange_layers = []
        self.layers = nn.Sequential(
            *(init_layer + hidden_layers + rearrange_layers))

    def forward(self, x):
        return self.layers(x)


class TransposeConvBlock(nn.Module):
    '''
    Transpose convolution block, for use in StyleSynth U-net
    '''
    def __init__(self, in_chs, out_chs, group_size, num_hidden_layers=1):
        super().__init__()

        # define layers
        transp_conv_layer = [
            nn.ConvTranspose2d(in_chs, out_chs, 2, 2),
            nn.GroupNorm(group_size, out_chs),
            nn.GELU()
        ]
        hidden_layers = [
            nn.Conv2d(out_chs, out_chs, 3, 1, 1),
            nn.GroupNorm(group_size, out_chs),
            nn.GELU()
        ] * (num_hidden_layers + 1)
        self.layers = nn.Sequential(*(transp_conv_layer + hidden_layers))

    def forward(self, x_w_skip):
        return self.layers(x_w_skip)


class SinPosEmbedBlock(nn.Module):
    '''
    Sinusoidal positional embedding block, for use in StyleSynth U-net
    '''
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        # by using log, we avoid overflow errors caused by raising 10000 to a
        # large power
        half_embed_dim = self.embed_dim // 2
        embeddings = t[:, None] * torch.exp(
            torch.arange(half_embed_dim, device=t.device)
            * (-1 * math.log(10000) / (half_embed_dim - 1)))[None, :]

        # interleave sin and cos values by pairing each sin value with the
        # corresponding cos value (creating 2-element lists in the index-2
        # dimension), and then reshaping the output to join the 2-element lists
        # together across the index-1 dimension to form embeddings of the
        # desired length (one embedding for each time step provided in t)
        embeddings = torch.stack(
            [embeddings.sin(), embeddings.cos()], dim=-1).reshape(
                t.shape[0], self.embed_dim)

        return embeddings


class EmbedBlock(nn.Module):
    '''
    General embedding block, for use in StyleSynth U-net
    '''
    def __init__(self, in_dim, embed_dim, num_hidden_layers=1):
        super().__init__()

        # define layers
        init_layer = [
            nn.Linear(in_dim, embed_dim),
            nn.GELU()
        ]
        hidden_layers = [
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        ] * num_hidden_layers
        out_layer = [
            nn.Unflatten(1, (embed_dim, 1, 1))
        ]
        self.layers = nn.Sequential(*(init_layer + hidden_layers + out_layer))

    def forward(self, x):
        return self.layers(x)


class StyleSynth_UNet(nn.Module):
    '''
    U-net for predicting noise added to an image at a particular time step.
    Used in the diffusion model
    '''
    def __init__(self, img_size, img_chs, T, down_chs, group_size,
                 t_embed_dim=10, c_embed_dim=10, conv_hidden_layers=2,
                 dense_embed_hidden_layers=2, t_embed_hidden_layers=2,
                 c_embed_hidden_layers=2, transp_conv_hidden_layers=2,
                 device='cpu'):
        super().__init__()
        self.device = device
        self.T = T
        self.img_size = img_size
        self.img_chs = img_chs
        up_chs = list(reversed(down_chs))

        # convolution blocks
        self.init_conv_block = ConvBlock(
            img_chs, down_chs[0], group_size, conv_hidden_layers, pool=False)\
            .to(self.device)
        self.conv_blocks = nn.ModuleList(list(map(
            lambda i: ConvBlock(
                down_chs[i], down_chs[i + 1], group_size, conv_hidden_layers)
            .to(self.device),
            range(0, len(down_chs) - 1))))

        # for reshaping into latent vector
        latent_image_size = img_size // (2 ** len(self.conv_blocks))
        latent_vec_len = (latent_image_size ** 2) * down_chs[-1]
        self.to_latent_vec = nn.Flatten().to(self.device)

        # dense embedding for latent vector
        dense_embed_layers = [
            nn.Linear(latent_vec_len, latent_vec_len),
            nn.GELU()
        ] * (dense_embed_hidden_layers + 2)
        self.dense_embed_latent_vec = nn.Sequential(*dense_embed_layers).to(
            self.device)

        # for reshaping into input feature map for transpose convolution blocks
        self.to_feature_map = nn.Unflatten(
            1, (up_chs[0], latent_image_size, latent_image_size)).to(
                self.device)

        # transpose convolution blocks
        self.transp_conv_blocks = nn.ModuleList(list(map(
            lambda i: TransposeConvBlock(
                2 * up_chs[i], up_chs[i + 1], group_size,
                transp_conv_hidden_layers).to(self.device),
            range(len(up_chs) - 1))))

        # time embedding blocks
        self.sin_pos_embed_block = SinPosEmbedBlock(t_embed_dim).to(
            self.device)
        self.time_embed_blocks = nn.ModuleList(list(map(
            lambda i: EmbedBlock(
                t_embed_dim, up_chs[i + 1], t_embed_hidden_layers).to(
                    self.device),
            range(len(up_chs) - 1))))

        # context embedding blocks
        self.context_embed_blocks = nn.ModuleList(list(map(
            lambda i: EmbedBlock(
                c_embed_dim, up_chs[i + 1], c_embed_hidden_layers).to(
                    self.device),
            range(len(up_chs) - 1))))

        # final convolution block for matching input dimensions
        self.output_conv_block = nn.Sequential(
            ConvBlock(2 * up_chs[-1], up_chs[-1], group_size,
                      conv_hidden_layers, pool=False),
            nn.Conv2d(up_chs[-1], img_chs, 3, 1, 1)).to(self.device)

    def forward(self, x_t, t, c):
        # initial convolution
        init_conv_output = self.init_conv_block(x_t)

        # downsampling
        conv_outputs = []
        for i in range(len(self.conv_blocks)):
            if i == 0:
                conv_input = init_conv_output
            else:
                conv_input = conv_outputs[-1]
            conv_outputs.append(self.conv_blocks[i](conv_input))

        # conversion to latent vector, dense embedding
        latent_vec = self.to_latent_vec(conv_outputs[-1])
        latent_vec_embed = self.dense_embed_latent_vec(latent_vec)

        # time embeddings
        scaled_t = t / self.T  # scales time value from 0 to 1
        sin_pos_embed = self.sin_pos_embed_block(scaled_t)
        time_embeds = list(map(
            lambda teb: teb(sin_pos_embed), self.time_embed_blocks))

        # context embeddings
        context_embeds = list(map(
            lambda ceb: ceb(c), self.context_embed_blocks))

        # reshape into feature map, before upsampling
        feature_map = self.to_feature_map(latent_vec_embed)

        # upsampling
        transp_conv_outputs = []
        for i in range(len(self.transp_conv_blocks)):
            if i == 0:
                transp_conv_input = feature_map
            else:
                transp_conv_input = (
                    context_embeds[i - 1] * transp_conv_outputs[-1]
                    + time_embeds[i - 1])
            transp_conv_outputs.append(self.transp_conv_blocks[i](
                torch.concat([transp_conv_input, conv_outputs[-1 - i]], dim=1))
                )

        # output convolution to match input dimensions
        output = self.output_conv_block(torch.concat([
            context_embeds[-1] * transp_conv_outputs[-1] + time_embeds[-1],
            init_conv_output], axis=1))

        return output
