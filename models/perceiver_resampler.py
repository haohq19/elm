import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()


# FFN
def FeedForward(d_model, mult = 4):
    in_dim = int(d_model * mult)
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, in_dim, bias = False),
        nn.GELU(),
        nn.Linear(in_dim, d_model, bias = False)
    )


# Cross attention
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        d_model,
        d_head = 64,
        nheads = 8,
    ):
        super().__init__()
        self.scale = d_head ** -0.5
        self.nheads = nheads
        in_dim = d_head * nheads

        self.norm_medias = nn.LayerNorm(d_model)
        self.norm_latents = nn.LayerNorm(d_model)

        self.to_q = nn.Linear(d_model, in_dim, bias = False)
        self.to_kv = nn.Linear(d_model, in_dim * 2, bias = False)
        self.to_out = nn.Linear(in_dim, d_model, bias = False)

    def forward(self, x, latents):
        # x.shape = [batch_size, nsteps, d_model]
        # latents.shape = [batch_size, num_latents, d_model]

        x = self.norm_medias(x)
        latents = self.norm_latents(latents)

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)  # kv_input.shape = [batch_size, nsteps + num_latents, d_model]
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q = q.reshape(q.shape[0], q.shape[1], self.nheads, -1).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.nheads, -1).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.nheads, -1).permute(0, 2, 1, 3)

        q = q / self.scale

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)   # sim.shape = [batch_size, nheads, num_latents, nsteps + num_latents]
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()  
        atten = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', atten, v)   # out.shape = [batch_size, nheads, num_latents, d_head]
        out = out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], -1)  # out.shape = [batch_size, num_latents, d_model]
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        d_model = 4096,
        num_layers = 2,
        d_head = 256,
        nheads = 16,
        num_latents = 4,
        ff_mult = 2
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_model))
        self.pos_encoding = PositionalEncoding(d_model)

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(d_model = d_model, d_head = d_head, nheads = nheads),
                FeedForward(d_model = d_model, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):

        x = self.pos_encoding(x)  # x.shape = [batch_size, nsteps, d_model]
        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])  # latents.shape = [batch_size, num_latents, d_model]

        for atten, ffn in self.layers:
            latents = atten(x, latents) + latents
            latents = ffn(latents) + latents

        return self.norm(latents)


if __name__ == '__main__':
    model = PerceiverResampler()
    x = torch.randn(2, 64, 512)
    out = model(x)
    print(out.shape)