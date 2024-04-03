import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0., d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x, temb):
        _, _, H, W = x.shape
        x = F.interpolate(
            x, scale_factor=2, mode='nearest')
        x = self.main(x)
        return x


class AttnBlock(nn.Module):  # 自注意力模块
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)  # 对输入特征图进行归一化
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)  # 查询
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)  # 键
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)  # 值
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    # 使用GroupNorm对输入特征图进行归一化。
    # 然后，我们使用三个卷积层（proj_q、proj_k和proj_v）将输入特征图映射到查询、键和值空间。
    # 接着，我们计算查询和键之间的相似度，并使用softmax函数将其转换为注意力权重。
    # 最后，我们使用值和注意力权重计算加权平均值，并使用一个卷积层（proj）将其映射回原始特征图的形状。
    # 最终，我们将原始特征图和注意力输出相加，得到最终输出特征图。
    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class CrossAttention(nn.Module):
    def __init__(self, in_channels,key_channels, value_channels , tdim ):
        super(CrossAttention, self).__init__()

        # Query from image features
        self.query_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)

        # Key and value from flow features
        # self.key_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(key_channels, in_channels, kernel_size=1)

        self.value_conv = nn.Conv2d(value_channels, in_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, in_channels),
        )

    def forward(self, x, key, value, temb):
        batch_size, _, height, width = x.size()
        # Query from image features
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        # print('proj_query', proj_query.size())
        # print('key', key.size())
        # Key and value from flow features
        proj_key = self.key_conv(key).view(batch_size, -1, width * height)
        # print('proj_key', proj_key.size())

        # Attention
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        proj_value = self.value_conv(value).view(batch_size, -1, width * height)

        # Output
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)

        # Fusion
        out = self.gamma * out + x
        out += self.temb_proj(temb)[:, :, None, None]

        return out


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x, temb):

        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class UNet(nn.Module):
    def __init__(self, T, ch, ch_mult, attn, num_res_blocks, dropout):
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.flowhead = nn.Conv2d(4, ch, kernel_size=3, stride=1, padding=1)

        self.downblocks = nn.ModuleList()
        # self.flowencoder = flowencoder()

        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult

            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                    self.downblocks.append(CrossAttention(now_ch, now_ch, now_ch, tdim=tdim,))
                    self.downblocks.append(DownSample(now_ch))
                    chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):

            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch

            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        assert len(chs) == 0


        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.xavier_uniform_(self.flowhead.weight)
        init.zeros_(self.head.bias)
        init.zeros_(self.flowhead.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, t, flow):
        # Timestep embedding
        temb = self.time_embedding(t)
        f = self.flowhead(flow)
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            if isinstance(layer, CrossAttention):
                h = layer(h, f, f, temb)
                hs[-1] = h
            else:
                h = layer(h, temb)
                f = layer(f, temb)
                hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)
        assert len(hs) == 0  # 用于确保在上采样过程中，下采样层的列表已经被完全清空。
        return h


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 3, 4], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size,))
    f = torch.rand([batch_size, 4, 32, 32])
    y = model(x, t, f)
    print('y', y.shape)


