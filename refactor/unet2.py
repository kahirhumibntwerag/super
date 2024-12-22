import torch
import torch.nn as nn
import math

class Swish(nn.Module):
  def forward(self, x):
    return x*torch.sigmoid(x)


class TimeEmbedding(nn.Module):
  def __init__(self, emb_dim ):
    super().__init__()
    self.emb_dim = emb_dim
    self.fc1 = nn.Linear(self.emb_dim//4, self.emb_dim)
    self.act1 = Swish()
    self.fc2 = nn.Linear(self.emb_dim, self.emb_dim)

  def forward(self, t):
    half_dim = self.emb_dim//8
    emb = math.log(10_000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
    emb = t[:, None]*emb[None,:]
    emb = torch.cat((emb.sin(), emb.cos()), dim=1)

    emb = self.fc1(emb)
    emb = self.act1(emb)
    emb = self.fc2(emb)
    return emb

class ResidualBlockG(nn.Module):
  def __init__(self, in_channels, out_channels, emb_dim, num_groups=1, dropout=0.1):
    super().__init__()


    self.norm1 = nn.GroupNorm(num_groups, in_channels)
    self.act1 = Swish()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))


    self.norm2 = nn.GroupNorm(num_groups, out_channels)
    self.act2 = Swish()
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1))

    self.act_time = Swish()
    self.time_emb = nn.Linear(emb_dim, out_channels)



    if in_channels != out_channels:
      self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
    else:
      self.shortcut = nn.Identity()


    self.dropout = nn.Dropout(dropout)

  def forward(self, x, t):

    h = self.conv1(self.act1(self.norm1(x)))

    h += self.time_emb(self.act_time(t))[:,:, None, None]

    h = self.conv2(self.dropout(self.act2(self.norm2(h))))

    return h + self.shortcut(x)


class AttentionBlock(nn.Module):
  def __init__(self, n_channels, n_heads=1, d_k=None, num_groups=32):
    super().__init__()
    if d_k is None:
      d_k = n_channels
    self.norm = nn.GroupNorm(num_groups, n_channels)
    self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
    self.output = nn.Linear(n_heads * d_k, n_channels)
    self.scale = d_k ** -0.5

    self.n_heads = n_heads
    self.d_k = d_k

  def forward(self, x):
    batch_size, n_channels, height, width = x.shape
    x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
    qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
    q, k, v = torch.chunk(qkv, 3, dim=-1)

    attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
    attn = attn.softmax(dim=2)
    res = torch.einsum('bijh,bjhd->bihd', attn, v)
    res = res.view(batch_size, -1, self.n_heads * self.d_k)
    res = self.output(res)
    res += x
    res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
    return res



class DownBlock(nn.Module):
  def __init__(self, in_channels, out_channels, emb_dim, has_att=False):
    super().__init__()
    self.res = ResidualBlockG(in_channels, out_channels, emb_dim)
    if has_att:
      self.att = AttentionBlock(out_channels)
    else:
      self.att = nn.Identity()

  def forward(self, x, t):
    x = self.res(x, t)
    x = self.att(x)

    return x


class UpBlock(nn.Module):
  def __init__(self, in_channels, out_channels, emb_dim, has_att=False):
    super().__init__()
    self.res = ResidualBlockG(in_channels + out_channels, out_channels, emb_dim)

    if has_att:
      self.att = AttentionBlock(out_channels)
    else:
      self.att = nn.Identity()

  def forward(self, x, t):
    x = self.res(x, t)
    x = self.att(x)

    return x



class MiddleBlock(nn.Module):
  def __init__(self, in_channels, emb_dim, has_att=False):
    super().__init__()
    self.res1 = ResidualBlockG(in_channels, in_channels, emb_dim)
    self.res2 = ResidualBlockG(in_channels, in_channels, emb_dim)

    if has_att:
      self.att = AttentionBlock(in_channels)
    else:
      self.att = nn.Identity()

  def forward(self, x, t):
    x = self.res1(x, t)
    x = self.att(x)
    x  = self.res2(x,t)

    return x



class Upsample(nn.Module):
  def __init__(self, n_channels):
    super().__init__()
    self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4,4), (2,2), (1,1))

  def forward(self, x, t):
    _ = t

    return self.conv(x)

class Downsample(nn.Module):
  def __init__(self, n_channels):
    super().__init__()
    self.conv = nn.Conv2d(n_channels, n_channels, (3,3), (2,2), (1,1))

  def forward(self, x, t):
    _ = t

    return self.conv(x)



class Unet(nn.Module):
  def __init__(self, image_channels=4,
               n_channels=64,
               channels_factors=[1, 2, 3, 4],
               att=[False, False, False, False],
               n_blocks=1,
               lr=0.0001
               ):
    super().__init__()
    self.lr = lr
    n_resolutions = len(channels_factors)

    self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

    self.time_emb = TimeEmbedding(n_channels*4)

    down = []

    out_channels = in_channels = n_channels

    for i in range(n_resolutions):
      out_channels = in_channels * channels_factors[i]
      for _ in range(n_blocks):
        down.append(DownBlock(in_channels, out_channels, n_channels*4, att[i]))
        in_channels = out_channels

      if i < n_resolutions - 1:
        down.append(Downsample(in_channels))

    self.down = nn.ModuleList(down)

    self.middle = MiddleBlock(out_channels, n_channels*4, has_att=True)

    ups = []

    in_channels = out_channels
    for i in reversed(range(n_resolutions)):
      out_channels = in_channels
      for _ in range(n_blocks):
        ups.append(UpBlock(in_channels, out_channels, n_channels * 4, att[i]))

      out_channels = in_channels // channels_factors[i]
      ups.append(UpBlock(in_channels, out_channels, n_channels * 4, att[i]))
      in_channels = out_channels

      if i > 0:
        ups.append(Upsample(in_channels))


    self.ups = nn.ModuleList(ups)
    self.norm = nn.GroupNorm(8, n_channels)
    self.act = Swish()
    self.final = nn.Conv2d(in_channels, 3, kernel_size=(3, 3), padding=(1, 1))

  def forward(self, x, t):
    t = self.time_emb(t)
    x = self.image_proj(x)

    h = [x]
    for m in self.down:
      x = m(x, t)
      h.append(x)

    x = self.middle(x, t)

    for m in self.ups:
      if isinstance(m, Upsample):
        x = m(x, t)
      else:
        s = h.pop()
        x = torch.cat((x, s), dim=1)

        x = m(x, t)

    return self.final(self.act(self.norm(x)))