import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC , abstractmethod
import math
import logging
from einops import rearrange

# Set up logging
logging.basicConfig(level=logging.CRITICAL + 1, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#######################################################################
################################# ABSTRACT BLOCKS #####################
#######################################################################
class XCBlock(nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

class XBlock(nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

class XTCBlock(nn.Module):
    @abstractmethod
    def forward(self, x, t, c):
        pass

class TBlock(nn.Module):
    @abstractmethod
    def forward(self, t):
        pass


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class XTBlock(nn.Module):
    @abstractmethod
    def forward(self, x, t):
        pass


class Downsample(XBlock):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, recompute_scale_factor=True)



class Upsample(XBlock):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


######################################################################
########################### TIME EMBEDDING BLOCKS ##########################
#####################################################################
class SinusoidalTimeEmbedder(TBlock):
    def __init__(self, base_channels, max_period = 10000):
        super().__init__()
        self.base_channels = base_channels
        self.max_period = max_period
        self.half = base_channels // 2
        self.freqs = torch.exp(-math.log(max_period) * torch.arange(self.half, dtype=torch.float32) / self.half)

    def forward(self, t):
        freqs = self.freqs.to(t.device)
        args = t[:, None].float() * freqs
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.base_channels % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class LearnableTimeEmbedder(TBlock):
    def __init__(self, base_channels, time_embedding_dim):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            nn.Linear(base_channels, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim),
        )

    def forward(self, t):
        t = self.time_embed(t)
        return t


class TimeEmbedder(TBlock):
    def __init__(self, base_channels, time_embedding_dim):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedder(base_channels),
            LearnableTimeEmbedder(base_channels, time_embedding_dim)
        )
    def forward(self, t):
        return self.time_embed(t)



class TimeInjector(XTBlock):
    def __init__(self, time_embedding_dim, in_channels):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, in_channels)
        )
        self.proj_out = nn.Sequential(
                        nn.GroupNorm(32, in_channels),
                        nn.SiLU(),
                        zero_module(nn.Conv2d(in_channels, in_channels, 3, padding=1))
        )
    
    def forward(self, x, t):
        t = self.time_embed(t).type(x.dtype)
        while len(t.shape) < len(x.shape):
            t = t[..., None]
        return self.proj_out(x + t)
    
############################################################
################### CONVOLUTIONAL BLOCKS ###################
#############################################################
class InConvBlock(XBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
    def forward(self, x):
        return self.in_layers(x)
    

class OutConvBlock(XBlock):
    def __init__(self, out_channels):
        super().__init__()
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        )
    
    def forward(self, x):
        return self.out_layers(x)
    


class ResBlock(XTBlock):
    def __init__(self, in_channels, time_embedding_dim):
        super().__init__()
        self.inconv = InConvBlock(in_channels, in_channels)
        self.injector = TimeInjector(time_embedding_dim, in_channels)
    def forward(self, x, t):
        h = self.inconv(x)
        h = self.injector(h, t)
        return x + h   


##############################################################
############################ MAGIC ###########################
##############################################################

class Connection:
    def __init__(self, start_block, target_block, operation):
        self.target_block = target_block
        self.start_block = start_block
        self.operation = operation
        self.collected_tensor = None

    def is_target_block(self, name):
        return name == self.target_block

    def is_start_block(self, name):
        return name == self.start_block

    def excute_operation(self, x):
        return self.operation(x, self.collected_tensor)

    def collect(self, x):
        self.collected_tensor = x

    def __repr__(self):
        return (f"Connection(start_block={self.start_block}, "
                f"target_block={self.target_block}, "
                f"operation={self.operation.__name__ if hasattr(self.operation, '__name__') else str(self.operation)}, "
                f"collected_tensor={'Set' if self.collected_tensor is not None else 'None'})")



class Router(nn.Sequential):
    def __init__(self, connections, *args):
        super(Router, self).__init__(*args)
        
        self.connections = connections

    def forward(self, x, t, c=None):

        for block_name, block in self.named_children():
            logger.debug(f'{block.__class__.__name__}')
            for connection in self.connections:
                if connection.is_target_block(block_name):
                    x = connection.excute_operation(x)
                    logger.debug(f'{x.shape}')

            logger.debug(f'{x.shape}')
            if isinstance(block, XTBlock):
                x = block(x, t)
            elif isinstance(block, TBlock):
                x = block(t)
            elif isinstance(block, XBlock):
                x = block(x)
            elif isinstance(block, XTCBlock):
                x = block(x, t, c)
            elif isinstance(block, XCBlock):
                x = block(x, c)            
            logger.debug(f'{x.shape}')

            for connection in self.connections:
                if connection.is_start_block(block_name):
                    connection.collect(x)
        return x
    
class ChannelChanger(XBlock):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    def forward(self, x):
        return self.layer(x)


def connection_finder(unet, name):
    names = []
    left = []
    right = []
    for i, block in enumerate(unet):
        if block.__class__.__name__ == name:
            names.append(str(i))
    
    while len(names) != 1:
        left.append(names.pop(0))
        right.append(names.pop())

    connections = [Connection(first, second, concat) for first, second in zip(left, right)]
    logger.debug(f'left: {left}')
    logger.debug(f'right: {right}')

    return connections



def concat(x, y):
    return torch.cat([x, y], dim=1)

################################################################
################### ATTENTION BLOCKS ###########################
################################################################
                    
class CrossAttention(XCBlock):
    def __init__(self, d_query, d_context, n_heads, head_dim):
        super().__init__()
        
        inner_dim = n_heads*head_dim
        self.n_heads = n_heads
        self.scale = head_dim**-0.5
        
        self.Q = nn.Linear(d_query, inner_dim, bias=False)
        self.K = nn.Linear(d_context, inner_dim, bias=False)
        self.V = nn.Linear(d_context, inner_dim, bias=False)
        
        self.proj_out = nn.Linear(inner_dim, d_query)
        self.norm = nn.LayerNorm(d_query)

    def forward(self, x, context):
        h = self.norm(x)
        
        Q = self.Q(h)
        K = self.K(context)
        V = self.V(context)

        Q, K, V = map(lambda t: rearrange( t, 'b n (h d) -> (b h) n d', h=self.n_heads), (Q, K, V))

        attention = torch.einsum('b i d, b j d -> b i j', Q, K)
        attention = attention.softmax(dim=-1)
        attention = attention*self.scale

        values = torch.einsum('b i j, b j d -> b i d', attention, V)
        values = rearrange(values, '(b h) n d -> b n (h d)', h=self.n_heads)
        values = self.proj_out(values)
        return x + values


class SelfAttention(XBlock):
    def __init__(self, d, n_heads, head_dim):
        super().__init__()
        
        inner_dim = n_heads*head_dim
        self.n_heads = n_heads
        self.scale  = head_dim**-0.5
        
        self.Q = nn.Linear(d, inner_dim, bias=False)
        self.K = nn.Linear(d, inner_dim, bias=False)
        self.V = nn.Linear(d, inner_dim, bias=False)
        
        self.proj_out = nn.Linear(inner_dim, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        h = self.norm(x)
        
        Q = self.Q(h)
        K = self.K(h)
        V = self.V(h)

        Q, K, V = map(lambda t: rearrange( t, 'b n (h d) -> (b h) n d', h=self.n_heads), (Q, K, V))

        attention = torch.einsum('b i d, b j d -> b i j', Q, K)
        attention = attention.softmax(dim=-1)
        attention = attention*self.scale

        values = torch.einsum('b i j, b j d -> b i d', attention, V)
        values = rearrange(values, '(b h) n d -> b n (h d)', h=self.n_heads)
        values = self.proj_out(values)
        return x + values 

class GEGLU(XBlock):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        
        x, gate = self.proj(x).chunk(2, dim=-1)
        
        return x * F.gelu(gate)
    

class FeedForwardGEGLU(XBlock):
    def __init__(self, d_query, dropout=0.):
        super().__init__()
        
        self.net = nn.Sequential(nn.LayerNorm(d_query),
            GEGLU(d_query, d_query*4),
            nn.Dropout(dropout),
            nn.Linear(d_query*4, d_query)
        )

    def forward(self, x):
        return self.net(x) + x
    
class FeedForwardGLU(XBlock):
    def __init__(self, d_query, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_query),
            nn.Linear(d_query, 4*d_query),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_query, d_query)
        )
    def forward(self, x):
        return self.net(x) + x
    

class Adapter(XBlock):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return rearrange(x , 'b c h w -> b (h w) c')
    
    
class UnetBlock(XTCBlock):
    def __init__(self,
                  in_channels,
                  context_dim=128,
                  time_embedding_dim=128,
                  n_heads=8,
                  head_dim=32
    ):
        super().__init__()
        self.block = Router([],
            *[
            ResBlock(in_channels, time_embedding_dim),
            Adapter(),
            SelfAttention(in_channels, n_heads, head_dim),
            CrossAttention(in_channels, context_dim, n_heads, head_dim),
            FeedForwardGEGLU(in_channels)   
            ]
        )
    def forward(self, x, t, context):
        _, _, h, w = x.shape
        x = self.block(x, t, context)
        return rearrange(x , 'b (h w) c -> b c h w', h=h, w=w)

#######################################################
####################### UNET ##########################
#######################################################

class ResChain(XTBlock):
    def __init__(self, in_channels, num_resblocks, time_embedding_dim=128):
        super().__init__()
        self.resblocks = nn.ModuleList(
            [
                ResBlock(in_channels, time_embedding_dim) for _ in range(num_resblocks)
            ]
                        
        )
  

    def forward(self, x, t):
        for block in self.resblocks:
            x = block(x, t)
        return x
    
class Unet(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], num_resblocks=5):
        super().__init__()        
        self.in_proj = nn.Conv2d(1, channels[0], 3, padding=1)
        self.out_proj = nn.Conv2d(channels[0], 1, 3, padding=1)

        self.time_embedder = TimeEmbedder(channels[0], time_embedding_dim=128)
        
        
        self.down = []
        for i in range(len(channels)-1):
            self.down.append(ResChain(channels[i], num_resblocks))
            self.down.append(Downsample(0.5, 'nearest'))
            self.down.append(ChannelChanger(channels[i], channels[i+1]))
        
        self.mid = [ResChain(channels[-1], num_resblocks)]

        channels_reversed = list(reversed(channels))
        self.up = []
        for i in range(len(channels_reversed)-1):
            self.up.append(Upsample(2, 'nearest'))
            self.up.append(ChannelChanger(channels_reversed[i], channels_reversed[i+1]))
            self.up.append(ResChain(2*channels_reversed[i+1], num_resblocks))
            self.up.append(ChannelChanger(2*channels_reversed[i+1], channels_reversed[i+1]))

        unet = self.down + self.mid + self.up
        connections = connection_finder(unet, 'ResChain')
        self.unet = Router(connections, *unet)




    def forward(self, x, t):
        t = self.time_embedder(t)
        x = self.in_proj(x)
        x = self.unet(x, t)
        return self.out_proj(x) 





