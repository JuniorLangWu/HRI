import copy as cp
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from ...utils import Graph, cache_checkpoint
from ..builder import BACKBONES
from .utils import mstcn, unit_gcn, unit_tcn
import numpy as np
EPS = 1e-4


class STGCNBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 stride=1,
                 residual=True,
                 **kwargs):
        super().__init__()

        gcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'gcn_'}
        tcn_kwargs = {k[4:]: v for k, v in kwargs.items() if k[:4] == 'tcn_'}
        kwargs = {k: v for k, v in kwargs.items() if k[:4] not in ['gcn_', 'tcn_']}
        assert len(kwargs) == 0, f'Invalid arguments: {kwargs}'

        tcn_type = tcn_kwargs.pop('type', 'unit_tcn')
        assert tcn_type in ['unit_tcn', 'mstcn']
        gcn_type = gcn_kwargs.pop('type', 'unit_gcn')
        assert gcn_type in ['unit_gcn']

        self.gcn = unit_gcn(in_channels, out_channels, A, **gcn_kwargs)

        if tcn_type == 'unit_tcn':
            self.tcn = unit_tcn(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == 'mstcn':
            self.tcn = mstcn(out_channels, out_channels, stride=stride, **tcn_kwargs)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)  


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
 
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
 
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
 
        # Multi-scale information fusion
        y = self.sigmoid(y)
 
        return x * y.expand_as(x)
    
class Attention_LG(nn.Module):
    def __init__(self, dim, kernel_size=3, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.kernel_size = kernel_size
        self.lconv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=1, bias=False),
                                   norm_layer(dim, eps=1e-6),
                                   act_layer())

        self.gconv = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                   norm_layer(dim, eps=1e-6),
                                   nn.Conv2d(dim, dim, kernel_size=(1, 1), bias=False),
                                   act_layer())
        self.act = act_layer()
        self.norm = norm_layer(dim, eps=1e-6)
        self.linear = nn.Conv2d(2*dim, dim, kernel_size=(1, 1), bias=False)

    def forward(self, x):
        # b, c, h, w = x.shape
        # x = x.permute(0, 3, 1, 2).contiguous()
        lconv = self.lconv(x)   # [b, c, h, w]
        gconv = self.gconv(x)   # [b, c, h, w]
        hconv = self.act(torch.cat([lconv, gconv], dim=1))  # [b, 2c, h, w]
        hconv = self.linear(hconv)  # [b, c, h, w]
        out = self.norm(x*hconv)
        # out = out.permute(0, 2, 3, 1).contiguous()
        return out
    
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.key = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.value = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 获取输入的形状
        B, C, H, W = x.size()
        
        # 生成Query, Key, Value矩阵
        query = self.query(x).view(B, C, -1)  # [B, C, H*W]
        key = self.key(x).view(B, C, -1)      # [B, C, H*W]
        value = self.value(x).view(B, C, -1)  # [B, C, H*W]

        # 计算Query和Key的点积
        attention_scores = torch.bmm(query.permute(0, 2, 1), key)  # [B, H*W, H*W]
        
        # 通过softmax计算注意力权重
        attention_weights = self.softmax(attention_scores)  # [B, H*W, H*W]
        
        # 计算加权值
        attention_output = torch.bmm(value, attention_weights.permute(0, 2, 1))  # [B, C, H*W]
        
        # 恢复原始形状
        attention_output = attention_output.view(B, C, H, W)  # [B, C, H, W]
        
        # 将输入和注意力输出相加 (残差连接)
        out = x + attention_output

        return out
    
class NHAttention(nn.Module):
    def __init__(self,
                 dim,   # 输入token的dim
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super().__init__()
        self.value = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        # 获取输入的形状
        B, C, H, W = x.size()
        value = self.value(x).view(B, C, -1)  # [B, C, H*W]
        x = value.permute(0, 2, 1)  # [B, H*W, C]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = value.permute(0, 2, 1)  # [B, C, H*W]
        x = x.view(B, C, H, W)  # [B, C, H, W]
        return x

def init_MMTMweights(m):
#  print(m)
 if type(m) == nn.Linear:
#    print(m.weight)
    pass
 else:
   print('error')

class MMTM(nn.Module):
  def __init__(self, dim_visual, dim_skeleton, ratio):
    super(MMTM, self).__init__()
    dim = dim_visual + dim_skeleton
    dim_out = int(2*dim/ratio)
    self.fc_squeeze = nn.Linear(dim, dim_out)

    self.fc_visual = nn.Linear(dim_out, dim_visual)
    self.fc_skeleton = nn.Linear(dim_out, dim_skeleton)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()

    # initialize
    with torch.no_grad():
      self.fc_squeeze.apply(init_MMTMweights)
      self.fc_visual.apply(init_MMTMweights)
      self.fc_skeleton.apply(init_MMTMweights)

  def forward(self, visual, skeleton):
    squeeze_array = []
    for tensor in [visual, skeleton]:
      tview = tensor.view(tensor.shape[:2] + (-1,))
      squeeze_array.append(torch.mean(tview, dim=-1))
    squeeze = torch.cat(squeeze_array, 1)

    excitation = self.fc_squeeze(squeeze)
    excitation = self.relu(excitation)

    vis_out = self.fc_visual(excitation)
    sk_out = self.fc_skeleton(excitation)

    vis_out = self.sigmoid(vis_out)
    sk_out = self.sigmoid(sk_out)

    dim_diff = len(visual.shape) - len(vis_out.shape)
    vis_out = vis_out.view(vis_out.shape + (1,) * dim_diff)

    dim_diff = len(skeleton.shape) - len(sk_out.shape)
    sk_out = sk_out.view(sk_out.shape + (1,) * dim_diff)

    return visual * vis_out, skeleton * sk_out


@BACKBONES.register_module()
class STGCN(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # print(x[0,0,0,:,1])
        # print(x.size())
        # exit(0)
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            x = self.gcn[i](x)

        x = x.reshape((N, M) + x.shape[1:])
        return x   
    
@BACKBONES.register_module()
class STGCNCOSECA(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages

        modules = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
        self.context_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.eca_layer = eca_layer(out_channels)
        self.gamma = nn.Parameter(torch.zeros(1))

        if self.in_channels == self.base_channels:
            num_stages -= 1

        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.pretrained = pretrained

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = nn.functional.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        return y_weight
    
    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        # print(x.size())
        # exit(0)
        if self.data_bn_type == 'MVC':
            x = self.data_bn(x.view(N, M * V * C, T))
        else:
            x = self.data_bn(x.view(N * M, V * C, T))
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        for i in range(self.num_stages):
            # print(i)
            x = self.gcn[i](x)

        # 计算注意力权重
        context = self.context_conv(x)
        # print(context.shape)
        attention_weights = self.weight(x, context)
        # print(attention_weights.shape)

        # attention_output = attention_weights*self.eca_layer(x)
        attention_output = self.eca_layer(attention_weights*x)
        # print(attention_output.shape)
        # exit(0)
        x = self.gamma * attention_output + x

        x = x.reshape((N, M) + x.shape[1:])
        # print(x.shape)
        # exit(0)
        return x
    
@BACKBONES.register_module()
class STGCNMB(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
            self.data_bn_b = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
            self.data_bn_b = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()
            self.data_bn_b = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        modules_b = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]
            modules_b = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            modules_b.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.gcn_b = nn.ModuleList(modules_b)

        self.pretrained = pretrained

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = nn.functional.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        return y_weight
    
    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        # print(x.size())
        # exit(0)
        xj = x[:,:,:,:,:3]
        xb = x[:,:,:,:,3:]
        # print(xj.shape)
        # print(xb.shape)
        # exit(0)
        N, M, T, V, C = xj.size()
        xj = xj.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            xj = self.data_bn(xj.view(N, M * V * C, T))
        else:
            xj = self.data_bn(xj.view(N * M, V * C, T))
        xj = xj.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        xb = xb.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            xb = self.data_bn_b(xb.view(N, M * V * C, T))
        else:
            xb = self.data_bn_b(xb.view(N * M, V * C, T))
        xb = xb.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):
            # print(i)
            xj = self.gcn[i](xj)
            xb = self.gcn_b[i](xb)
            # print(x.shape)           

        x = torch.cat((xj, xb), dim=1)
        # print(x.shape)
        # exit(0)


        x = x.reshape((N, M) + x.shape[1:])
        # print(x.shape)
        # exit(0)
        return x

@BACKBONES.register_module()
class STGCNMBLATE(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
            self.data_bn_b = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
            self.data_bn_b = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()
            self.data_bn_b = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        modules_b = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]
            modules_b = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            modules_b.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))

        if self.in_channels == self.base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.gcn_b = nn.ModuleList(modules_b)

        self.pretrained = pretrained

    def weight(self, value, y_last):
        y_weight = torch.cosine_similarity(value, y_last, dim=1)
        y_weight = nn.functional.softmax(y_weight, dim=-1)
        y_weight = y_weight.unsqueeze(1)
        return y_weight
    
    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        # print(x.size())
        # exit(0)
        xj = x[:,:,:,:,:3]
        xb = x[:,:,:,:,3:]
        # print(xj.shape)
        # print(xb.shape)
        # exit(0)
        N, M, T, V, C = xj.size()
        xj = xj.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            xj = self.data_bn(xj.view(N, M * V * C, T))
        else:
            xj = self.data_bn(xj.view(N * M, V * C, T))
        xj = xj.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        xb = xb.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            xb = self.data_bn_b(xb.view(N, M * V * C, T))
        else:
            xb = self.data_bn_b(xb.view(N * M, V * C, T))
        xb = xb.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):
            # print(i)
            xj = self.gcn[i](xj)
            xb = self.gcn_b[i](xb)
            # print(x.shape)           

        return (xj, xb)    

@BACKBONES.register_module()
class STGCNMBMMTMLATE(nn.Module):

    def __init__(self,
                 graph_cfg,
                 in_channels=3,
                 base_channels=64,
                 data_bn_type='VC',
                 ch_ratio=2,
                 num_person=2,  # * Only used when data_bn_type == 'MVC'
                 num_stages=10,
                 inflate_stages=[5, 8],
                 down_stages=[5, 8],
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.data_bn_type = data_bn_type
        self.kwargs = kwargs

        if data_bn_type == 'MVC':
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
            self.data_bn_b = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == 'VC':
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
            self.data_bn_b = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()
            self.data_bn_b = nn.Identity()

        lw_kwargs = [cp.deepcopy(kwargs) for i in range(num_stages)]
        for k, v in kwargs.items():
            if isinstance(v, tuple) and len(v) == num_stages:
                for i in range(num_stages):
                    lw_kwargs[i][k] = v[i]
        lw_kwargs[0].pop('tcn_dropout', None)

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules = []
        modules_b = []
        mmtm_list = []
        if self.in_channels != self.base_channels:
            modules = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]
            modules_b = [STGCNBlock(in_channels, base_channels, A.clone(), 1, residual=False, **lw_kwargs[0])]

        inflate_times = 0
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_channels = base_channels
            if i in inflate_stages:
                inflate_times += 1
            out_channels = int(self.base_channels * self.ch_ratio ** inflate_times + EPS)
            base_channels = out_channels
            modules.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            modules_b.append(STGCNBlock(in_channels, out_channels, A.clone(), stride, **lw_kwargs[i - 1]))
            if 2<= i < num_stages:
                mmtm_list.append(MMTM(out_channels, out_channels, 4))
        if self.in_channels == self.base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.gcn_b = nn.ModuleList(modules_b)
        self.mmtm_fuse = nn.ModuleList(mmtm_list)
        self.pretrained = pretrained
    
    def init_weights(self):
        if isinstance(self.pretrained, str):
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False)

    def forward(self, x):
        # print(x.size())
        # exit(0)
        xj = x[:,:,:,:,:3]
        xb = x[:,:,:,:,3:]
        # print(xj.shape)
        # print(xb.shape)
        # exit(0)
        N, M, T, V, C = xj.size()
        xj = xj.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            xj = self.data_bn(xj.view(N, M * V * C, T))
        else:
            xj = self.data_bn(xj.view(N * M, V * C, T))
        xj = xj.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        xb = xb.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == 'MVC':
            xb = self.data_bn_b(xb.view(N, M * V * C, T))
        else:
            xb = self.data_bn_b(xb.view(N * M, V * C, T))
        xb = xb.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        j = 0
        for i in range(self.num_stages):
            # print(i)
            xj = self.gcn[i](xj)
            xb = self.gcn_b[i](xb)
            # print(x.shape)           
            if 1<= i < self.num_stages-1:
                xj,xb = self.mmtm_fuse[j](xj,xb)
                j+1
        return (xj, xb) 





