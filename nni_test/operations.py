import torch
import torch.nn as nn



OPS = {
  'none': lambda C_in, C_out, stride, affine, C_mid=None: Zero(stride),
  'avg_pool_3x3': lambda C_in, C_out, stride, affine, C_mid=None: ResizablePool(C_in, C_out, 3, stride, padding=1, affine=affine, pool_type=nn.AvgPool2d),
  'max_pool_3x3': lambda C_in, C_out, stride, affine, C_mid=None: ResizablePool(C_in, C_out, 3, stride, padding=1, affine=affine),
  'skip_connect': lambda C_in, C_out, stride, affine, C_mid=None: Identity() if stride == 1 else FactorizedReduce(C_in, C_out, 1, stride, 0, affine=affine),
  'sep_conv_3x3': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 3, stride, padding=1, affine=affine),
  'sep_conv_5x5': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 5, stride, padding=2, affine=affine),
  'flood_conv_3x3': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 3, stride, padding=2, affine=affine, C_mid_mult=4),
  'flood_conv_5x5': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 5, stride, padding=2, affine=affine, C_mid_mult=4),
  'sep_conv_7x7': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 7, stride, padding=3, affine=affine),
  'dil_conv_3x3': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 3, stride, padding=2, dilation=2, affine=affine),
  'dil_conv_5x5': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 5, stride, padding=4, dilation=2, affine=affine),
  'conv_7x1_1x7': lambda C_in, C_out, stride, affine, C_mid=None: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C_in, C_in, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C_in, C_out, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C_out, eps=1e-3, affine=affine)
    ),
  'flood_conv_3x3': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 3, stride, padding=1, affine=affine, C_mid_mult=4),
  'dil_flood_conv_3x3': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 3, stride, padding=2, dilation=2, affine=affine, C_mid_mult=4),
  'dil_flood_conv_5x5': lambda C_in, C_out, stride, affine, C_mid=None: SharpSepConv(C_in, C_out, 5, stride, padding=2, dilation=2, affine=affine, C_mid_mult=4),
  'choke_conv_3x3': lambda C_in, C_out, stride, affine, C_mid=32: SharpSepConv(C_in, C_out, 3, stride, padding=1, affine=affine, C_mid=C_mid),
  'dil_choke_conv_3x3': lambda C_in, C_out, stride, affine, C_mid=32: SharpSepConv(C_in, C_out, 3, stride, padding=2, dilation=2, affine=affine, C_mid=C_mid),
}



PRIMITIVES = [
    'none',
    'max_pool_3x3',
    # 'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    # 'sep_conv_5x5',
    'dil_conv_3x3',
    # 'dil_conv_5x5',
    # 'nor_conv_3x3',
    # 'nor_conv_5x5',
    # 'nor_conv_7x7',
    'flood_conv_3x3',
    'dil_flood_conv_3x3',
    'choke_conv_3x3',
    'dil_choke_conv_3x3',
]

SHARPER_PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'dil_conv_3x3',
    'dil_conv_5x5',
    # 'nor_conv_3x3',
    # 'nor_conv_5x5',
    # 'nor_conv_7x7',
    'flood_conv_3x3',
    'flood_conv_5x5',
    'dil_flood_conv_3x3',
    # TODO(ahundt) sharpsepconv doesn't correctly support dil_flood_conv_5x5, padding is not sufficient
    # w shape: torch.Size([]) op type: <class 'operations.SharpSepConv'> i: 12 self._primitives[i]: dil_flood_conv_5x5x size: torch.Size([16, 16, 32, 32]) stride: 1
    # op_out size: torch.Size([16, 16, 28, 28])
    # 'dil_flood_conv_5x5',
    # 'choke_conv_3x3',
    # 'dil_choke_conv_3x3',
]

# Primitives for the original darts search space
DARTS_PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


class ConvBNReLU(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ConvBNReLU, self).__init__()
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      nn.ReLU(inplace=False)
    )

  def forward(self, x):
    return self.op(x)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)



class ResizablePool(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, affine=True, pool_type=nn.MaxPool2d):
    super(ResizablePool, self).__init__()
    if C_in == C_out:
      self.op = pool_type(kernel_size=kernel_size, stride=stride, padding=padding)
    else:
      self.op = nn.Sequential(
        pool_type(kernel_size=kernel_size, stride=stride, padding=padding),
        nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, eps=1e-3, affine=affine)
      )

  def forward(self, x):
    return self.op(x)


class SharpSepConv(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1, dilation=1, affine=True, C_mid_mult=1, C_mid=None):
    super(SharpSepConv, self).__init__()
    if C_mid is not None:
      c_mid = C_mid
    else:
      c_mid = int(C_out * C_mid_mult)

    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, c_mid, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(c_mid, affine=affine),
      nn.ReLU(inplace=False),
      # Padding is set based on the kernel size for this convolution which is always stride 1 and not dilated
      # https://pytorch.org/docs/stable/nn.html#conv2d
      # H_out = (((H_in + (2*padding) − dilation * (kernel_size − 1) − 1 ) / stride) + 1)
      nn.Conv2d(c_mid, c_mid, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, dilation=1, groups=c_mid, bias=False),
      nn.Conv2d(c_mid, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)




class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, kernel_size=1, stride=2, padding=0, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, kernel_size, stride=stride, padding=padding, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, kernel_size, stride=stride, padding=padding, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)
    self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

  def forward(self, x):
    x = self.relu(x)
    y = self.pad(x)
    out = torch.cat([self.conv_1(x), self.conv_2(y[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out