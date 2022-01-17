import torch 
import torchvision
import torch.nn.functional as F
from torch import nn 

"""
Models
- ConvNet
- ResidualNet
- ProfNet
- MnistSimpleCNN
  -> https://arxiv.org/pdf/2008.10400v2.pdf
- SimpleNetv1
  -> https://arxiv.org/pdf/1608.06037v7.pdf
- Resnet18
  -> https://arxiv.org/pdf/1512.03385.pdf
- Resnet18_dd
  -> Resnet18
  -> Depth Dropout: https://users.cecs.anu.edu.au/~sgould/papers/dicta16-depthdropout.pdf
"""

# Generic classes

class ConvBlock(nn.Module):

  def __init__(self, ch_in, ch_out, mpool=False, dropout=False, padding=1, kernel_size=3):
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv2d(ch_in, ch_out, kernel_size, padding=padding)
    self.bn = nn.BatchNorm2d(ch_out)
    self.mpool = mpool
    self.dropout = dropout
    
  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = F.relu(x)

    if self.mpool:
      x = F.max_pool2d(x, kernel_size=2, stride=2)
    if self.dropout:
      x = F.dropout2d(x, p=0.1)
    return x


class LinearBlock(nn.Module):

  def __init__(self, in_size, out_size):
    super(LinearBlock, self).__init__()
    self.lin = nn.Linear(in_size, out_size)
    self.bn = nn.BatchNorm1d(out_size)
    self.relu = nn.ReLU()
    self.drop = nn.Dropout(p=0.5)

  def forward(self, x):
    x = self.lin(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.drop(x)
    return x

"""ConvNet"""
class ConvNet(nn.Module):

  def __init__(self, output_size=27):
    super(ConvNet, self).__init__()

    # 1x32x32
    self.conv1 = ConvBlock(1, 6, mpool=True, dropout=True)
    # 6x16x16
    self.conv2 = ConvBlock(6, 64, mpool=True, dropout=True)
    # 64x8x8
    self.conv3 = ConvBlock(64, 64, dropout=True)
    self.conv4 = ConvBlock(64, 64, dropout=True)
    self.conv5 = ConvBlock(64, 64, dropout=True)
    # 64x8x8
    self.fc1 = LinearBlock(64 * 7 * 7, 512)
    self.fc2 = LinearBlock(512, output_size)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x

"""
ResidualNet building block
"""
class ResidualBlock(nn.Module):

  def __init__(self, channels, residual_count):
    super(ResidualBlock, self).__init__()

    res_arr = [ ConvBlock(channels, channels) for _ in range(residual_count) ]
    self.res_block = nn.ModuleList(res_arr)

  def forward(self, x):
    x_saved = x
    for layer in self.res_block:
      x = layer(x)
    x = x_saved + x
    return x

"""
ResidualNet
"""
class ResidualNet(nn.Module):

  def __init__(self, output_size=27):
    super(ResidualNet, self).__init__()

    # 1x32x32
    self.init1 = ConvBlock(1, 6, mpool=True, dropout=True)
    # 6x16x16
    self.init2 = ConvBlock(6, 64, mpool=True, dropout=True)
    # 64x8x8
    # Three residual blocks
    self.residual = nn.ModuleList([ ResidualBlock(64, residual_count=2) for _ in range(3) ])
    # 64x8x8
    self.fc1 = LinearBlock(64 * 7 * 7, 512)
    self.fc2 = LinearBlock(512, output_size)

  def forward(self, x):

    x = self.init1(x)
    x = self.init2(x)
    for residual in self.residual:
      x = residual(x)

    x = x.view(x.shape[0], -1)
    x = self.fc1(x)
    x = self.fc2(x)
    return x


"""
MnistSimpleCNN

https://paperswithcode.com/paper/an-ensemble-of-simple-convolutional-neural
3x3, 5x5 and 7x7 versions
https://arxiv.org/pdf/2008.10400v2.pdf
"""
class MnistSimpleCNN(nn.Module):
  
  def __init__(self, kernel_size, output_size, dropout=False):
    super(MnistSimpleCNN, self).__init__()
    assert kernel_size in [ 3, 5, 7 ], 'Kernel size must be 3, 5 or 7'
    
    if kernel_size == 3:
      channels = [ 1, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176 ]
      linear_input_size = 8 * 8 * 176
    elif kernel_size == 5:
      channels = [ 1, 32, 64, 96, 128, 160 ]
      linear_input_size = 8 * 8 * 160
    elif kernel_size == 7:
      channels = [ 1, 48, 96, 144, 192 ]
      linear_input_size = 4 * 4 * 192

    self.convolutions = nn.ModuleList()

    for idx, conv_in in enumerate(channels[:-1]):
      conv_out = channels[idx + 1]
      self.convolutions.append(ConvBlock(
          conv_in,
          conv_out,
          kernel_size=kernel_size,
          dropout=dropout,
          padding=0
      ))
    
    self.linear = nn.Linear(linear_input_size, output_size)
    self.bn1d = nn.BatchNorm1d(output_size)

  def forward(self, x):
    for conv_block in self.convolutions:
      x = conv_block(x)
    x = x.view(x.shape[0], -1)
    x = self.linear(x)
    x = self.bn1d(x)
    return x

"""
SimpleNetv1 building block
"""
class SimpleNetv1_block(nn.Module):

  def __init__(self, in_ch, out_ch, kernel_size, max_pool, dropout):
    super(SimpleNetv1_block, self).__init__()
    self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1)
    self.dropout = dropout
    self.max_pool = max_pool

  def forward(self, x):
    x = self.conv(x)
    if self.max_pool:
      x = F.max_pool2d(x, 2, 2)
    x = F.relu(x)
    if self.dropout:
      x = F.dropout2d(x, p=0.1)
    return x


"""
SimpleNetv1

- https://paperswithcode.com/paper/lets-keep-it-simple-using-simple
- https://arxiv.org/pdf/1608.06037v7.pdf
- https://raw.githubusercontent.com/Coderx7/SimpleNet/master/SimpNet_V1/images(plots)/SimpleNet_Arch_Larged.jpg
"""
class SimpleNetv1(nn.Module):
  
  def __init__(self, output_size, dropout=False):
    super(SimpleNetv1, self).__init__()
    self.convolutions = nn.ModuleList()
    initializer = [
      { 'ch_in': 1, 'ch_out': 64 },
      { 'ch_in': 64, 'ch_out': 128 },
      { 'ch_in': 128, 'ch_out': 128 },
      { 'ch_in': 128, 'ch_out': 128, 'max_pool': True },
      { 'ch_in': 128, 'ch_out': 128 },
      { 'ch_in': 128, 'ch_out': 128 },
      { 'ch_in': 128, 'ch_out': 128, 'max_pool': True },
      { 'ch_in': 128, 'ch_out': 128 },
      { 'ch_in': 128, 'ch_out': 128, 'max_pool': True }, # conv9
      { 'ch_in': 128, 'ch_out': 128 },
      { 'ch_in': 128, 'ch_out': 128, 'kernel_size': 1 }, # conv11
      { 'ch_in': 128, 'ch_out': 128, 'kernel_size': 1, 'max_pool': True }, # conv12
    ]

    for sett in initializer:
      max_pool = sett["max_pool"] if 'max_pool' in sett else False
      kernel_size = sett["kernel_size"] if 'kernel_size' in sett else 3
      self.convolutions.append(SimpleNetv1_block(
          sett["ch_in"], 
          sett["ch_out"],
          kernel_size,
          max_pool,
          dropout
      ))
    self.convolutions.append(nn.Conv2d(128, 128, 3, padding=1))
    self.convolutions.append(nn.MaxPool2d(2, 2))
    self.classifier = nn.Linear(128 * 1 * 1, output_size)

  def forward(self, x):
    for layer in self.convolutions:
      x = layer(x)
    x = x.view(x.shape[0], -1)
    x = self.classifier(x)
    return x

"""
Resnet18

- https://arxiv.org/pdf/1512.03385.pdf
"""
class Resnet18_EMNIST(nn.Module):

  def __init__(self, output_size):
    super(Resnet18_EMNIST, self).__init__()
    self.resnet18 = torchvision.models.resnet18(pretrained=False)
    linear_in = self.resnet18.fc.in_features
    
    self.resnet18.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
    self.resnet18.fc = nn.Linear(linear_in, output_size)

    self.once = True

  def forward(self, x):
    return self.resnet18(x)


"""
Resnet18_dropout building block
"""
class Resnet18_dd_block(nn.Module):

  def __init__(self, ch_in, ch_rest, dropout_p=0, out_drop=False):
    super(Resnet18_dd_block, self).__init__()
    
    downsample = True if dropout_p != 0 else False
    stride_in = 2 if downsample else 1
    self.conv1 = nn.Conv2d(ch_in, ch_rest, 3, stride_in, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(ch_rest)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(ch_rest, ch_rest, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(ch_rest)
    self.final_drop = nn.Dropout2d(p=0.1) if out_drop else None

    if downsample:
      self.downsample = nn.Sequential(
          nn.Conv2d(ch_in, ch_rest, 1, 2, bias=False),
          nn.BatchNorm2d(ch_rest),
          nn.Dropout2d(dropout_p)
      )
    else:
      self.downsample = None

  def forward(self, x):
    x_identity = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample != None:
      x_identity = self.downsample(x)

    out += x_identity 
    out = self.relu(out)
    if self.final_drop != None:
      out = self.final_drop(out)
    return out

"""
Resnet18_dropout

Resnet18 with depth dropout defined in:
https://users.cecs.anu.edu.au/~sgould/papers/dicta16-depthdropout.pdf

Drop prob.: residual_drop_p in [ 0.25, 0.5, 0.75 ]
"""
class Resnet18_dd(nn.Module):

  def __init__(self, output_size, residual_drop_p, in_channels=3, out_drop=False):
    super(Resnet18_dd, self).__init__()
    assert residual_drop_p in [ 0.25, 0.5, 0.75 ], 'Invalid depth dropout rate.'

    self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
    self.layer1 = nn.Sequential(
        Resnet18_dd_block(64, 64, out_drop=out_drop),
        Resnet18_dd_block(64, 64, out_drop=out_drop),
    )
    self.layer2 = nn.Sequential(
        Resnet18_dd_block(64, 128, residual_drop_p, out_drop),
        Resnet18_dd_block(128, 128, out_drop=out_drop),
    )
    self.layer3 = nn.Sequential(
        Resnet18_dd_block(128, 256, residual_drop_p, out_drop),
        Resnet18_dd_block(256, 256, out_drop=out_drop),
    )
    self.layer4 = nn.Sequential(
        Resnet18_dd_block(256, 512, residual_drop_p, out_drop),
        Resnet18_dd_block(512, 512, out_drop=out_drop),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Linear(512, output_size)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)
    return x


"""ProfNet"""
class ProfNet(nn.Module):

    def __init__(self, output_size):
        super(ProfNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout1 = nn.Dropout(p = 0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.dropout2 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(512, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(p = 0.5)

        self.fc3 = nn.Linear(512, output_size)

    def forward(self, x):
        # Konvolucijski slojevi
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.maxpool2(x)

        # Linearni slojevi
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x
