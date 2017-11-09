import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable

def conv(in_channels, out_channels, kernel_size, stride, padding):
    convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    batch_norm = nn.BatchNorm2d(out_channels)
    layer = nn.Sequential(convolution, batch_norm)
    return layer

## Returns function to concatenate tensors.
## Used in skip layers to join encoder and decoder layers.
def join(ind):
    return lambda x, y: torch.cat( (x,y), ind )

def decode(encoder, decoder, upsampler): 
    def forward(shape, lights):
        encoded = []
        for ind in range(len(encoder)):
            x = encoder[ind](x)
            x = F.leaky_relu(x)
            encoded.append(x)

        encoded[-1] = torch.cat( (encoded[-1], ) )

        for ind in range(len(decoder)-1):
            x = decoder[ind](x)
            if ind != 0:
                x = upsampler(x)
            x = join(1)(x, encoded[-(ind+1)])
            x = F.leaky_relu(x)

        x = decoder[-1](x)

        return x
    return forward

## normalize to unit vectors
def normalize(normals):
    magnitude = torch.pow(normals, 2).sum(1)
    magnitude = magnitude.sqrt().repeat(1,3,1,1)
    normed = normals / (magnitude + 1e-6)
    return normed

## channels : list of ints
## kernel_size : int
## padding : int
## stride_fn : fn(channel_index) --> int
## mult=1 if encoder, 2 if decoder
def build_encoder(channels, kernel_size, padding, stride_fn, mult=1):
    layers = []
    sys.stdout.write( '    %3d' % channels[0] )
    for ind in range(len(channels)-1):
        m = 1 if ind == 0 else mult
        in_channels = channels[ind] * m
        out_channels = channels[ind+1]
        stride = stride_fn(ind)
        sys.stdout.write( ' --> %3d' % out_channels )

        if ind < len(channels)-2:
            block = conv(in_channels, out_channels, kernel_size, stride, padding)
        else:
            block = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        layers.append(block)
    sys.stdout.write('\n')
    sys.stdout.flush()
    return nn.ModuleList(layers)
