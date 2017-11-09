import sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable
import primitives

'''
Predicts reflectance, shape, and lighting conditions given an image

Reflectance and shape are 3-channel images of the 
same dimensionality as input (expects 256x256). 
Lights have dimensionality lights_dim. By default,
they are represented as [x, y, z, energy].
'''
class Decomposer(nn.Module):

    def __init__(self, lights_dim = 4):
        super(Decomposer, self).__init__()

        #######################
        #### shape encoder #### 
        #######################
        ## there is a single shared convolutional encoder
        ## for all intrinsic images
        channels = [3, 16, 32, 64, 128, 256, 256]
        kernel_size = 3
        padding = 1
        ## stride of 1 on first layer and 2 everywhere else
        stride_fn = lambda ind: 1 if ind==0 else 2
        sys.stdout.write( '<Decomposer> Building Encoder' )
        self.encoder = primitives.build_encoder(channels, kernel_size, padding, stride_fn)

        #######################
        #### shape decoder #### 
        #######################
        ## link encoder and decoder
        channels.append( channels[-1] )
        ## reverse channel order for decoder
        channels = list(reversed(channels))
        stride_fn = lambda ind: 1
        sys.stdout.write( '<Decomposer> Building Decoder' )
        ## separate reflectance and normals decoders.
        ## mult = 2 because the skip layer concatenates
        ## an encoder layer with the decoder layer,
        ## so the number of input channels in each layer is doubled.
        self.decoder_reflectance = primitives.build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        self.decoder_normals = primitives.build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        channels[-1] = 1
        self.decoder_depth = primitives.build_encoder(channels, kernel_size, padding, stride_fn, mult=2)
        self.upsampler = nn.UpsamplingNearest2d(scale_factor=2)

        #### lights encoder
        lights_channels = [256, 128, 64]
        stride_fn = lambda ind: 2
        sys.stdout.write( '<Decomposer> Lights Encoder  ' )
        self.decoder_lights = primitives.build_encoder(lights_channels, kernel_size, padding, stride_fn)
        lights_encoded_dim = 2
 
        self.lights_fc1 = nn.Linear(lights_channels[-1] * (lights_encoded_dim ** 2), 32)
        self.lights_fc2 = nn.Linear(32, lights_dim)

    def __decode(self, decoder, encoded, inp):
        x = inp
        for ind in range(len(decoder)-1):
            x = decoder[ind](x)
            if ind != 0:
                x = self.upsampler(x)
            x = primitives.join(1)(x, encoded[-(ind+1)])
            x = F.leaky_relu(x)

        x = decoder[-1](x)
        return x

    def forward(self, inp, mask):
        ## shared encoder
        x = inp
        encoded = []
        for ind in range(len(self.encoder)):
            x = self.encoder[ind](x)
            x = F.leaky_relu(x)
            encoded.append(x)

        ## decode lights
        lights = x
        for ind in range(len(self.decoder_lights)):
            lights = self.decoder_lights[ind](lights)
            lights = F.leaky_relu(lights)
        lights = lights.view(lights.size(0), -1)
        lights = F.leaky_relu( self.lights_fc1(lights) )
        lights = self.lights_fc2(lights)

        ## separate decoders
        reflectance = self.__decode(self.decoder_reflectance, encoded, x)
        normals = self.__decode(self.decoder_normals, encoded, x)
        depth = self.__decode(self.decoder_depth, encoded, x)

        ## R, G in [-1,1]
        rg = torch.clamp(normals[:,:-1,:,:], -1, 1)
        ## B in [0,1]
        b = torch.clamp(normals[:,-1,:,:].unsqueeze(1), 0, 1)
        clamped = torch.cat((rg, b), 1)
        ## normals are unit vector
        normed = primitives.normalize(clamped)

        ## turn float mask into bool array
        mask = mask < 0.25
        ## set background pixels to 0 so 
        ## we don't count them in error
        reflectance[mask] = 0
        normed[mask] = 0
        depth[mask[:,0]] = 0

        return reflectance, depth, normed, lights


if __name__ == '__main__':
    inp = Variable(torch.randn(5,3,256,256))
    mask = Variable(torch.randn(5,3,256,256))
    # lights = Variable(torch.randn(5,4))
    decomposer = Decomposer()
    out = decomposer.forward(inp, mask)
    print decomposer
    print [i.size() for i in out]



