import sys, torch, torch.nn as nn, torch.nn.functional as F, pdb
from torch.autograd import Variable
import primitives

class Composer(nn.Module):

    def __init__(self, decomposer, shader):
        super(Composer, self).__init__()

        self.decomposer = decomposer
        self.shader = shader

    def forward(self, inp, mask):
        reflectance, depth, shape, lights = self.decomposer(inp, mask)
        # print(reflectance.size(), lights.size())
        shading = self.shader(shape, lights)
        shading_rep = shading.repeat(1,3,1,1)
        # print(shading.size())
        reconstruction = reflectance * shading_rep
        return reconstruction, reflectance, depth, shape, lights, shading


if __name__ == '__main__':
    import sys
    sys.path.append('../')
    import models
    decomposer_path = '../logs/separated_decomp_0.01lr_0.1lights/model.t7'
    shader_path = '../logs/separated_shader_0.01/model.t7'
    decomposer = torch.load(decomposer_path)
    shader = torch.load(shader_path)
    composer = Composer(decomposer, shader).cuda()
    print composer
    # pdb.set_trace()
    inp = Variable(torch.randn(5,3,256,256).cuda())
    mask = Variable(torch.randn(5,3,256,256).cuda())

    out = composer.forward(inp, mask)

    print [i.size() for i in out]

    # pdb.set_trace()




