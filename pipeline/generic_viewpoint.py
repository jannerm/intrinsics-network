import torch, torch.nn.functional as F, pdb
from torch.autograd import Variable

'''
returns < num_relights x 3 x M x N > 
'''
def relight(shader, normals, lights, num_relights, sigma=2.5):
    shadings = Variable( torch.zeros([num_relights] + [i for i in normals.size()]), requires_grad=False )
    for i in range(num_relights):
        lights_delta = Variable( torch.randn(lights.size()).cuda() * sigma )
        ## energy should stay the same
        lights_delta[:,0] = 0
        # pdb.set_trace()
        augmented = lights + lights_delta
        shad = shader(normals, augmented)
        shad_rep = shad.repeat(1,3,1,1)
        shadings[i] = shad_rep
    return shadings