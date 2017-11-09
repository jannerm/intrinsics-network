import math, os, subprocess, numpy as np, torch, pdb

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.vals = [0 for i in range(self.size)]
        self.avgs = [0 for i in range(self.size)]
        self.sums = [0 for i in range(self.size)]
        self.count = 0

    def update(self, vals, n=1):
        assert len(vals) == self.size
        self.vals = vals
        self.sums = map(sum, zip(self.sums, vals))
        self.count += n
        self.avgs = [float(s) / self.count for s in self.sums]

def initialize(args):
    mkdir(args.save_path)
    mkdir( os.path.join(args.save_path, 'raw_original') )
    mkdir( os.path.join(args.save_path, 'raw_trained') )

def mkdir(path):
    path = os.path.abspath(path)
    parent = os.path.join(path, '..')
    if not os.path.exists(parent):
        mkdir(parent)
    if not os.path.exists(path):
        subprocess.Popen(['mkdir', path])

def make_mask(img):
    mask = np.power(img,2).sum(0)[np.newaxis,:,:] > .01
    mask = np.tile(mask, (3,1,1))
    return mask

## expects np array
## [0,1] --> [-1,1]
def image_to_vector(img):
    mask = make_mask(img)
    img[mask] -= .5
    img[mask] *= 2.
    return img

## expects tensor
## [-1,1] --> [0,1]
def vector_to_image(vector):
    # mask = make_mask(img)
    dim = vector.dim()
    ## batch
    if dim == 4:
        mask = torch.pow(vector,2).sum(1) > .01
        mask = mask.repeat(1,3,1,1)
    elif dim == 3:
        mask = torch.pow(vector,2).sum(0) > .01
        mask = mask.repeat(3,1,1)
    else:
        raise RuntimeError 
    img = vector.clone()
    img[mask] /= 2.
    img[mask] += .5
    # img = (vector / 2.) + 5.
    return img

# CMAP = torch.Tensor( np.load('jet.npy') )

def colormap(img):
    steps = 512
    c = 3
    img = img.squeeze()

    ## single image
    if img.dim() < 3:
        img = img.unsqueeze(0)
    b, m, n = img.size()

    if img.eq(img[0][0][0]).all():
        indices = torch.Tensor(b,m,n).fill_( math.floor(steps / 2) )
    else:
        batch_min = img.min(1)[0].min(2)[0].repeat(1,m,n)
        img = img - batch_min
        batch_max = img.max(1)[0].max(2)[0].repeat(1,m,n)
        img = img / batch_max * (steps-1)
        indices = torch.floor(img)
        # print indices.min(1)[0].min(2)[0].squeeze(), indices.max(1)[0].max(2)[0].squeeze()
    indices = indices.view(-1)
    cimg = CMAP.index_select(0, indices.long())
    cimg = cimg.view(b, m, n, c)
    cimg = cimg.transpose(2,3).transpose(1,2)
    pdb.set_trace()
    return cimg

if __name__ == '__main__':
    def read_img(path):
        img = scipy.misc.imread(path)[:,:,:-1].transpose(2,0,1) / 255.
        img = torch.Tensor(img)[0].unsqueeze(0)
        return img


    # img = torch.randn(10,256,256)
    import scipy.misc
    d1 = 1 - read_img('test_images/356_depth.png')
    d2 = 2 - read_img('test_images/401_depth.png')
    d3 = 3 - read_img('test_images/5502_depth.png')
    d4 = torch.randn(1,256,256)
    batch = torch.cat((d1, d2, d3, d4), 0)
    print batch.size()
    cimg = colormap(batch)
    print cimg.size()
    for ind in range(batch.size(0)):
        original = batch[ind].numpy()
        print original.shape
        colored = cimg[ind].numpy().transpose(1,2,0)
        print colored.shape
        scipy.misc.imsave('test_images/original_' + str(ind) + '.png', original)
        scipy.misc.imsave('test_images/colored_' + str(ind) + '.png', colored)
#     # c = 3
# -- converts a gresycale image 
# -- to RGB based on current 
# -- style and step number;
# -- img should have 2 
# -- non-singleton dimensions  
# function colormap:convert(img)
#     local img = img:squeeze()
#     local m, n
#     m, n = img:size()[1], img:size()[2]
#     local c = self.channels
    
#     local indices
#     if torch.all(img:eq(img[1][1])) then
#         indices = torch.Tensor(m,n):fill(math.floor(self.steps / 2))
#     else
#         img = img - torch.min(img)
#         img = img / torch.max(img) * (self.steps-1) + 1
#         indices = torch.ceil(img)
#     end
#     indices = indices:reshape(indices:numel())

#     local cimg = self.currentMap:index(1, indices:long())
#     cimg = cimg:reshape(m, n, c)
#     cimg = cimg:transpose(3,2):transpose(1,2)
#     return cimg
# end


# def channel_lookup(intrinsic):
    # channels = {'reflectance': 3, 'normals'}