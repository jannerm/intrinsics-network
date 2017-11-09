import torch, torch.nn.functional as F, pdb
from torch.autograd import Variable
from torch.nn.parameter import Parameter


_f1m = Parameter( torch.Tensor([ [-1,    -2,     -1],
                                [0,     0,      0],
                                [1,     2,      1]]) / 8 )

_f2m = Parameter( torch.Tensor([ [-1,     0,      1],
                                [-2,     0,      2],
                                [-1,     0,      1]]) / 8 )

_f1m = _f1m.unsqueeze(0).unsqueeze(0).cuda()
_f2m = _f2m.unsqueeze(0).unsqueeze(0).cuda()

_bias = Parameter( torch.zeros(1) ).cuda()
_stride = (1,1)
_padding = (1,1)

# print _f1m, _f2m

'''
expects depth to be < B x 1 x M x N >
'''
def depth_to_normals(depth, mask=None):
    # pdb.set_trace()
    depth = (1 - depth) * 255
    n1 = F.conv2d(depth, _f1m, _bias, _stride, _padding)
    n2 = F.conv2d(depth, _f2m, _bias, _stride, _padding)

    N3 = 1/torch.sqrt( torch.pow(n1, 2) + torch.pow(n2, 2) + 1 )

    N1 = n1 * N3;
    N2 = n2 * N3;

    N = torch.cat( (N1, N2, N3), 1 )

    N[mask.ne(1)] = 0

    return N




if __name__ == '__main__':
    import scipy.misc

    def read_img(path):
        img = scipy.misc.imread(path)[:,:,:-1].transpose(2,0,1) / 255.
        return img
    img = read_img('test_images/356_depth.png')[0]
    img = torch.Tensor(img).unsqueeze(0).unsqueeze(0)
    img = Variable(img)
    img2 = read_img('test_images/5502_depth.png')[0]
    img2 = torch.Tensor(img2).unsqueeze(0).unsqueeze(0)
    img2 = Variable(img2)
    img = torch.cat( (img, img2), 0 )
    print img.size()

    normals = depth_to_normals(img)

    pdb.set_trace()
    loss = normals.sum()
    loss.backward()
    
    print normals, normals
    norm1 = normals[0]
    norm2 = normals[1]
    norm1 = norm1.data.numpy()
    norm1 = norm1.transpose(1,2,0)
    print norm1.shape
    scipy.misc.imsave('test_images/356_approx_normals.png', norm1)

    norm2 = norm2.data.numpy()
    norm2 = norm2.transpose(1,2,0)
    print norm2.shape
    scipy.misc.imsave('test_images/5502_approx_normals.png', norm2)
    # inp = Variable( torch.randn(3,5,5) )
    # weight = Parameter( torch.randn(3,3) )







