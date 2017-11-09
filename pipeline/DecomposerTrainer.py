import sys, math, numpy as np, pdb
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import pipeline

class DecomposerTrainer:
    def __init__(self, model, loader, lr, lights_mult):
        self.model = model
        self.loader = loader
        self.criterion = nn.MSELoss(size_average=True).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lights_mult = lights_mult

    def __epoch(self):
        self.model.train()
        losses = pipeline.AverageMeter(3)
        progress = tqdm( total=len(self.loader.dataset) )

        for ind, tensors in enumerate(self.loader):
            tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
            inp, mask, refl_targ, depth_targ, shape_targ, lights_targ = tensors
            self.optimizer.zero_grad()
            refl_pred, depth_pred, shape_pred, lights_pred = self.model.forward(inp, mask)
            refl_loss = self.criterion(refl_pred, refl_targ)
            depth_loss = self.criterion(depth_pred, depth_targ)
            shape_loss = self.criterion(shape_pred, shape_targ)
            lights_loss = self.criterion(lights_pred, lights_targ)
            loss = refl_loss + depth_loss + shape_loss + (lights_loss * self.lights_mult)
            loss.backward()
            self.optimizer.step()

            losses.update( [l.data[0] for l in [refl_loss, shape_loss, lights_loss] ])
            progress.update(self.loader.batch_size)
            progress.set_description( '%.5f | %.5f | %.5f | %.3f' % (refl_loss.data[0], depth_loss.data[0], shape_loss.data[0], lights_loss.data[0]) )
        print '<Train> Losses: ', losses.avgs
        return losses.avgs

    def train(self):
        # t = trange(iters)
        # for i in t:
        err = self.__epoch()
        # print 
            # t.set_description( str(err) )
        return err

if __name__ == '__main__':
    import sys
    sys.path.append('../')




