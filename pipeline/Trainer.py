import sys, math, numpy as np, pdb
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import pipeline

class Trainer:
    def __init__(self, model, loader, lr):
        self.model = model
        self.loader = loader
        self.criterion = nn.MSELoss(size_average=True).cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def __epoch(self):
        self.model.train()
        losses = pipeline.AverageMeter(1)
        progress = tqdm( total=len(self.loader.dataset) )

        for ind, tensors in enumerate(self.loader):

            inp = [ Variable( t.float().cuda(async=True) ) for t in tensors[:-1] ]
            targ = Variable( tensors[-1].float().cuda(async=True) )

            self.optimizer.zero_grad()
            out = self.model.forward(*inp)
            loss = self.criterion(out, targ)
            loss.backward()
            self.optimizer.step()

            losses.update( [loss.data[0]] )
            progress.update(self.loader.batch_size)
            progress.set_description( str(loss.data[0]) )
        return losses.avgs

    def train(self):
        # t = trange(iters)
        # for i in t:
        err = self.__epoch()
            # t.set_description( str(err) )
        return self.model





