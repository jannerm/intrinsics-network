import sys, math, numpy as np, pdb
from tqdm import tqdm, trange
import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import pipeline

class ComposerTrainer:
    def __init__(self, model, loader, lr, lights_mult, un_mult, lab_mult, transfer, relight_fn=None, relight_sigma=1, relight_mult=0.1, normals_fn=None, correspondence_mult=1e-2, style_fn=None, style_mult=1e-9, epoch_size=15000, iters=1):
        self.model = model
        self.loader = loader
        self.criterion = nn.MSELoss(size_average=True).cuda()
        self.iters = iters

        # parameters = [  {'params': self.model.decomposer.encoder.parameters(), 'lr': lr}, 
        #                 {'params': self.model.decomposer.decoder_normals.parameters(), 'lr': lr}  ]
        parameters = []
        print 'Parameters: '
        if 'reflectance' in transfer:
            print '    |-- reflectance'
            parameters.append( {'params': self.model.decomposer.decoder_reflectance.parameters(), 'lr': lr} )
        if 'normals' in transfer:
            print '    |-- normals'
            parameters.append( {'params': self.model.decomposer.decoder_normals.parameters(), 'lr': lr} )
        if 'lights' in transfer:
            print '    |-- lights'
            parameters.append( {'params': self.model.decomposer.decoder_lights.parameters(), 'lr': lr} )
        if 'shader' in transfer:
            print '    |-- shader'
            parameters.append( {'params': self.model.shader.parameters(), 'lr': lr} )
        # parameters = [  {'params': self.model.decomposer.decoder_normals.parameters(), 'lr': lr} ]
                        # {'params': self.model.decomposer.decoder_lights.parameters(), 'lr': lr*10},
                        # {'params': self.model.shader.parameters(), 'lr': lr}  ]
        # pdb.set_trace()
        self.optimizer = optim.Adam(parameters, lr=lr)
        self.lights_mult = lights_mult
        self.un_mult = un_mult
        self.lab_mult = lab_mult
        self.epoch_size = epoch_size
        self.style_fn = style_fn
        self.style_mult = style_mult
        self.normals_fn = normals_fn
        self.correspondence_mult = correspondence_mult
        self.relight_fn = relight_fn
        self.relight_sigma = relight_sigma
        self.relight_mult = relight_mult

    def __epoch(self):
        self.model.train()
        losses = pipeline.AverageMeter(9)
        # pdb.set_trace()
        # progress = tqdm( total=len(self.loader.dataset), ncols=0 )

        for ind, (unlabeled, labeled) in enumerate(self.loader):
            unlabeled = [Variable(t.float().cuda(async=True)) for t in unlabeled]
            labeled = [Variable(t.float().cuda(async=True)) for t in labeled]
            # print [i.size() for i in unlabeled]
            # print [i.size() for i in labeled]
            un_inp, un_mask = unlabeled
            lab_inp, lab_mask, lab_refl_targ, lab_depth_targ, lab_shape_targ, lab_lights_targ, lab_shad_targ = labeled

            self.optimizer.zero_grad()
            un_recon,  _, un_depth_pred, un_shape_pred,  un_lights_pred,  un_shad_pred = self.model.forward(un_inp, un_mask)
            lab_recon, lab_refl_pred, lab_depth_pred, lab_shape_pred, lab_lights_pred, lab_shad_pred = self.model.forward(lab_inp, lab_mask)
            
            # print un_inp.size(), un_recon.size()
            # print type(un_inp), type(un_recon)
            un_loss = self.criterion(un_recon, un_inp)

            refl_loss = self.criterion(lab_refl_pred, lab_refl_targ)
            depth_loss = self.criterion(lab_depth_pred, lab_depth_targ)
            shape_loss = self.criterion(lab_shape_pred, lab_shape_targ)
            lights_loss = self.criterion(lab_lights_pred, lab_lights_targ)
            shad_loss = self.criterion(lab_shad_pred, lab_shad_targ)
            lab_loss = refl_loss + shape_loss + (lights_loss * self.lights_mult) + shad_loss

            loss = (self.un_mult * un_loss) + (self.lab_mult * lab_loss)

            if self.style_fn:
                style_loss = self.style_fn(un_shape_pred, lab_shape_targ)
                # print 'style loss: ', style_loss
                loss += (self.style_mult * style_loss)
            else:
                style_loss = Variable( torch.zeros(1) )

            if self.normals_fn:
                approx_normals = self.normals_fn(un_depth_pred, mask=un_mask)
                depth_normals_loss = self.criterion(un_shape_pred, approx_normals.detach())
                loss += (self.correspondence_mult * depth_normals_loss)
            else:
                depth_normals_loss = Variable( torch.zeros(1) )

            if self.relight_fn:
                relit = self.relight_fn(self.model.shader, un_shape_pred, un_lights_pred, 2, sigma=self.relight_sigma)
                relit_mean = relit.mean(0).squeeze()[:,0]
                # pdb.set_trace()
                # print 'mean: ', type(un_shape_pred), type(relit_mean)
                relight_loss = self.criterion(un_shad_pred, relit_mean.detach().cuda() )
                loss += (self.relight_mult * relight_loss)
            else:
                relight_loss = Variable( torch.zeros(1) )

            loss.backward()
            self.optimizer.step()

            loss_data = [l.data[0] for l in [un_loss, refl_loss, depth_loss, shape_loss, lights_loss, shad_loss, depth_normals_loss, relight_loss, style_loss] ]
            losses.update(loss_data)
            # progress.update(self.loader.batch_size)
            if losses.count * self.loader.batch_size > self.epoch_size:
                break
            # print loss_data
            # progress.set_description( '%.4f | R %.4f | D %.3f | N %.3f | L %.3f | S %.4f | C %.3f | G %.3f | S %.3f' % tuple(loss_data) )
        print 'Losses: ', losses.avgs
        return losses.avgs

    def train(self):
        # t = trange(iters)
        # for i in t:
        for i in range(self.iters):
            errors = self.__epoch()
        # print 
            # t.set_description( str(err) )
        return errors

if __name__ == '__main__':
    import sys
    sys.path.append('../')




