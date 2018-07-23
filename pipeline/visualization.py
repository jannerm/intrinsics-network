import os, math, torch, torch.nn as nn, torchvision.utils, numpy as np, scipy.misc, pdb
from torch.autograd import Variable
from tqdm import tqdm
import pipeline

# def make_grid(images, columns):
#     num_images = len(images)
#     rows = int(math.ceil(num_images / columns))
#     c, m, n = images[0].size()
#     grid = torch.ones(c, rows * m, columns * n)
#     for ind in range(len(images)):
#         row = ind / columns
#         col = ind % columns
#         grid[:, row*m:(row+1)*m, col*n:(col+1)*n] = images[ind]
#     return grid

def visualize_shader(model, loader, save_path, save_raw = False):
        model.train(mode=False)
        # progress = tqdm( total=len(loader.dataset) )
        inputs = []
        predictions = []
        targets = []
        for ind, tensors in enumerate(loader):

            inp = [ Variable( t.float().cuda(async=True) ) for t in tensors[:-1] ]
            targ = tensors[-1].float().cuda()
            pred = model.forward(*inp).data

            # pdb.set_trace()

            ## shape inputs
            inputs.extend([pipeline.vector_to_image(img.squeeze()) for img in inp[0].data.split(1)])
            ## shading predictions
            predictions.extend([img.repeat(1,3,1,1).squeeze() for img in pred.split(1)])
            ## shading targets
            targets.extend([img.repeat(3,1,1) for img in targ.split(1)])

        # print len(inputs), len(predictions), len(targets)
        images = [[inputs[i], predictions[i], targets[i]] for i in range(len(inputs))]
        images = [img for sublist in images for img in sublist]
        # pdb.set_trace()
        if save_raw:
            for ind, img in enumerate(images):
                img = img.cpu().numpy().transpose(1,2,0)
                img = np.clip(img, 0, 1)
                scipy.misc.imsave( os.path.join(save_path, 'shader_' + str(ind) + '.png'), img)
        else:
            grid = torchvision.utils.make_grid(images, nrow=3, padding=0).cpu().numpy().transpose(1,2,0)
            grid = np.clip(grid, 0, 1)
            scipy.misc.imsave(save_path, grid)
        # torchvision.utils.save_image(grid, os.path.join(save_path, 'shader.png'))
            return grid

def visualize_relit_shader(model, loader, save_path, params, save_raw = False):
        model.train(mode=False)
        # progress = tqdm( total=len(loader.dataset) )
        inputs = []
        predictions = []
        targets = []
        for ind, tensors in enumerate(loader):

            inp = [ Variable( t.float().cuda(async=True) ) for t in tensors[:-1] ]
            targ = tensors[-1].float()
            normals = inp[0]
            num_lights = normals.size(0)

            for param in params:
                # pdb.set_trace()
                lights = Variable( torch.Tensor(param).cuda().repeat(num_lights,1) )
                print(normals.size(), lights.size())
                # pdb.set_trace()
                pred = model.forward(normals, lights).data

                # pdb.set_trace()

                ## shape inputs
                inputs.extend([pipeline.vector_to_image(img.squeeze()) for img in inp[0].data.split(1)])
                ## shading predictions
                predictions.extend([img.repeat(1,3,1,1).squeeze() for img in pred.split(1)])
                ## shading targets
                targets.extend([img.repeat(3,1,1) for img in targ.split(1)])


        # print len(inputs), len(predictions), len(targets)
        images = [[inputs[i], predictions[i], targets[i]] for i in range(len(inputs))]
        images = [img for sublist in images for img in sublist]
        # pdb.set_trace()

        if save_raw:
            for ind, img in enumerate(images):
                img = img.cpu().numpy().transpose(1,2,0)
                img = np.clip(img, 0, 1)
                scipy.misc.imsave( os.path.join(save_path, 'relit_' + str(ind) + '.png'), img)
        else:
            grid = torchvision.utils.make_grid(images, nrow=3, padding=0).cpu().numpy().transpose(1,2,0)
            grid = np.clip(grid, 0, 1)
            scipy.misc.imsave(save_path, grid)
        # torchvision.utils.save_image(grid, os.path.join(save_path, 'shader.png'))
            return grid


def visualize_decomposer(model, loader, save_path, epoch, save_raw = False):
    model.train(mode=False)
    # progress = tqdm( total=len(loader.dataset) )
    # inputs = []
    # refl_preds = []
    # shape_preds = []
    # refl_targets = []
    # shape_targets = []

    criterion = nn.MSELoss(size_average=True).cuda()
    refl_loss = 0
    shape_loss = 0
    lights_loss = 0

    images = []

    for ind, tensors in enumerate(loader):
        tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
        inp, mask, refl_targ, depth_targ, shape_targ, lights_targ = tensors

        refl_pred, depth_pred, shape_pred, lights_pred = model.forward(inp, mask)

        refl_loss += criterion(refl_pred, refl_targ).data[0]
        shape_loss += criterion(shape_pred, shape_targ).data[0]
        lights_loss += criterion(lights_pred, lights_targ).data[0]

        shape_targ = pipeline.vector_to_image(shape_targ)
        shape_pred = pipeline.vector_to_image(shape_pred)

        depth_targ = depth_targ.unsqueeze(1).repeat(1,3,1,1)
        depth_pred = depth_pred.repeat(1,3,1,1)

        # pdb.set_trace()
        splits = []
        for tensor in [inp, refl_pred, refl_targ, depth_pred, depth_targ, shape_pred, shape_targ]:
            splits.append( [img.squeeze() for img in tensor.data.split(1)] )
        # pdb.set_trace()
        splits = [sublist[ind] for ind in range(len(splits[0])) for sublist in splits]
        images.extend(splits)
    

    refl_loss /= float(ind+1)
    shape_loss /= float(ind+1)
    lights_loss /= float(ind+1)

    # pdb.set_trace()
    # grid = torchvision.utils.make_grid(images, nrow=7).cpu().numpy().transpose(1,2,0)
    # grid = np.clip(grid, 0, 1)

    if epoch == 0:
        fullpath = os.path.join(save_path, 'original.png')
    else:
        fullpath = os.path.join(save_path, 'trained.png')

    # scipy.misc.imsave(fullpath, grid)

    losses = [refl_loss, shape_loss, lights_loss]
    print '<Val> Losses: ', losses
    # torchvision.utils.save_image(grid, os.path.join(save_path, 'shader.png'))
    return losses

def visualize_decomposer_full(model, loader, save_path):
    model.train(mode=False)
    # progress = tqdm( total=len(loader.dataset) )
    # inputs = []
    # refl_preds = []
    # shape_preds = []
    # refl_targets = []
    # shape_targets = []


    criterion = nn.MSELoss(size_average=True).cuda()
    refl_loss = 0
    shape_loss = 0
    lights_loss = 0

    images = []
    masks = []

    for ind, tensors in enumerate(loader):
        tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
        inp, mask, refl_targ, depth_targ, shape_targ, lights_targ = tensors

        refl_pred, depth_pred, shape_pred, lights_pred = model.forward(inp, mask)

        refl_loss += criterion(refl_pred, refl_targ).data[0]
        shape_loss += criterion(shape_pred, shape_targ).data[0]
        lights_loss += criterion(lights_pred, lights_targ).data[0]

        shad_targ = shad_targ.unsqueeze(1).repeat(1,3,1,1)

        shape_targ = pipeline.vector_to_image(shape_targ)
        shape_pred = pipeline.vector_to_image(shape_pred)

        lights_rendered_targ = render.vis_lights(lights_targ, verbose=False)
        lights_rendered_pred = render.vis_lights(lights_pred, verbose=False)

        depth_targ = depth_targ.unsqueeze(1).repeat(1,3,1,1)
        depth_pred = depth_pred.repeat(1,3,1,1)

        # pdb.set_trace()
        splits = []
        for tensor in [inp, refl_pred, refl_targ, depth_pred, depth_targ, shape_pred, shape_targ, lights_rendered_pred, lights_rendered_targ]:
            splits.append( [img.squeeze() for img in tensor.data.split(1)] )
            masks.append(mask)
        # pdb.set_trace()
        splits = [sublist[ind] for ind in range(len(splits[0])) for sublist in splits]
        images.extend(splits)

    labels = [  'recon_targ', 'refl_targ', 'depth_targ', 'depth_normals_targ', 'shape_targ', 'shad_targ', 'lights_targ',
            'recon_pred', 'refl_pred', 'depth_pred', 'depth_normals_pred', 'shape_pred', 'shad_pred', 'lights_pred']
              
    masks = [i.split(1) for i in masks]
    masks = [item.squeeze()[0].unsqueeze(0).data.cpu().numpy().transpose(1,2,0) for sublist in masks for item in sublist]

    save_raw(images, masks, labels, save_path)
            
    # pdb.set_trace()
    grid = torchvision.utils.make_grid(images, nrow=7).cpu().numpy().transpose(1,2,0)
    grid = np.clip(grid, 0, 1)
    scipy.misc.imsave( os.path.join(save_path, 'grid.png'), grid)
    # torchvision.utils.save_image(grid, os.path.join(save_path, 'shader.png'))
    return grid

def visualize_composer(model, loader, save_path, epoch, raw=False):
    model.train(mode=False)
    render = pipeline.Render()
    images = []

    criterion = nn.MSELoss(size_average=True).cuda()
    recon_loss = 0
    refl_loss = 0
    depth_loss = 0
    shape_loss = 0
    lights_loss = 0
    shad_loss = 0
    depth_normals_loss = 0

    masks = []

    for ind, tensors in enumerate(loader):
        tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
        inp, mask, refl_targ, depth_targ, shape_targ, lights_targ, shad_targ = tensors
        depth_normals_targ = pipeline.depth_to_normals(depth_targ.unsqueeze(1), mask=mask)
        # depth_normals_targ

        depth_targ = depth_targ.unsqueeze(1).repeat(1,3,1,1)
        shad_targ = shad_targ.unsqueeze(1).repeat(1,3,1,1)

        recon, refl_pred, depth_pred, shape_pred, lights_pred, shad_pred = model.forward(inp, mask)
        # relit = pipeline.relight(model.shader, shape_pred, lights_pred, 6)
        # relit_mean = relit.mean(0).squeeze()

        depth_normals_pred = pipeline.depth_to_normals(depth_pred, mask=mask)

        depth_pred = depth_pred.repeat(1,3,1,1)
        shad_pred = shad_pred.repeat(1,3,1,1)

        recon_loss += criterion(recon, inp).data[0]
        refl_loss += criterion(refl_pred, refl_targ).data[0]
        depth_loss += criterion(depth_pred, depth_targ).data[0]
        shape_loss += criterion(shape_pred, shape_targ).data[0]
        lights_loss += criterion(lights_pred, lights_targ).data[0]
        shad_loss += criterion(shad_pred, shad_targ).data[0]
        depth_normals_loss += lights_pred[:,1].sum() ##criterion(shape_pred, depth_normals_pred.detach()).data[0]

        lights_rendered_targ = render.vis_lights(lights_targ, verbose=False)
        lights_rendered_pred = render.vis_lights(lights_pred, verbose=False)
        # pdb.set_trace()

        shape_targ = pipeline.vector_to_image(shape_targ)
        shape_pred = pipeline.vector_to_image(shape_pred)

        depth_normals_targ = pipeline.vector_to_image(depth_normals_targ)
        depth_normals_pred = pipeline.vector_to_image(depth_normals_pred)


        splits = []
        # pdb.set_trace()
        for tensor in [ inp,    refl_targ,  depth_targ,     depth_normals_targ,     shape_targ,     shad_targ,  lights_rendered_targ, 
                        recon,  refl_pred,  depth_pred,     depth_normals_pred,     shape_pred,     shad_pred,  lights_rendered_pred ]:
                        # relit[0], relit[1], relit[2], relit[3], relit[4], relit[5], relit_mean]:
            splits.append( [img.squeeze() for img in tensor.data.split(1)] )

        masks.append(mask)

        # pdb.set_trace()
        # print shad_targ.size()
        # print shad_pred.size()
        # print [len(sublist) for sublist in splits]
        splits = [sublist[ind] for ind in range(len(splits[0])) for sublist in splits]
        images.extend(splits)

    labels = [  'recon_targ', 'refl_targ', 'depth_targ', 'depth_normals_targ', 'shape_targ', 'shad_targ', 'lights_targ',
                'recon_pred', 'refl_pred', 'depth_pred', 'depth_normals_pred', 'shape_pred', 'shad_pred', 'lights_pred']
              
    masks = [i.split(1) for i in masks]
    masks = [item.squeeze()[0].unsqueeze(0).data.cpu().numpy().transpose(1,2,0) for sublist in masks for item in sublist]

    if epoch == 0:
        raw_path = os.path.join(save_path, 'raw_original')
        grid_path = os.path.join(save_path, 'original.png')
    else:
        raw_path = os.path.join(save_path, 'raw_trained')
        grid_path = os.path.join(save_path, 'trained.png')

    if raw:
        save_raw(images, masks, labels, raw_path)

    recon_loss /= float(ind)
    refl_loss /= float(ind)
    depth_loss /= float(ind)
    shape_loss /= float(ind)
    lights_loss /= float(ind)
    shad_loss /= float(ind)
    depth_normals_loss /= float(ind)
    depth_normals_loss = depth_normals_loss.data[0]
    print 'depth_normals_loss: ', depth_normals_loss

    # pdb.set_trace()
    grid = torchvision.utils.make_grid(images, nrow=7).cpu().numpy().transpose(1,2,0)
    grid = np.clip(grid, 0, 1)
    # fullpath = os.path.join(save_path, str(epoch) + '.png')
    scipy.misc.imsave(grid_path, grid)
    # torchvision.utils.save_image(grid, os.path.join(save_path, 'shader.png'))
    return [recon_loss, refl_loss, depth_loss, shape_loss, lights_loss, shad_loss, depth_normals_loss]

def visualize_composer_alt(model, loader, save_path, epoch, raw=False):
    model.train(mode=False)
    render = pipeline.Render()
    images = []

    criterion = nn.MSELoss(size_average=True).cuda()
    recon_loss = 0
    refl_loss = 0
    depth_loss = 0
    shape_loss = 0
    lights_loss = 0
    shad_loss = 0
    depth_normals_loss = 0

    masks = []

    for ind, tensors in enumerate(loader):
        tensors = [Variable(t.float().cuda(async=True)) for t in tensors]
        inp, mask, refl_targ, depth_targ, shape_targ, lights_targ, shad_targ = tensors
        depth_normals_targ = pipeline.depth_to_normals(depth_targ.unsqueeze(1), mask=mask)
        # depth_normals_targ

        depth_targ = depth_targ.unsqueeze(1).repeat(1,3,1,1)
        shad_targ = shad_targ.unsqueeze(1).repeat(1,3,1,1)

        recon, refl_pred, depth_pred, shape_pred, lights_pred, shad_pred = model.forward(inp, mask)
        
        ####
        shad_pred = model.shader(shape_pred, lights_pred)
        print 'shad_pred: ', shad_pred.size()
        # shad_pred = shad_pred.repeat(1,3,1,1)

        # relit = pipeline.relight(model.shader, shape_pred, lights_pred, 6)
        # relit_mean = relit.mean(0).squeeze()

        depth_normals_pred = pipeline.depth_to_normals(depth_pred, mask=mask)

        depth_pred = depth_pred.repeat(1,3,1,1)
        shad_pred = shad_pred.repeat(1,3,1,1)

        recon_loss += criterion(recon, inp).data[0]
        refl_loss += criterion(refl_pred, refl_targ).data[0]
        depth_loss += criterion(depth_pred, depth_targ).data[0]
        shape_loss += criterion(shape_pred, shape_targ).data[0]
        lights_loss += criterion(lights_pred, lights_targ).data[0]
        shad_loss += criterion(shad_pred, shad_targ).data[0]
        depth_normals_loss += criterion(shape_pred, depth_normals_pred.detach()).data[0]

        lights_rendered_targ = render.vis_lights(lights_targ, verbose=False)
        lights_rendered_pred = render.vis_lights(lights_pred, verbose=False)
        # pdb.set_trace()

        shape_targ = pipeline.vector_to_image(shape_targ)
        shape_pred = pipeline.vector_to_image(shape_pred)

        depth_normals_targ = pipeline.vector_to_image(depth_normals_targ)
        depth_normals_pred = pipeline.vector_to_image(depth_normals_pred)


        splits = []
        # pdb.set_trace()
        for tensor in [ inp,    refl_targ,  depth_targ,     depth_normals_targ,     shape_targ,     shad_targ,  lights_rendered_targ, 
                        recon,  refl_pred,  depth_pred,     depth_normals_pred,     shape_pred,     shad_pred,  lights_rendered_pred ]:
                        # relit[0], relit[1], relit[2], relit[3], relit[4], relit[5], relit_mean]:
            splits.append( [img.squeeze() for img in tensor.data.split(1)] )

        masks.append(mask)

        # pdb.set_trace()
        # print shad_targ.size()
        # print shad_pred.size()
        # print [len(sublist) for sublist in splits]
        splits = [sublist[ind] for ind in range(len(splits[0])) for sublist in splits]
        images.extend(splits)

    labels = [  'recon_targ', 'refl_targ', 'depth_targ', 'depth_normals_targ', 'shape_targ', 'shad_targ', 'lights_targ',
                'recon_pred', 'refl_pred', 'depth_pred', 'depth_normals_pred', 'shape_pred', 'shad_pred', 'lights_pred']
              
    masks = [i.split(1) for i in masks]
    masks = [item.squeeze()[0].unsqueeze(0).data.cpu().numpy().transpose(1,2,0) for sublist in masks for item in sublist]

    if epoch == 0:
        raw_path = os.path.join(save_path, 'raw_original')
    else:
        raw_path = os.path.join(save_path, 'raw_trained')

    if raw:
        save_raw(images, masks, labels, raw_path)

    recon_loss /= float(ind)
    refl_loss /= float(ind)
    depth_loss /= float(ind)
    shape_loss /= float(ind)
    lights_loss /= float(ind)
    shad_loss /= float(ind)
    depth_normals_loss /= float(ind)

    # pdb.set_trace()
    grid = torchvision.utils.make_grid(images, nrow=7).cpu().numpy().transpose(1,2,0)
    grid = np.clip(grid, 0, 1)
    fullpath = os.path.join(save_path, str(epoch) + '.png')
    scipy.misc.imsave(fullpath, grid)
    # torchvision.utils.save_image(grid, os.path.join(save_path, 'shader.png'))
    return [recon_loss, refl_loss, depth_loss, shape_loss, lights_loss, shad_loss, depth_normals_loss]

def save_raw(images, masks, labels, save_path):
    for ind, img in enumerate(images):
        img_num = ind / len(labels)
        lab = labels[ind % len(labels)]

        img = img.cpu().numpy().transpose(1,2,0)
        img = np.clip(img, 0, 1)

        # if mask != None:
        if 'lights' in lab:
            mask = (img.sum(-1)>0).astype(float)
            mask = mask[:,:,np.newaxis]
        else:
            mask = masks[img_num]

        alpha = np.concatenate((img,mask),-1)
        # else:
        #     alpha = img
        # pdb.set_trace()
        fullpath = os.path.join(save_path, str(img_num) + '_' + lab + '.png')
        scipy.misc.imsave(fullpath, alpha)








