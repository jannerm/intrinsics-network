#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import sys, os, argparse, torch, pdb
sys.path.append('/om/user/janner/mit/urop/intrinsic/pytorch/')
import models, pipeline, style

parser = argparse.ArgumentParser()
parser.add_argument('--decomposer',         type=str,   default='saved/vector_depth_state_decomp_0.001lr_0.1lights/state.t7',
        help='decomposer network state file')
parser.add_argument('--shader',             type=str,   default='saved/vector_shader_0.01/model.t7',
        help='shader network file')
parser.add_argument('--data_path',          type=str,   default='../dataset/output/',
        help='base folder of datasets')
parser.add_argument('--unlabeled',          type=str,   default='car_train',
        help='unlabeled dataset(s), separated by commas, to use during training')
parser.add_argument('--labeled',            type=str,   default='motorbike_train,airplane_train,bottle_train',
        help='labeled dataset(s), separated by commas, to use during training')
parser.add_argument('--val_sets',           type=str,   default='car_val,motorbike_val',
        help='validation dataset(s), separated by commas')
parser.add_argument('--val_intrinsics',     type=list,  default=['input', 'mask', 'albedo', 'depth', 'normals', 'lights', 'shading'],
        help='intrinsic images to load for validation sets')
parser.add_argument('--save_path',          type=str,   default='logs/composer/',
        help='save path of model, visualizations, and error plots')
parser.add_argument('--labeled_array',      type=str,   default='shader',
        help='array of lighting parameters for unlabeled data')
parser.add_argument('--unlabeled_array',    type=str,   default='shader',
        help='array of lighting parameters for labeled data')
parser.add_argument('--lr',                 type=float, default=0.01,
        help='learning rate')
parser.add_argument('--num_val',            type=int,   default=10,
        help='number of validation images')
parser.add_argument('--lights_mult',        type=float, default=1,
        help='multiplier on lights loss')
parser.add_argument('--un_mult',            type=float, default=1,
        help='multipler on reconstruction loss')
parser.add_argument('--lab_mult',           type=float, default=1,
        help='multipler on labeled intrinsic images loss')
parser.add_argument('--loader_threads',     type=float, default=4,
        help='number of parallel data-loading threads')
parser.add_argument('--save_model',         type=bool,  default=True,
        help='whether to save model or not')
parser.add_argument('--transfer',           type=str,   default='10-normals,shader_10-shader',
        help='specifies which parameters are updated and for how many epochs')
parser.add_argument('--iters',              type=int,   default=1,
        help='number of expected times an image is reused during an epoch')
parser.add_argument('--set_size',           type=int,   default=10000,
        help='number of images per training dataset')
parser.add_argument('--val_offset',         type=int,   default=10,
        help='number of images per validation set which are used in visualizations')
parser.add_argument('--epoch_size',         type=int,   default=3200,
        help='number of images in an epoch')
parser.add_argument('--num_epochs',         type=int,   default=300)
parser.add_argument('--batch_size',         type=float, default=32)
args = parser.parse_args()

pipeline.initialize(args)

## decomposer : image --> reflectance, normals, lighting
decomposer = models.Decomposer().cuda()
decomposer.load_state_dict( torch.load(args.decomposer) )
## shader : normals, lighting --> shading
shader = torch.load(args.shader)
## composer : image --> reflectance, normals, lighting, shading --> image
model = models.Composer(decomposer, shader).cuda()

## data loader for train set, which includes half labeled and half unlabeled data
train_set = pipeline.ComposerDataset(args.data_path, args.unlabeled, args.labeled, unlabeled_array=args.unlabeled_array, labeled_array=args.labeled_array, size_per_dataset=args.set_size)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size / 2, num_workers=args.loader_threads, shuffle=True)

## data loader for val set, which is completely labeled
val_set = pipeline.IntrinsicDataset(args.data_path, args.val_sets, args.val_intrinsics, inds=range(0,args.num_val*args.val_offset,args.val_offset), array=args.unlabeled_array, size_per_dataset=args.set_size)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.loader_threads, shuffle=False)

## print out error plots after every epoch for every prediction
logger = pipeline.Logger(['recon', 'refl', 'depth', 'shape', 'lights', 'shading'], args.save_path)
param_updater = pipeline.ParamUpdater(args.transfer)

for epoch in range(args.num_epochs):
    print '<Main> Epoch {}'.format(epoch)

    if param_updater.check(epoch):
        ## update which parameters are updated
        transfer = param_updater.refresh(epoch)
        print 'Updating params: ', epoch, transfer
        ## get a new trainer with different learnable parameters
        trainer = pipeline.ComposerTrainer( model, train_loader, args.lr, args.lights_mult, args.un_mult, args.lab_mult, transfer, 
                                    epoch_size=args.epoch_size, iters=args.iters)

    if args.save_model:
        state = model.state_dict()
        torch.save( state, open(os.path.join(args.save_path, 'state.t7'), 'w') )

    
    ## visualize intrinisc image predictions and reconstructions of the val set
    val_losses = pipeline.visualize_composer(model, val_loader, args.save_path, epoch)
    
    ## one sweep through the args.epoch_size images
    train_losses = trainer.train()

    ## save plots of the errors
    logger.update(train_losses, val_losses)








