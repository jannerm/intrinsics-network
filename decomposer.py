#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import sys, os, argparse, torch, pdb
import models, pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',      type=str,   default='dataset/output/',
        help='base path for datasets')
parser.add_argument('--train_sets',     type=str,   default='motorbike_train,bottle_train',
        help='folders within data_path to draw from during training')
parser.add_argument('--val_sets',       type=str,   default='motorbike_val,bottle_val',
        help='folders within data_path to draw from during validation')
parser.add_argument('--intrinsics',     type=list,  default=['input', 'mask', 'albedo', 'depth', 'normals', 'lights'],
        help='intrinsic images to load from the train and val sets')
parser.add_argument('--save_path',      type=str,   default='components/test_logger/',
        help='save folder for model, plots, and visualizations')
parser.add_argument('--lr',             type=float, default=0.01,
        help='learning rate')
parser.add_argument('--num_epochs', type=int,   default=500,
        help='number of training epochs')
parser.add_argument('--lights_mult',    type=float, default=0.01,
        help='multiplier on the lights loss')
parser.add_argument('--array',          type=str,   default='shader',
        help='array with lighting parameters')
parser.add_argument('--num_train',  type=int,   default=100,
        help='number of training images per object category')
parser.add_argument('--num_val',    type=int,   default=100,
        help='number of validation images per object category')
parser.add_argument('--loaders',    type=int,   default=4,
        help='number of parallel data loading processes')
parser.add_argument('--batch_size',    type=int,   default=32)
args = parser.parse_args()

pipeline.initialize(args)

## load model : image --> reflectance x normals x depth x lighting 
model = models.Decomposer().cuda()

## get a data loader for train and val sets
train_set = pipeline.IntrinsicDataset(args.data_path, args.train_sets, args.intrinsics, array=args.array, size_per_dataset=args.num_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, num_workers=args.loaders, shuffle=True)

val_set = pipeline.IntrinsicDataset(args.data_path, args.val_sets, args.intrinsics, array=args.array, size_per_dataset=args.num_val)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.loaders, shuffle=False)

## plots errors for the intrinsic image predictions
logger = pipeline.Logger(['refl', 'shape', 'lights'], args.save_path)

trainer = pipeline.DecomposerTrainer(model, train_loader, args.lr, args.lights_mult)

for epoch in range(args.num_epochs):
    print '<Main> Epoch {}'.format(epoch)

    ## save model state
    state = model.state_dict()
    torch.save( state, open(os.path.join(args.save_path, 'state.t7'), 'w') )
    
    ## get losses and save visualization on val images
    val_losses = pipeline.visualize_decomposer(model, val_loader, args.save_path, epoch)
    
    ## one sweep through the dataset
    train_losses = trainer.train()

    ## save plots of the errors
    logger.update(train_losses, val_losses)








