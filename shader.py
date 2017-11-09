#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import sys, os, argparse, torch
import models, pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',  type=str,   default='dataset/output/',
        help='base path for datasets')
parser.add_argument('--train_sets', type=str,   default='motorbike_train,airplane_train,bottle_train', 
        help='folders within data_path to draw from during training')
parser.add_argument('--val_sets',   type=str,   default='motorbike_val,airplane_val,bottle_val',
        help='folders within data_path to draw from during validation')
parser.add_argument('--intrinsics', type=list,  default=['normals', 'lights', 'shading'],
        help='intrinsic images to load from the train and val sets')
parser.add_argument('--save_path',  type=str,   default='saved/shader/',
        help='save folder for model and visualizations')
parser.add_argument('--lr',         type=float, default=0.01,
        help='learning rate')
parser.add_argument('--num_epochs', type=int,   default=500,
        help='number of training epochs')
parser.add_argument('--num_train',  type=int,   default=100,
        help='number of training images per object category')
parser.add_argument('--num_val',    type=int,   default=100,
        help='number of validation images per object category')
args = parser.parse_args()

pipeline.initialize(args)

## load model : shape x lighting --> shading
shader = models.Shader().cuda()

## get a data loader for train and val sets
train_set = pipeline.IntrinsicDataset(args.data_path, args.train_sets, args.intrinsics, size_per_dataset=args.num_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, num_workers=4, shuffle=True)

val_set = pipeline.IntrinsicDataset(args.data_path, args.val_sets, args.intrinsics, size_per_dataset=10)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, num_workers=4, shuffle=False)

trainer = pipeline.Trainer(shader, train_loader, args.lr)

for epoch in range(args.num_epochs):
    print '<Main> Epoch {}'.format(epoch)

    ## save model and state
    torch.save( shader, open(os.path.join(args.save_path, 'model.t7'), 'w') )
    torch.save( shader.state_dict(), open(os.path.join(args.save_path, 'state.pth'), 'w') )
    
    ## visualize predictions of shader
    save_path =  os.path.join(args.save_path, str(epoch) + '.png')
    pipeline.visualize_shader(shader, val_loader, save_path )
    
    ## one sweep through the dataset
    trainer.train()
