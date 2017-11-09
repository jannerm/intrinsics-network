import sys
sys.path.append('/om/user/janner/anaconda/lib/python3.4/site-packages/')
sys.path.append('/om/user/janner/mit/urop/picture/centos/')
sys.path.append('/om/user/janner/mit/urop/picture/centos/script_directory/modules/')
sys.path.append('/om/user/janner/mit/urop/intrinsic/dataset/')
import os, math, argparse, scipy.io, scipy.stats, time, random, subprocess
import numpy as np
from BlenderShapenet import BlenderRender, ShapenetRender, IntrinsicRender

import io
from contextlib import redirect_stdout

print('starting!')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=False)
parser.add_argument('--working', default='staging')
parser.add_argument('--shapenet', default='/om/data/public/ilkery/ShapeNetCore.v1/')
parser.add_argument('--output', default='output/shadows/')
parser.add_argument('--category', default='')
parser.add_argument('--x_res', default=256)
parser.add_argument('--y_res', default=256)
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--finish', default=10, type=int)
parser.add_argument('--pos_low', default=[0, 0, 0])
parser.add_argument('--pos_high', default=[0, 0, 0])
parser.add_argument('--scale_low', default=6)
parser.add_argument('--scale_high', default=9)
## y (middle coordinate) used to be 0
parser.add_argument('--theta_low', default=[60, -45, 0])
parser.add_argument('--theta_high', default=[120, 45, 360])
# parser.add_argument('--lights_energy_low', default=1)
# parser.add_argument('--lights_energy_high', default=5)
# parser.add_argument('--lights_pos_low', default=[-3.5, -2.5, 1])
# parser.add_argument('--lights_pos_high', default=[3.5, -2.5, 3])
# parser.add_argument('--array_path', default='arrays/shader2.npy')
parser.add_argument('--repeat', default=10, type=int)

cmd = sys.argv
print(cmd[cmd.index('--')+1:])
args = parser.parse_args(cmd[cmd.index('--')+1:])

categories = args.category.split(',')
names = [categories[i] + '_' + str(i) for i in range(len(categories))]
shapenet_ids = {'car': '02958343', 'chair': '03001627', 'airplane': '02691156', 'sofa': '04256520', 'boat': '04530566', 'motorbike': '03790512', 'flowerpot': '03991062', 'piano': '03928116', 'tower': '04460130', 'bottle': '02876657', 'train': '04468005'}
positions_low = [[0,0,-.5],[0,-2,-.5]]
positions_high = [[-1,5,-.5],[1,-3,-.5]]
scales_low = [3, 2]
scales_high = [3, 2]
# if args.category != 'random': 
#     category = categories[args.category] 
# else:
#     category = args.category

staging = args.working + '/' + str(random.random()) + '/'

Blender = BlenderRender(args.gpu, spotlight=True)
Shapenet = ShapenetRender(args.shapenet, staging, args.output, create=True)
Intrinsic = IntrinsicRender(args.x_res, args.y_res)

lights = np.load('arrays/spot1.npy')

Blender.sphere([0,0,0], 2.5, label = 'sphere')

count = args.start

while count < args.finish:
    for ind, cat in enumerate(categories):
        name = names[ind]
        Shapenet.load(shapenet_ids[cat], name )
        Blender.duplicate(name, name + '_shading')
        Blender.duplicate(name, name + '_normals')

    for rep in range(args.repeat):
        
        for ind, cat in enumerate(categories):
            name = names[ind]
            Blender.translate( [name, name + '_shading', name + '_normals'], Blender.random(positions_low[ind], positions_high[ind]) )
            Blender.resize( [name, name + '_shading', name + '_normals'], Blender.random(scales_low[ind], scales_high[ind]) )
            Blender.rotate( [name, name + '_shading', name + '_normals'], Blender.random(args.theta_low, args.theta_high) )

        # phi = np.random.uniform(low=eps, high=math.pi-eps)
        # theta = np.random.uniform(low=eps, high=math.pi-eps)
        # Blender.light(rho, phi, theta)

        # energy = lights[count][0]
        # pos = lights[count][1:]
        # Blender.light(  energy, pos )

        Blender.spotlight(*lights[count])

        for mode in ['normals', 'shading', 'mask', 'specular', 'lights']:
            filename = str(count) + '_' + mode 
            extension = 'exr' if mode == 'depth_hires' else 'png'
            for ind, cat in enumerate(categories):
                name = names[ind]
                Intrinsic.changeMode(mode, name=name)
            Blender.write(args.output, filename, extension=extension)

        count += 1

    for ind, cat in enumerate(categories):
        name = names[ind]
        Blender.delete(lambda x: x.name in [name, name + '_shading', name + '_normals'] )

#### Sphere
# Shapenet.load(category)
Blender.sphere([0,0,0], 2.5)
Blender.duplicate('shape', 'shape_shading')
Blender.duplicate('shape', 'shape_normals')
    
# Blender.translate( ['shape', 'shape_shading', 'shape_normals'], Blender.random(args.pos_low, args.pos_high) )
# Blender.resize( ['shape', 'shape_shading', 'shape_normals'], Blender.random(args.scale_low, args.scale_high) )
# Blender.rotate( ['shape', 'shape_shading', 'shape_normals'], Blender.random(args.theta_low, args.theta_high) )

energy = lights[count][0]
pos = lights[count][1:]
Blender.light(  energy, pos )

for mode in ['composite', 'albedo', 'depth', 'depth_hires', 'normals', 'shading', 'mask', 'specular']:
    filename = 'sphere_' + mode 
    extension = 'exr' if mode == 'depth_hires' else 'png'
    Intrinsic.changeMode(mode)
    Blender.write(args.output, filename, extension=extension)

    # count += 1

Blender.delete(lambda x: x.name in ['shape'] )
