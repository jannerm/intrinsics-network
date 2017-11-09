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
parser.add_argument('--lights_path', default='')
parser.add_argument('--save_path', default='test/')
parser.add_argument('--x_res', default=256)
parser.add_argument('--y_res', default=256)
cmd = sys.argv
print(cmd[cmd.index('--')+1:])
args = parser.parse_args(cmd[cmd.index('--')+1:])

# categories = {'car': '02958343', 'chair': '03001627', 'airplane': '02691156', 'sofa': '04256520', 'boat': '04530566'}
# if args.category != 'random': 
    # category = categories[args.category] 
# else:
    # category = args.category

# staging = args.working + '/' + str(random.random()) + '/'

Blender = BlenderRender(args.gpu)
# Shapenet = ShapenetRender(args.shapenet, staging, args.output, create=True)
Intrinsic = IntrinsicRender(args.x_res, args.y_res)

lights = np.load(args.lights_path)

Blender.sphere([0,0,0], 2.5, label = 'sphere')
# Blender.duplicate('shape', 'shape_shading')
# Blender.duplicate('shape', 'shape_normals')
Intrinsic.changeMode('lights')
print(lights)
print(lights.shape)
num_lights = lights.shape[0]

for ind in range(num_lights):

    energy = lights[ind][0]
    pos = lights[ind][1:]
    Blender.light(  energy, pos )

    filename = str(ind)
    Blender.write(args.save_path, filename, extension='png')


