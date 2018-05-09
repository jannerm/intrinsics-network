import sys
import os, math, argparse, scipy.io, scipy.stats, time, random, subprocess
import numpy as np
from BlenderShapenet import BlenderRender, ShapenetRender, IntrinsicRender

import io
from contextlib import redirect_stdout

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=False)
parser.add_argument('--lights_path', default='')
parser.add_argument('--save_path', default='test/')
parser.add_argument('--x_res', default=256)
parser.add_argument('--y_res', default=256)
cmd = sys.argv
print(cmd[cmd.index('--')+1:])
args = parser.parse_args(cmd[cmd.index('--')+1:])

Blender = BlenderRender(args.gpu)
Intrinsic = IntrinsicRender(args.x_res, args.y_res)

lights = np.load(args.lights_path)

Blender.sphere([0,0,0], 2.5, label = 'sphere')
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


