import sys, argparse

################################
############ Setup #############
################################

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=False)
parser.add_argument('--staging', default='staging')
parser.add_argument('--shapenet', default='/om/data/public/ilkery/ShapeNetCore.v1/')
parser.add_argument('--output', default='output/normals/')
parser.add_argument('--category', default='random')
parser.add_argument('--x_res', default=256)
parser.add_argument('--y_res', default=256)
parser.add_argument('--start', default=0, type=int)
parser.add_argument('--finish', default=10, type=int)
parser.add_argument('--array_path', default='arrays/shader.npy')
parser.add_argument('--include', type=str)
parser.add_argument('--repeat', default=10, type=int)

## ignore the blender arguments
cmd = sys.argv
args = cmd[cmd.index('--')+1:]
args = parser.parse_args(args)

## blender doesn't by default include, the working 
## directory, so add the repo folder manually
sys.path.append(args.include)

## grad config parameters from repo folder
import config, dataset.utils as utils

## add any other libraries not by default in blender's python
## (e.g., scipy)
sys.path.append(config.include)

## import everything else
import os, math, argparse, scipy.io, scipy.stats, time, random, subprocess, pdb
import numpy as np

## import repo modules
from dataset import BlenderRender, ShapeNetRender, IntrinsicRender, PrimitiveRender
from dataset import utils
# from dataset.BlenderShapenet import BlenderRender, ShapenetRender, IntrinsicRender
# from dataset.PrimitiveRender import PrimitiveRender

## convert params to lists if strings (e.g., '[0,0,0]' --> [0,0,0])
# utils.parse_attributes(args, 'theta_high', 'theta_low', 'pos_high', 'pos_low')

## use a temp folder for copying and manipulating ShapeNet objects
staging = os.path.join(args.staging, str(random.random()))

## choose a renderer based on category
## if from ShapeNet, category is its ID (the mapping is in config.py),
## otherwise category is its name (e.g., Suzanne)
if args.category in config.categories: 
    category = config.categories[args.category] 
    loader = ShapeNetRender(args.shapenet, staging, args.output, create=True)
else:
    category = args.category
    loader = PrimitiveRender()

render_opt = utils.render_parameters[args.category]


################################
########## Rendering ###########
################################

## standard blender operations
blender = BlenderRender(args.gpu)

## rendering intrinsic images along with composite object
intrinsic = IntrinsicRender(args.x_res, args.y_res)

## load light array created with make_array.py
lights = np.load(args.array_path)

blender.sphere([0,0,0], 2.5, label = 'sphere')

count = args.start

while count < args.finish:

    ## load a new object from the category
    ## and copy it for shading / shape renderings
    loader.load(category)
    blender.duplicate('shape', 'shape_shading')
    blender.duplicate('shape', 'shape_normals')

    ## render it args.repeat times in different positions and orientations
    for rep in range(args.repeat):
        
        ## get position, orientation, and scale uniformly at random based on high / low from arguments
        blender.translate( ['shape', 'shape_shading', 'shape_normals'], blender.random(render_opt['pos_low'],   render_opt['pos_high']  ) )
        blender.resize(    ['shape', 'shape_shading', 'shape_normals'], blender.random(render_opt['scale_low'], render_opt['scale_high']) )
        blender.rotate(    ['shape', 'shape_shading', 'shape_normals'], blender.random(render_opt['theta_low'], render_opt['theta_high']) )

        ## lighting parameters from array
        energy, lights_pos = lights[count][0], lights[count][1:]
        blender.light(  energy, lights_pos )

        ## render the composite image and intrinsic images
        for mode in ['composite', 'albedo', 'depth', 'normals', 'shading', 'mask', 'specular', 'lights']:
            filename = str(count) + '_' + mode 
            intrinsic.changeMode(mode)
            blender.write(args.output, filename)
        count += 1

    ## delete object
    blender.delete(lambda x: x.name in ['shape', 'shape_shading', 'shape_normals'] )


################################
########## Reference ###########
################################

#### create a sphere as a shape / shading reference
blender.sphere([0,0,0], 2.5)
blender.duplicate('shape', 'shape_shading')
blender.duplicate('shape', 'shape_normals')

## light it
energy = lights[count][0]
position = lights[count][1:]
blender.light(  energy, position )

## render it
for mode in ['composite', 'albedo', 'depth', 'depth_hires', 'normals', 'shading', 'mask', 'specular']:
    filename = 'sphere_' + mode 
    intrinsic.changeMode(mode)
    blender.write(args.output, filename)

## delete it
blender.delete(lambda x: x.name in ['shape'] )




