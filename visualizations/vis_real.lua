#!/om/user/janner/torch/install/bin/th 

require 'paths'
require 'image'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'

require '../skip/models'
require '../skip/trainer'
require '../skip/loader'

require '../skip/extensions'
-- require '../skip/logger'
require '../skip/visualize'


cmd = torch.CmdLine()
cmd:option('-net_path', '../skip/saved_multitrain_components/')
cmd:option('-channels_net', 'car_normalized,motorbike_normalized,chair_normalized,bottle_normalized_channels_0.01')
cmd:option('-lights_net', 'car_normalized,motorbike_normalized,chair_normalized,bottle_normalized_lights_0.01')
cmd:option('-shader_net', 'car_normalized,motorbike_normalized,chair_normalized,bottle_normalized_shader_0.01')
cmd:option('-image_path', '../coco/airplane_processed')
-- cmd:option('-array_path', '../dataset/arrays/shader2.npy')
cmd:option('-save_path', 'multitrain_airplane_processed')
cmd:option('-channels', 3)
cmd:option('-m', 256)
cmd:option('-n', 256)
-- cmd:option('-param_dim', 4)
cmd:option('-test_size', 244)
opt = cmd:parse(arg)

paths.mkdir(opt.save_path)

local intrinsics = {'albedo', 'normals', 'shading'}

print('<Main> Loading networks')
local channels_net = torch.load( paths.concat(opt.net_path, opt.channels_net, 'model.net') )
local lights_net = torch.load( paths.concat(opt.net_path, opt.lights_net, 'model.net') )
local shader_net = torch.load( paths.concat(opt.net_path, opt.shader_net, 'model.net') )
print('<Main> Done')

local inputs, masks = load_real( opt.image_path, opt.test_size, opt.channels, opt.m, opt.n, true )
-- local channels_targets = targets[{{},{1,opt.channels*2}}]
-- local shading_targets = targets[{{},{opt.channels*2+1}}]

print('visualizing real')
visualize_real(channels_net, lights_net, shader_net, inputs, masks, opt.save_path)

-- print('Visualizing albedo, normals')
-- local albedo_pred, normals_pred, masks = visualize_channels(channels_net, inputs, channels_targets, opt.save_path)
-- print('Visualizing lights')
-- local lights_pred = visualize_lights(lights_net, inputs, params, opt.save_path)

-- local normals_pred = normals_pred:cuda()
-- local lights_pred = lights_pred:cuda()
-- local shading_input = {normals_pred, lights_pred}

-- print('Visualizing shading')
-- local shading_pred = visualize_shader(shader_net, shading_input, masks, shading_targets, opt.save_path)
-- print('Visualizing reconstructions')
-- local reconstructions = visualize_reconstructions(albedo_pred, shading_pred, masks, opt.save_path)






