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
require '../skip/visualize'

cmd = torch.CmdLine()
cmd:option('-net_path', '../skip/saved_composer_images/')
cmd:option('-model', 'bottle_car_10:0,1,0,5_0.0001')
cmd:option('-image_path', '../dataset/output/car_normalized')
cmd:option('-array_path', '../dataset/arrays/shader2.npy')
cmd:option('-save_path', 'composer_output_bottle_car_2000')
cmd:option('-channels', 3)
cmd:option('-m', 256)
cmd:option('-n', 256)
cmd:option('-param_dim', 4)
cmd:option('-test_size', 500)
opt = cmd:parse(arg)

local formatted_path = paths.concat(opt.save_path, 'formatted')
paths.mkdir(opt.save_path)
paths.mkdir( formatted_path )

local intrinsics = {'albedo', 'normals', 'shading'}

print('<Main> Loading networks')
local model = torch.load( paths.concat(opt.net_path, opt.model, 'model.net') )
local fixed = torch.load( paths.concat(opt.net_path, opt.model, 'fixed.net') )
local model_raw = torch.load( paths.concat(opt.net_path, opt.model, 'model_raw.net') )
local fixed_raw = torch.load( paths.concat(opt.net_path, opt.model, 'fixed_raw.net') )
print('<Main> Composing networks')

local inputs, params, targets = load_sequential( opt.image_path, opt.array_path, intrinsics, opt.test_size, opt.channels, opt.m, opt.n, opt.param_dim, true )
local channels_targets = targets[{{},{1,opt.channels*2}}]
local shading_targets = targets[{{},{opt.channels*2+1}}]

visualize_composer(model, fixed, inputs, targets, params, opt.save_path, formatted_path, 'pred', true)
visualize_composer(model_raw, fixed_raw, inputs, targets, params, opt.save_path, formatted_path, 'raw', false)

