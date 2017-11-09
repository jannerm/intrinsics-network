#!/om/user/janner/torch/install/bin/th 

require 'image'
require 'nn'
require 'cutorch'
require 'cunn'
require 'nngraph'
require 'optim'
require 'colormap'
require 'extensions'

-- require 'models'
require 'trainer'
require 'loader'

cmd = torch.CmdLine()
cmd:option('-save_path', 'saved/')
cmd:option('-channels_net', 'small_0.1')
cmd:option('-lights_net', 'lights_only_0.1')
cmd:option('-shader_net', 'norm_shader_0.1')

cmd:option('-image_path', '../dataset/output/car1/')
cmd:option('-array_path', '../dataset/arrays/shader1.npy')
cmd:option('-channels', 3)
cmd:option('-m', 256)
cmd:option('-n', 256)
cmd:option('-param_dim', 4)
cmd:option('-setSize', 10000)
cmd:option('-selectionSize', 20)

cmd:option('-predict_path', 'predictions')
opt = cmd:parse(arg)

paths.mkdir(opt.predict_path)

local intrinsics = {'albedo', 'specular', 'normals'}

local channels_net = torch.load( paths.concat(opt.save_path, opt.channels_net, 'model.t7') )
local lights_net = torch.load( paths.concat(opt.save_path, opt.lights_net, 'model.t7') )
local shader_net = torch.load( paths.concat(opt.save_path, opt.shader_net, 'model.t7') )

local inputs, params, targets = load(opt.image_path, opt.array_path, intrinsics, opt.setSize, opt.selectionSize, opt.channels, opt.m, opt.n, opt.param_dim, true)
-- local val_inp, val_par, _ = load(opt.image_path, opt.array_path, intrinsics, opt.setSize, 10, opt.channels, opt.m, opt.n, opt.param_dim)

inputs = inputs:cuda()

local ch_out = channels_net:forward(inputs)
local li_out = lights_net:forward(inputs)

local albedo = ch_out[{{},{1,3}}]
local specular = ch_out[{{},{4}}]
local normals = ch_out[{{},{5,7}}]
print('channels: ', ch_out:size())
print('lights: ', li_out:size())
print('normals: ', normals:size())

local sh_out = shader_net:forward({normals, li_out})

print('shading: ', sh_out:size())

-- shading = shading:repeatTensor(1,3,1,1)
specular = specular:repeatTensor(1,3,1,1)
-- depth = colormap:convert(depth)
sh_out = sh_out:repeatTensor(1,3,1,1)

reconstruction = torch.cmul(albedo, sh_out) + specular

print(reconstruction:size())

local outputs = {}
local nrow = 6
for ind = 1, opt.selectionSize do
    -- input
    table.insert(outputs, inputs[ind]:float())
    -- channels
    table.insert(outputs, albedo[ind]:float())
    -- table.insert(outputs, shading[ind]:float())
    table.insert(outputs, specular[ind]:float())
    table.insert(outputs, normals[ind]:float())
    -- shader
    table.insert(outputs, sh_out[ind]:float())
    -- reconstruction
    table.insert(outputs, reconstruction[ind]:float())
end

grid = trainer:__grid(outputs, nrow)

image.save( paths.concat(opt.predict_path, 'test_norm.png'), grid)
















