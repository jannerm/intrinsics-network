#!/om/user/janner/torch/install/bin/th 

require 'paths'
require 'image'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'

require 'Render'
require 'grid'

cmd = torch.CmdLine()
cmd:option('-net_path', '../skip/nips_components/')
cmd:option('-shader_net', 'vector_shader_car_vector_0.01')
cmd:option('-image_path', '../dataset/output/car_vector/')
cmd:option('-save_pos_path', 'nips/shader_vector_car_pos')
cmd:option('-save_energy_path', 'nips/shader_vector_car_energy')
-- cmd:option('-net_path', '../skip/saved_components/')
-- cmd:option('-shader_net', 'car_shader_0.1')
-- cmd:option('-image_path', '../dataset/output/car_normalized/')
-- cmd:option('-save_pos_path', 'shader_car_pos_output_normed_single')
-- cmd:option('-save_energy_path', 'shader_car_energy_output_normed_single')
cmd:option('-channels', 3)
cmd:option('-m', 256)
cmd:option('-n', 256)
cmd:option('-param_dim', 4)
opt = cmd:parse(arg)

print('reading')
local shader = torch.load( paths.concat(opt.net_path, opt.shader_net, 'model.net') )
print('done')
shader:evaluate()

local IMG_INDS = {}
for ind = 1850, 1900, 1 do
    table.insert(IMG_INDS, ind)
end
-- IMG_INDS = {8880, 6320, 6810}
-- local X = {-7, -6, -5, 0, 5, 6, 5}
-- local Z = {10, 9, 8, 7, 6, 2, -2, -3, -4, -5, -6}
local X = {-5, 0, 5}
-- local Z = {6, 2, -2}
local Z = {2}

local POSITIONS = {}
for _, z in pairs(Z) do
  for _, x in pairs(X) do
    table.insert(POSITIONS, {x, z})
  end
end

paths.mkdir(opt.save_pos_path)
for outer_count, i in pairs(IMG_INDS) do
    xlua.progress(outer_count, #IMG_INDS)
    local normals = torch.Tensor(#X*#Z, opt.channels, opt.m, opt.n)
    local params = torch.Tensor(#X*#Z, opt.param_dim)
    local img = image.load( paths.concat(opt.image_path, i .. '_normals.png') )[{{1,opt.channels}}]

    -- if outer_count == 1 then
    --   img = image.load( 'real_images/beethoven_normed.png' )
    -- end

    local mask = img:sum(1):ne(0):double()
    local y = -3.5
    local energy = 2.5
    local ind = 1
    for _, p in pairs(POSITIONS) do
      local x = p[1]
      local z = p[2]
      local par = torch.Tensor{energy, x, y, z}
      normals[ind] = img
      params[ind] = par
      ind = ind + 1
    end

    -- save normals
    local masked_normals = torch.cat(img, mask, 1)
    image.save( paths.concat(opt.save_pos_path, i..'_normals.png'), masked_normals)
    
    -- save lights
    if outer_count == 1 then 
      local rendered_lights = Render:vis_lights(params)
      local formatted_lights = make_grid(rendered_lights, #X)
      image.save( paths.concat(opt.save_pos_path, 'lights.png'), formatted_lights )
    end

    -- save outputs
    local input = {normals:cuda(), params:cuda()}
    local output = shader:forward(input):float()

    local plot = {}
    for j = 1, output:size()[1] do
        local masked_output = torch.cat(output[j]:repeatTensor(3,1,1), mask:float(), 1)
        table.insert(plot, masked_output)
        image.save( paths.concat(opt.save_pos_path, i..'_'..j..'.png'), masked_output )
    end

    local formatted_outputs = make_grid(plot, #X)
    image.save( paths.concat(opt.save_pos_path, i..'_output.png'), formatted_outputs )
end



local ENERGY = {.25, 1.5, 4}

paths.mkdir(opt.save_energy_path)
for outer_count, i in pairs(IMG_INDS) do
    xlua.progress(outer_count, #IMG_INDS)
    local normals = torch.Tensor(#ENERGY, opt.channels, opt.m, opt.n)
    local params = torch.Tensor(#ENERGY, opt.param_dim)
    local img = image.load( paths.concat(opt.image_path, i .. '_normals.png') )[{{1,opt.channels}}]
    
    if outer_count == 1 then
      img = image.load( 'real_images/beethoven_normed.png' )
    end

    local mask = img:sum(1):ne(0):double()
    local x, y, z = 0, -3.5, 2
    local ind = 1
    for _, e in pairs(ENERGY) do
        local par = torch.Tensor{e, x, y, z}
        normals[ind] = img:clone()
        params[ind] = par:clone()
        ind = ind + 1
    end

    -- save normals
    local masked_normals = torch.cat(img, mask, 1)
    image.save( paths.concat(opt.save_energy_path, i..'_normals.png'), masked_normals)

    -- save lights
    if outer_count == 1 then 
      local rendered_lights = Render:vis_lights(params)
      local formatted_lights = make_grid(rendered_lights, #ENERGY)
      image.save( paths.concat(opt.save_energy_path, 'lights.png'), formatted_lights )
    end

    local input = {normals:cuda(), params:cuda()}
    local output = shader:forward(input):float()

    local plot = {}
    for j = 1, output:size()[1] do
        local masked_output = torch.cat(output[j]:repeatTensor(3,1,1), mask:float(), 1)
        table.insert(plot, masked_output)
        image.save( paths.concat(opt.save_energy_path, i..'_'..j..'.png'), masked_output )
    end
    local formatted = make_grid(plot, #ENERGY)
    image.save( paths.concat(opt.save_energy_path, i..'.png'), formatted )
end

