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
cmd:option('-shader_net', 'shadows_vector_shader_0.1')
-- cmd:option('-net_path', '../skip/saved_components/')
-- cmd:option('-shader_net', 'shadows_shader_0.1')
cmd:option('-image_path', '../dataset/output/shadows_vector/')
-- cmd:option('-image_path', '../dataset/output/shadows/')
cmd:option('-save_path', 'nips/shadows_vector_shader_output')
cmd:option('-channels', 3)
cmd:option('-m', 256)
cmd:option('-n', 256)
cmd:option('-param_dim', 6)
opt = cmd:parse(arg)

print('reading')
local shader = torch.load( paths.concat(opt.net_path, opt.shader_net, 'model.net') )
print('done')
shader:evaluate()

local IMG_INDS = {}
for ind = 1000, 3000, 10 do
    table.insert(IMG_INDS, ind)
end

local lights = torch.Tensor{  {-4.5, -10, 1, 95, 0, -25},
                              {0, -10, 1, 95, 0, 0},
                              {4.5, -10, 1, 95, 0, 25}}

local num_lights = lights:size()[1]
-- -- local X = {-7, -6, -5, 0, 5, 6, 5}
-- -- local Z = {10, 9, 8, 7, 6, 2, -2, -3, -4, -5, -6}
-- local X = {-5, 0, 5}
-- -- local Z = {6, 2, -2}
-- local Z = {2}

-- local POSITIONS = {}
-- for _, z in pairs(Z) do
--   for _, x in pairs(X) do
--     table.insert(POSITIONS, {x, z})
--   end
-- end

paths.mkdir(opt.save_path)
for outer_count, i in pairs(IMG_INDS) do
    xlua.progress(outer_count, #IMG_INDS)
    local normals = torch.Tensor(num_lights, opt.channels, opt.m, opt.n)
    local params = torch.Tensor(num_lights, opt.param_dim)
    local img = image.load( paths.concat(opt.image_path, i .. '_normals.png') )[{{1,opt.channels}}]

    if outer_count == 1 then
      img = image.load( 'real_images/beethoven_masked.png' )
    end

    local mask = img:sum(1):ne(0):double()

    for ind = 1, num_lights do
      -- local par = torch.Tensor{energy, x, y, z}
      normals[ind] = img
      params[ind] = lights[ind]
    end

    -- save normals
    local masked_normals = torch.cat(img, mask, 1)
    image.save( paths.concat(opt.save_path, i..'_normals.png'), masked_normals)
    
    -- save lights
    -- if outer_count == 1 then 
      -- local rendered_lights = Render:vis_lights(params)
      -- local formatted_lights = make_grid(rendered_lights, #X)
      -- image.save( paths.concat(opt.save_pos_path, 'lights.png'), formatted_lights )
    -- end

    -- save outputs
    local input = {normals:cuda(), params:cuda()}
    local output = shader:forward(input):float()

    local plot = {}
    for i = 1, output:size()[1] do
        local masked_output = torch.cat(output[i]:repeatTensor(3,1,1), mask:float(), 1)
        table.insert(plot, masked_output)
    end

    local formatted_outputs = make_grid(plot, num_lights)
    image.save( paths.concat(opt.save_path, i..'_output.png'), formatted_outputs )
end




