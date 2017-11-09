#!/om/user/janner/torch/install/bin/th 

require 'image'
require 'nn'
require 'nngraph'
require 'optim'
require 'colormap'
require 'extensions'

require 'models'
require 'trainer'
require 'loader'


cmd = torch.CmdLine()
cmd:option('-image_path', '../dataset/output/')
cmd:option('-array_path', '../dataset/arrays/shader2.npy')
cmd:option('-train_sets', 'car_normalized')
cmd:option('-val_sets', 'car_normalized,motorbike_normalized,boat_normalized,bottle_normalized')
cmd:option('-channels', 3)
cmd:option('-m', 256)
cmd:option('-n', 256)
cmd:option('-param_dim', 4)
cmd:option('-expand_dim', 8) -- parameters will be mapped to an expand_dim x expand_dim layer
cmd:option('-setSize', 12000)
cmd:option('-selectionSize', 1600)
cmd:option('-batchSize', 16)
cmd:option('-repeats', 3)
cmd:option('-save_path', 'saved/test/')
cmd:option('-save_model', 0)
cmd:option('-gpu', 1)
cmd:option('-lr', 1)
cmd:option('-momentum', 0)
cmd:text()

-- parse input params
opt = cmd:parse(arg)

opt.train_paths = datasets_to_table(opt.image_path, opt.train_sets)
opt.val_paths = datasets_to_table(opt.image_path, opt.val_sets)
-- opt.val_paths = table.join(opt.train_paths, val_paths)

print(opt)

-- opt.image_path = paths.concat( opt.image_path, opt.dataset )
-- opt.array_path = paths.concat( opt.array_path, opt.dataset .. '.npy')

paths.mkdir(opt.save_path)

intrinsics = {'normals'}

model = shader_model(opt.param_dim, opt.expand_dim)
criterion = nn.MSECriterion()

if opt.gpu >= 0 then
    model = model:cuda()
    criterion = criterion:cuda()
end

parameters, gradParameters = model:getParameters()
sgdState = {
  learningRate = opt.lr,
  momentum = opt.momentum,
}

trainer:init(optim.rmsprop, model, criterion, intrinsics, opt.channels)
logger = optim.Logger(paths.concat(opt.save_path , 'err.log'))

epoch = 1
while true do 
    local inputs, params, targets = load_shader(opt.train_paths, opt.array_path, opt.setSize, opt.selectionSize, opt.channels, opt.m, opt.n, opt.param_dim, true)
    local val_inp1, val_par1, val_targ1 = load_shader(opt.train_paths, opt.array_path, opt.setSize, 10, opt.channels, opt.m, opt.n, opt.param_dim)
    local val_inp2, val_par2, val_targ2 = load_shader(opt.val_paths, opt.array_path, 200, 10, opt.channels, opt.m, opt.n, opt.param_dim)

    local val_inp = torch.cat(val_inp1, val_inp2, 1)
    local val_par = torch.cat(val_par1, val_par2, 1)
    local val_targ = torch.cat(val_targ1, val_targ2, 1)

    
    if opt.gpu >= 0 then
        inputs = inputs:cuda()
        params = params:cuda()
        targets = targets:cuda()
        val_inp = val_inp:cuda()
        val_par = val_par:cuda()
        val_targ = val_targ:cuda()
    end

    local err = trainer:train_shader(inputs, params, targets, opt.repeats, opt.batchSize)
    logger:add{[tostring(opt.lr)] = err}; 
    -- logger:style{[tostring(opt.lr)] = '-'}; logger:plot()

    local formatted = trainer:visualize_shader(val_inp, val_par, val_targ)
    image.save( paths.concat(opt.save_path, epoch .. '.png'), formatted )

    if opt.save_model > 0 then
        local filename = paths.concat(opt.save_path, 'model.net')
        model:clearState()
        torch.save(filename, model)
    end

    sgdState.learningRate = math.max(opt.lr*0.991^epoch, 0.000001)
    sgdState.momentum = math.min(sgdState.momentum + 0.0008, 0.7)

    epoch = epoch + 1
end






