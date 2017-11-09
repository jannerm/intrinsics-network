#!/om/user/janner/torch/install/bin/th 

require 'image'
require 'nn'
require 'nngraph'
require 'optim'
require 'colormap'

require 'models'
require 'trainer'
require 'loader'


cmd = torch.CmdLine()
cmd:option('-image_path', '../dataset/output/')
cmd:option('-array_path', '../dataset/arrays/shader2.npy')
cmd:option('-train_sets', 'car_normalized')
cmd:option('-channels', 3)
cmd:option('-m', 256)
cmd:option('-n', 256)
cmd:option('-param_dim', 4)
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

paths.mkdir(opt.save_path)

intrinsics = {'albedo'}

model = lights_model(opt.param_dim)
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
    local inputs, params, _ = load(opt.train_paths, opt.array_path, intrinsics, opt.setSize, opt.selectionSize, opt.channels, opt.m, opt.n, opt.param_dim, true)
    -- local val_inp, val_par, _ = load(opt.image_path, opt.array_path, intrinsics, opt.setSize, 10, opt.channels, opt.m, opt.n, opt.param_dim)
    
    if opt.gpu >= 0 then
        inputs = inputs:cuda()
        params = params:cuda()
        -- targets = targets:cuda()
        -- val_inp = val_inp:cuda()
        -- val_par = val_par:cuda()
    end

    local err = trainer:train(inputs, params, opt.repeats, opt.batchSize)
    logger:add{[tostring(opt.lr)] = err}; 
    -- logger:style{[tostring(opt.lr)] = '-'}; logger:plot()

    -- local formatted = trainer:visualize(val_inp)
    -- image.save( paths.concat(opt.save_path, epoch .. '.png'), formatted )

    if opt.save_model > 0 then
        local filename = paths.concat(opt.save_path, 'model.net')
        model:clearState()
        torch.save(filename, model)
    end

    sgdState.learningRate = math.max(opt.lr*0.991^epoch, 0.000001)
    sgdState.momentum = math.min(sgdState.momentum + 0.0008, 0.7)

    epoch = epoch + 1
end






