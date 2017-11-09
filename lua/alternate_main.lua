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
require 'visualize'


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
cmd:option('-test_sets', 'car_normalized,motorbike_normalized,boat_normalized,bottle_normalized')
cmd:option('-test_size', 200)
cmd:option('-test_freq', 10)
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
opt.test_paths = datasets_to_table(opt.image_path, opt.test_sets)

paths.mkdir(opt.save_path)

intrinsics = {'albedo', 'shading'}

model = skip_model(intrinsics)
criterion = nn.MSECriterion()

if opt.gpu >= 0 then
    model = model:cuda()
    criterion = criterion:cuda()
end

---- test input
local inp = torch.randn(opt.batchSize, opt.channels+1, opt.m, opt.n):cuda()
local out = model:forward(inp)
print('<Main> Output size:')
print(out:size())
local norm = out:pow(2):sum(2):sqrt()
local min = torch.min(norm)
local max = torch.max(norm)
print('<Main> Normals min / max norm:')
print(min, max)
----

parameters, gradParameters = model:getParameters()
sgdState = {
  learningRate = opt.lr,
  momentum = opt.momentum,
}

trainer:init(optim.rmsprop, model, criterion, intrinsics, opt.channels)
logger = optim.Logger(paths.concat(opt.save_path , 'err.log'))

local test_inp, test_par, test_targ = {}, {}, {}
for i, path in pairs(opt.test_paths) do 
    paths.mkdir( paths.concat(opt.save_path, 'test_' .. i) )
    paths.mkdir( paths.concat(opt.save_path, 'test_saved_' .. i) )
    local ti, tp, tt = load_sequential( path, opt.array_path, intrinsics, opt.test_size, opt.channels, opt.m, opt.n, opt.param_dim, true )
    table.insert(test_inp, ti:cuda())
    table.insert(test_par, tp:cuda())
    table.insert(test_targ, tt:cuda())
end

epoch = 0
while true do 
    local inputs, params, targets = load(opt.train_paths, opt.array_path, intrinsics, opt.setSize, opt.selectionSize, opt.channels, opt.m, opt.n, opt.param_dim, true)
    local val_inp, val_par, _ = load(opt.train_paths, opt.array_path, intrinsics, opt.setSize, 10, opt.channels, opt.m, opt.n, opt.param_dim)
    
    if opt.gpu >= 0 then
        inputs = inputs:cuda()
        -- params = params:cuda()
        targets = targets:cuda()
        val_inp = val_inp:cuda()
        -- val_par = val_par:cuda()
    end

    local err = trainer:train(inputs, targets, opt.repeats, opt.batchSize)
    print('done train')
    logger:add{[tostring(opt.lr)] = err}; 
    print('done err')
    -- logger:style{[tostring(opt.lr)] = '-'}; logger:plot()
    print('done plot')

    local formatted = trainer:visualize(val_inp)
    print('done format')
    image.save( paths.concat(opt.save_path, epoch .. '.png'), formatted )
    print('done save')

    -- if epoch % opt.test_freq == 0 then
    --     -- generalization images
    --     for i = 1, #test_inp do
    --         local save_test_path = paths.concat(opt.save_path, 'test_' .. i)
    --         local ti, tp, tt = test_inp[i], test_par[i], test_targ[i]
    --         visualize_channels(model, ti, tt, save_test_path)
    --     end
    -- end

    if opt.save_model > 0 then
        -- local t7 = paths.concat(opt.save_path, 'model.t7')
        -- torch.save(t7, model)

        local filename = paths.concat(opt.save_path, 'model.net')
        model:clearState()
        torch.save(filename, model)

        -- if epoch % opt.test_freq == 0 then
        --     -- generalization images for loaded model
        --     local loaded_model = torch.load( filename )
        --     for i = 1, #test_inp do
        --         local save_test_path = paths.concat(opt.save_path, 'test_saved_' .. i)
        --         local ti, tp, tt = test_inp[i], test_par[i], test_targ[i]
        --         visualize_channels(loaded_model, ti, tt, save_test_path)
        --     end
        -- end
    end

    print('end while')
    sgdState.learningRate = math.max(opt.lr*0.991^epoch, 0.000001)
    sgdState.momentum = math.min(sgdState.momentum + 0.0008, 0.7)

    epoch = epoch + 1
end






