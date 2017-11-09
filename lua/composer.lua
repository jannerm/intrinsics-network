#!/om/user/janner/torch/install/bin/th 

require 'paths'
require 'image'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'

require 'models'
require 'trainer'
require 'loader'

require 'extensions'
require 'logger'


cmd = torch.CmdLine()
cmd:option('-net_path', 'saved_components/')
cmd:option('-base_class', 'car')

cmd:option('-image_path', '../dataset/output/')
cmd:option('-array_path', '../dataset/arrays/shader2.npy')
cmd:option('-train_sets', 'boat_normalized')
cmd:option('-labeled_sets', 'bottle_normalized')
cmd:option('-test_path', 'boat_normalized')
cmd:option('-channels', 3)
cmd:option('-m', 256)
cmd:option('-n', 256)
cmd:option('-param_dim', 4)
-- cmd:option('-expand_dim', 8) -- parameters will be mapped to an expand_dim x expand_dim layer
cmd:option('-setSize', 12000)
cmd:option('-selectionSize', 1200)
cmd:option('-batchSize', 16)
cmd:option('-repeats', 3)
cmd:option('-save_path', 'saved/test/')
cmd:option('-save_model', 0)
cmd:option('-gpu', 1)
cmd:option('-lr', 1)
cmd:option('-momentum', 0)
cmd:option('-multipliers', '10:1,1,1,0.1', 'albedo, normals, lights, supervised')
cmd:option('-test_size', 500)
cmd:option('-val_freq', 1)
cmd:option('-val_save_freq', 20)
cmd:option('-num_val_save', 100)
-- cmd:option('-sup_mult', 0.1)
opt = cmd:parse(arg)

opt.train_paths = datasets_to_table(opt.image_path, opt.train_sets)
opt.labeled_paths = datasets_to_table(opt.image_path, opt.labeled_sets)

-- albedo, specular, normals, lights
opt.multipliers, opt.duration = parse_multipliers(opt.multipliers)

print(opt)

print('<Main> Multipliers:', opt.multipliers)

paths.mkdir(opt.save_path)

intrinsics = {'albedo', 'normals'}

print('<Main> Loading networks')
local channels_net = torch.load( paths.concat(opt.net_path, opt.base_class .. '_channels_0.01', 'model.net') )
local lights_net = torch.load( paths.concat(opt.net_path, opt.base_class .. '_lights_0.01', 'model.net') )
local shader_net = torch.load( paths.concat(opt.net_path, opt.base_class .. '_shader_0.01', 'model.net') )
print('<Main> Composing networks')
local model, fixed = composer_model(channels_net, lights_net, shader_net)
criterion = nn.MSECriterion()

if opt.gpu >= 0 then
    model = model:cuda()
    fixed = fixed:cuda()
    criterion = criterion:cuda()
end

if opt.save_model > 0 then
    local model_filename = paths.concat(opt.save_path, 'model_raw.net')
    local fixed_filename = paths.concat(opt.save_path, 'fixed_raw.net')
    model:clearState()
    fixed:clearState()
    torch.save(model_filename, model)
    torch.save(fixed_filename, fixed)
end

local input = torch.randn(5,4,256,256):cuda()
local output = model:forward(input)
local trans = fixed:forward(output)

print(input:size())
print(output)
print(trans)

parameters, gradParameters = model:getParameters()
sgdState = {
  learningRate = opt.lr,
  momentum = opt.momentum,
}

trainer:init(optim.rmsprop, model, criterion, intrinsics, opt.channels)
log_path = paths.concat(opt.save_path , 'train_err')
train_logger = logger:init( log_path )

-- composite, albedo, specular, normals, lights, shading
-- zerosSingle = torch.zeros(opt.batchSize, 1, opt.m, opt.n)
zerosSingle = torch.zeros(opt.batchSize, 1, opt.m, opt.n):cuda()
-- zerosLights = torch.zeros(opt.batchSize, opt.param_dim)

-- local const_inp, const_par, const_targ, const_mask = load(opt.train_paths, opt.array_path, {'albedo', 'normals', 'shading'}, opt.setSize, opt.val_size, opt.channels, opt.m, opt.n, opt.param_dim, true)
-- local inputs, params, targets = load_sequential( opt.image_path, opt.array_path, intrinsics, opt.test_size, opt.channels, opt.m, opt.n, opt.param_dim, true )
local const_inp, const_par, const_targ, const_mask = load_sequential( paths.concat(opt.image_path, opt.test_path), opt.array_path, {'albedo', 'normals', 'shading'}, opt.test_size, opt.channels, opt.m, opt.n, opt.param_dim, true )
-- const_inp = const_inp:cuda()
-- const_par = const_par:cuda()
-- const_targ = const_targ:cuda()
local const_intrinsics, const_shading = convert_for_val(const_targ, const_par)

print('const targ')
print(const_targ:size())
print('const shading')
print(const_shading:size())

-- print(const_intrinsics)
-- print(const_shading:size())
-- print(const_par:size())


epoch = 0

while true do 
    
    local mults = choose_mults(opt.multipliers, epoch)

    if epoch % opt.val_freq == 0 then
        local errors, preds, truth = trainer:validate_composer(const_inp, const_mask, fixed, const_intrinsics, const_shading, true)
        local albedo_err, normals_err, lights_err, shading_err, render_err = unpack(errors)
        print(string.format('\n#### Intrinsic Error     albedo:  %.8f | normals:   %.8f', albedo_err, normals_err) )
        print(string.format('                         lights: %.8f | shading: %.8f', lights_err, shading_err) )
        print(string.format('                         render: %.8f\n', render_err) )
        trainer:log_intrinsics(opt.save_path, errors)
        if epoch % opt.val_save_freq == 0 then
            local folder
            if epoch == 0 then
                folder = 'original'
                -- model = normalize_loaded(model)
            else
                folder = 'trained'
            end
            trainer:save_val(paths.concat(opt.save_path, folder), const_inp, preds, truth, opt.num_val_save)
        end
    end

    ---- training unsupervised ---- 
    local unsup_inputs, _, _, _ = load(opt.train_paths, opt.array_path, intrinsics, opt.setSize, opt.selectionSize, opt.channels, opt.m, opt.n, opt.param_dim, true)
    local val_inp, _, _, val_masks = load(opt.train_paths, opt.array_path, intrinsics, opt.setSize, 10, opt.channels, opt.m, opt.n, opt.param_dim)
    
    ---- training with ground truth ----
    local sup_inputs, sup_params, sup_targets, _ = load(opt.labeled_paths, opt.array_path, {'albedo', 'normals', 'shading'}, opt.setSize, opt.selectionSize, opt.channels, opt.m, opt.n, opt.param_dim, true)

    if opt.gpu >= 0 then
        unsup_inputs = unsup_inputs:cuda()
        val_inp = val_inp:cuda()
        val_masks = val_masks:cuda()
        sup_inputs = sup_inputs:cuda()
        sup_params = sup_params:cuda()
        sup_targets = sup_targets:cuda()
    end

    local learn_output, fixed_output = convert_intrinsics_composer(sup_inputs, sup_params, sup_targets)

    local unsup_mults = table.slice(mults, 1, #mults-1)
    local sup_mult = mults[#mults]
    print('unsup mults: ', unsup_mults)
    print('sup mults: ', sup_mult)
    local unsup_err, sup_err = trainer:train_composer_interleaved(unsup_inputs, fixed, unsup_mults, sup_mult, sup_inputs, learn_output, opt.repeats, opt.batchSize)
    logger:add(log_path, unsup_err)
    trainer:plot_logs(opt.save_path)
    -- train_logger:add{[tostring(opt.lr)] = err}; 
    -- train_logger:style{[tostring(opt.lr)] = '-'}; train_logger:plot()
    -- inputs, val_inp, val_masks = nil, nil, nil
    -------------------------------


    ---- training with ground truth ----
    -- local inputs, params, targets, _ = load(opt.labeled_paths, opt.array_path, {'albedo', 'specular', 'normals', 'shading'}, opt.setSize, opt.selectionSize/4, opt.channels, opt.m, opt.n, opt.param_dim, true)
    -- -- local val_inp, _, _, val_masks = load(opt.labeled_paths, opt.array_path, intrinsics, opt.setSize, 10, opt.channels, opt.m, opt.n, opt.param_dim)
    -- if opt.gpu >= 0 then
    --     inputs = inputs:cuda()
    --     params = params:cuda()
    --     targets = targets:cuda()
    -- end
    -- local learn_output, fixed_output = convert_intrinsics_composer(inputs, params, targets)
    -- -- albedo, specular, normals, lights
    -- local err_learn = trainer:train_composer_table(model, inputs, learn_output, 2, opt.batchSize)
    -- -- render, shading
    -- local err_fixed = trainer:train_composer_table(fixed, learn_output, fixed_output, 2, opt.batchSize)
    -- print('Learning on ground truths: ', err_learn, err_fixed)
    ------------------------------------

    local formatted = trainer:visualize_composer(val_inp, val_masks, fixed)
    image.save( paths.concat(opt.save_path, epoch .. '.png'), formatted )

    if opt.save_model > 0 then
        local model_filename = paths.concat(opt.save_path, 'model.net')
        local fixed_filename = paths.concat(opt.save_path, 'fixed.net')
        model:clearState()
        fixed:clearState()
        torch.save(model_filename, model)
        torch.save(fixed_filename, fixed)
    end

    sgdState.learningRate = math.max(opt.lr*0.991^epoch, 0.000001)
    sgdState.momentum = math.min(sgdState.momentum + 0.0008, 0.7)

    epoch = epoch + 1

    unsup_inputs, val_inp, val_masks, sup_inputs, sup_params, sup_targets = nil, nil, nil, nil, nil, nil
    collectgarbage()
end





