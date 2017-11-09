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
cmd:option('-net_path', 'saved/')
-- cmd:option('-channels_net', 'saving200_channels_0.01')
-- cmd:option('-lights_net', 'saving_lights_0.01')
-- cmd:option('-shader_net', 'saving_shader_0.01')
-- cmd:option('-channels_net', 'normals_channels_0.1')
-- cmd:option('-lights_net', 'normals_lights_0.1')
-- cmd:option('-shader_net', 'normals_shader_0.1')
cmd:option('-image_path', '../dataset/output/')
cmd:option('-array_path', '../dataset/arrays/shader2.npy')
cmd:option('-sup_sets', 'car_normalized')
cmd:option('-unsup_sets', 'car_normalized')
cmd:option('-test_path', 'car_normalized')
cmd:option('-channels', 3)
cmd:option('-m', 256)
cmd:option('-n', 256)
cmd:option('-param_dim', 4)
cmd:option('-expand_dim', 8) -- parameters will be mapped to an expand_dim x expand_dim layer
cmd:option('-set_size', 12000)
cmd:option('-sup_size', 2000)
cmd:option('-selection_size', 400)
cmd:option('-batch_size', 16)
cmd:option('-repeats', 5)
cmd:option('-save_path', 'saved/test/')
cmd:option('-save_model', 0)
cmd:option('-gpu', 1)
cmd:option('-lr', 0.01)
cmd:option('-momentum', 0)
cmd:option('-multipliers', '10:0,1,1,1,1', 'render, albedo, normals, lights, shading')
cmd:option('-test_size', 500)
cmd:option('-val_freq', 1)
cmd:option('-val_save_freq', 20)
cmd:option('-num_val_save', 100)
opt = cmd:parse(arg)

opt.sup_paths = datasets_to_table(opt.image_path, opt.sup_sets)
opt.unsup_paths = datasets_to_table(opt.image_path, opt.unsup_sets)

-- albedo, specular, normals, lights
opt.multipliers, opt.duration = parse_multipliers(opt.multipliers)

print(opt)

print('<Main> Multipliers:', opt.multipliers)

paths.mkdir(opt.save_path)

local intrinsics = {'albedo', 'normals'}

print('<Main> Loading networks')
local channels_net = skip_model(intrinsics)
local lights_net = lights_model(opt.param_dim)
local shader_net = shader_model(opt.param_dim, opt.expand_dim)
print('<Main> Composing networks')
local model = semisupervised_model(channels_net, lights_net, shader_net)
local criterion = nn.MSECriterion()

if opt.gpu >= 0 then
    model = model:cuda()
    criterion = criterion:cuda()
end

local input = torch.randn(5,4,256,256):cuda()
local output = model:forward(input)

print(output)

parameters, gradParameters = model:getParameters()
sgdState = {
  learningRate = opt.lr,
  momentum = opt.momentum,
}

trainer:init(optim.rmsprop, model, criterion, intrinsics, opt.channels)
log_path = paths.concat(opt.save_path , 'train_err')
train_logger = logger:init( log_path )

local zerosSingle = torch.zeros(opt.batch_size, 1, opt.m, opt.n):cuda()
local zerosChannels = torch.zeros(opt.batch_size, opt.channels, opt.m, opt.n):cuda()
local zerosLights = torch.zeros(opt.batch_size, opt.param_dim):cuda()

-- local const_inp, const_par, const_targ, const_mask = load(opt.train_paths, opt.array_path, {'albedo', 'normals', 'shading'}, opt.setSize, opt.val_size, opt.channels, opt.m, opt.n, opt.param_dim, true)
-- local inputs, params, targets = load_sequential( opt.image_path, opt.array_path, intrinsics, opt.test_size, opt.channels, opt.m, opt.n, opt.param_dim, true )
local rep = 10
local const_inp, const_par, const_targ, const_mask = load_sequential( paths.concat(opt.image_path, opt.test_path), opt.array_path, {'albedo', 'normals', 'shading'}, opt.test_size, opt.channels, opt.m, opt.n, opt.param_dim, true, rep )
local const_intrinsics, const_shading = convert_for_val(const_targ, const_par)

print('const targ')
print(const_targ:size())
print('const shading')
print(const_shading:size())

-- print(const_intrinsics)
-- print(const_shading:size())
-- print(const_par:size())

local sup_inds = torch.floor(torch.rand(opt.sup_size)*opt.set_size)
-- grad_inds = torch.zeros(opt.batch_size)
-- grad_inds[{{1,opt.batch_size/2}}]:fill(1)
grad_mask = {zerosChannels, zerosChannels, zerosChannels, zerosLights, zerosSingle}
grad_mask = table.fill(grad_mask, 1, opt.batch_size/2, 1)
grad_mask[1][{{1,opt.batch_size/2}}]:fill(0)
grad_mask[1][{{1,opt.batch_size/2}}]:fill(1)

zero_table = {zerosChannels:clone(), zerosChannels:clone(), zerosChannels:clone(), zerosLights:clone(), zerosSingle:clone()}

epoch = 0

while true do 
    
    local mults = choose_mults(opt.multipliers, epoch)
    print(mults)

    if epoch % opt.val_freq == 0 then
        local errors, preds, truth = trainer:validate_semisupervised(const_inp, const_mask, fixed, const_intrinsics, const_shading, true)
        local albedo_err, normals_err, lights_err, shading_err, render_err = unpack(errors)
        print(string.format('\n#### Intrinsic Error     albedo:  %.8f | normals:   %.8f', albedo_err, normals_err) )
        print(string.format('                         lights: %.8f | shading: %.8f', lights_err, shading_err) )
        print(string.format('                         render: %.8f\n', render_err) )
        trainer:log_intrinsics(opt.save_path, errors)
        -- if epoch % opt.val_save_freq == 0 then
        --     local folder
        --     if epoch == 0 then
        --         folder = 'original'
        --     else
        --         folder = 'trained'
        --     end
        --     trainer:save_val(paths.concat(opt.save_path, folder), const_inp, preds, truth, opt.num_val_save)
        -- end
    end

    ---- training with labeled inputs ---- 
    local inds = random.choice(sup_inds, math.min(opt.selection_size, sup_inds:numel()) )
    local sup_inputs, sup_params, sup_channels, _ = load_inds(opt.sup_paths, opt.array_path, {'albedo', 'normals', 'shading'}, inds, opt.channels, opt.m, opt.n, opt.param_dim, true)
    local val_inp, _, _, val_masks = load(opt.sup_paths, opt.array_path, intrinsics, opt.set_size, 10, opt.channels, opt.m, opt.n, opt.param_dim)
    
    ---- training with unlabeled inputs ----
    local unsup_inputs, _, _, _ = load(opt.unsup_paths, opt.array_path, {'albedo'}, opt.set_size, opt.selection_size, opt.channels, opt.m, opt.n, opt.param_dim, true)

   local sup_targets = convert_semisupervised(sup_inputs, sup_channels, sup_params)

    if opt.gpu >= 0 then
        unsup_inputs = unsup_inputs:cuda()
        val_inp = val_inp:cuda()
        val_masks = val_masks:cuda()
        sup_inputs = sup_inputs:cuda()
        sup_params = sup_params:cuda()
        sup_targets = table.cuda(sup_targets)
    end

    local unsup_err, sup_err = trainer:train_semisupervised(unsup_inputs, sup_inputs, sup_targets, mults, opt.repeats, opt.batch_size)

    logger:add(log_path, unsup_err)
    trainer:plot_logs(opt.save_path)

    ---- save validation images ----
    local formatted = trainer:visualize_semisupervised(val_inp, val_masks, fixed)
    image.save( paths.concat(opt.save_path, epoch .. '.png'), formatted )

    if opt.save_model > 0 then
        local model_filename = paths.concat(opt.save_path, 'model.net')
        model:clearState()
        torch.save(model_filename, model)
    end

    sgdState.learningRate = math.max(opt.lr*0.991^epoch, 0.000001)
    sgdState.momentum = math.min(sgdState.momentum + 0.0008, 0.7)

    epoch = epoch + 1

    unsup_inputs, val_inp, val_masks, sup_inputs, sup_params, sup_targets = nil, nil, nil, nil, nil, nil
    collectgarbage()
end





