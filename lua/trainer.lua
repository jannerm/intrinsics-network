require 'optim'
-- require 'logger'

trainer = {}

function trainer:init(train_func, model, criterion, labels, channels)
  self.func = train_func
  self.model = model
  self.criterion = criterion
  self.labels = labels
  self.channels = channels
  self.loggers = {}
  self.initialized_loggers = false
end


function trainer:train(inputs, targets, repeats, batchSize)
  local setSize = inputs:size()[1]

  local iterations = torch.ceil(setSize / batchSize * repeats)
  local total_err = 0

  print('<Trainer> Beginning RMSprop, epoch', epoch)
  for iter = 1, iterations do 

    xlua.progress(iter, iterations)

    local inds = torch.ceil(torch.rand(batchSize)*setSize):long()
    local inp = inputs:index(1, inds)
    local targ = targets:index(1, inds)

    ---- enclosure to eval model
    local feval = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local out = self.model:forward(inp)
      local err = self.criterion:forward(out, targ)
      local gradErr = self.criterion:backward(out, targ)
      self.model:backward(inp, gradErr)

      total_err = total_err + err
      return err, gradParameters
    end
    ----

    self.func(feval, parameters, sgdState)
  end

  print('<Trainer> Total error: ', total_err)
  return total_err
end

function trainer:train_composer_table(model, inputs, targets, repeats, batchSize)
  local setSize
  if type(inputs) == 'table' then
    setSize = inputs[1]:size()[1]
  else
    setSize = inputs:size()[1]
  end

  local iterations = torch.ceil(setSize / batchSize * repeats)
  local total_err = 0

  print('<Trainer> Beginning RMSprop, epoch', epoch)
  for iter = 1, iterations do 

    xlua.progress(iter, iterations)

    local inds = torch.ceil(torch.rand(batchSize)*setSize):long()
    local inp, targ
    if type(inputs) == 'table' then
      inp = table.index(inputs, 1, inds)
    else
      inp = inputs:index(1, inds)
    end
    if type(targets) == 'table' then
      targ = table.index(targets, 1, inds)
    else
      targ = targets:index(1, inds)
    end

    ---- enclosure to eval model
    local feval = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local out = model:forward(inp)
      local err, grad_err = table.criterion(self.criterion, out, targ)
      model:backward(inp, grad_err)

      total_err = total_err + table.sum(err)
      return err, gradParameters
    end
    ----

    self.func(feval, parameters, sgdState)
  end

  print('<Trainer> Total error: ', total_err)
  return total_err
end

function trainer:train_semisupervised(unsup_inputs, sup_inputs, sup_targets, multipliers, repeats, batch_size)
  local set_size = unsup_inputs:size()[1]

  local iterations = torch.ceil(set_size / batch_size * repeats)
  local unsup_total_err = 0
  local sup_total_err = 0

  print('<Trainer> Beginning RMSprop, epoch', epoch)
  for iter = 1, iterations do 
    xlua.progress(iter, iterations)

    -- get a mini-batch of half labeled, half unlabeled
    local inds = torch.ceil(torch.rand(batch_size/2)*set_size):long()
    local unsup_inp = unsup_inputs:index(1, inds)
    local sup_inp = sup_inputs:index(1, inds)
    local inputs = torch.cat(sup_inp, unsup_inp, 1)
    -- print('got inputs')
    -- print(inputs:size())

    local sup_targ = table.index(sup_targets, 1, inds)
    sup_targ[1]=sup_targ[1][{{},{1,3}}]
    local unsup_targ = table.index(zero_table, 1, inds)
    local targets = table.cat(sup_targ, unsup_targ, 1)
    -- print('got targets')
    -- print(targets)

    ---- enclosure f(x), df(x)/dx
    local feval = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local outputs = self.model:forward(inputs)

      local err, grad_err = table.criterion(self.criterion, outputs, targets)
      grad_err = table.cmul(grad_err, grad_mask)
      -- local err = self.criterion:forward(render, unsup_targ)
      -- local grad_err = self.criterion:backward(render, unsup_targ)
      -- local grad_trans = {grad_err, zerosSingle}
      -- local grad = fixed:backward(out, grad_trans)
      assert(#grad_err == #multipliers, 'Unequal number of gradients and multipliers')
      -- print('before: ', grad_err[1]:sum(), grad_err[2]:sum(), grad_err[3]:sum(), grad_err[4]:sum(), grad_err[5]:sum())
      for i = 1, #grad_err do
        grad_err[i] = grad_err[i] * multipliers[i]
      end
      -- print('after: ', grad_err[1]:sum(), grad_err[2]:sum(), grad_err[3]:sum(), grad_err[4]:sum(), grad_err[5]:sum())
      self.model:backward(inputs, grad_err)

      unsup_total_err = unsup_total_err + err[1]
      sup_total_err = sup_total_err + table.sum(table.slice(err, 2, #err))
      return err, gradParameters
    end

    self.func(feval, parameters, sgdState)
  end

  print('<Trainer> Total unsupervised error: ', unsup_total_err, '| Supervised error', sup_total_err)
  return unsup_total_err, sup_total_err
end

function trainer:train_composer_interleaved(unsup_inputs, fixed, multipliers, sup_mult, sup_inputs, sup_targets, repeats, batchSize)
  local unsup_set_size = unsup_inputs:size()[1]
  local sup_set_size
  if type(sup_inputs) == 'table' then
    sup_set_size = sup_inputs[1]:size()[1]
  else
    sup_set_size = sup_inputs:size()[1]
  end

  local iterations = torch.ceil(unsup_set_size / batchSize * repeats)
  local unsup_total_err = 0
  local sup_total_err = 0

  print('<Trainer> Beginning RMSprop, epoch', epoch)
  for iter = 1, iterations do 

    xlua.progress(iter, iterations)

    ---- train unsupervised
    local unsup_inds = torch.ceil(torch.rand(batchSize)*unsup_set_size):long()
    local unsup_inp = unsup_inputs:index(1, unsup_inds)
    local unsup_targ = unsup_inp[{{},{1,3}}]:clone()

    ---- enclosure f(x), df(x)/dx
    local feval_unsupervised = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local out = self.model:forward(unsup_inp)
      local transformed = fixed:forward(out)
      local render = transformed[1]
      local err = self.criterion:forward(render, unsup_targ)
      local grad_err = self.criterion:backward(render, unsup_targ)
      local grad_trans = {grad_err, zerosSingle}
      local grad = fixed:backward(out, grad_trans)
      assert(#grad == #multipliers, 'Unequal number of gradients and multipliers')
      for i = 1, #grad do
        grad[i] = grad[i] * multipliers[i]
      end
      self.model:backward(unsup_inp, grad)

      unsup_total_err = unsup_total_err + err
      return err, gradParameters
    end

    if table.sum(multipliers) > 0 then
      self.func(feval_unsupervised, parameters, sgdState)
    end

    ---- train supervised
    local sup_inds = torch.ceil(torch.rand(batchSize)*sup_set_size):long()
    local sup_inp, sup_targ
    -- if type(sup_inputs) == 'table' then
      -- sup_inp = table.index(sup_inputs, 1, sup_inds)
    -- else
      sup_inp = sup_inputs:index(1, sup_inds)
    -- end
    -- if type(sup_targets) == 'table' then
      sup_targ = table.index(sup_targets, 1, sup_inds)
    -- else
      -- sup_targ = sup_targets:index(1, sup_inds)
    -- end
    
    ---- enclosure f(x), df(x)/dx
    local feval_supervised = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local out = self.model:forward(sup_inp)
      local err, grad_err = table.criterion(self.criterion, out, sup_targ)
      grad_err = table.mul(grad_err, sup_mult)
      self.model:backward(sup_inp, grad_err)

      sup_total_err = sup_total_err + table.sum(err)
      return err, gradParameters
    end

    if sup_mult > 0 then
      self.func(feval_supervised, parameters, sgdState)
    end

  end

  print('<Trainer> Total unsupervised error: ', unsup_total_err, '| Supervised error', sup_total_err)
  return unsup_total_err, sup_total_err
end

function trainer:train_alternate_interleaved(unsup_inputs, fixed, multipliers, sup_mult, sup_inputs, sup_targets, repeats, batchSize)
  local unsup_set_size = unsup_inputs:size()[1]
  local sup_set_size
  if type(sup_inputs) == 'table' then
    sup_set_size = sup_inputs[1]:size()[1]
  else
    sup_set_size = sup_inputs:size()[1]
  end

  local iterations = torch.ceil(unsup_set_size / batchSize * repeats)
  local unsup_total_err = 0
  local sup_total_err = 0

  print('<Trainer> Beginning RMSprop, epoch', epoch)
  for iter = 1, iterations do 

    xlua.progress(iter, iterations)

    ---- train unsupervised
    local unsup_inds = torch.ceil(torch.rand(batchSize)*unsup_set_size):long()
    local unsup_inp = unsup_inputs:index(1, unsup_inds)
    local unsup_targ = unsup_inp[{{},{1,3}}]:clone()

    ---- enclosure f(x), df(x)/dx
    local feval_unsupervised = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local out = self.model:forward(unsup_inp)
      local render = fixed:forward(out)
      local err = self.criterion:forward(render, unsup_targ)
      local grad_err = self.criterion:backward(render, unsup_targ)
      -- local grad_trans = {grad_err, zerosSingle}
      local grad = fixed:backward(out, grad_err)
      assert(#grad == #multipliers, 'Unequal number of gradients and multipliers')
      for i = 1, #grad do
        grad[i] = grad[i] * multipliers[i]
      end
      self.model:backward(unsup_inp, grad)

      unsup_total_err = unsup_total_err + err
      return err, gradParameters
    end

    self.func(feval_unsupervised, parameters, sgdState)

    ---- train supervised
    local sup_inds = torch.ceil(torch.rand(batchSize)*sup_set_size):long()
    local sup_inp, sup_targ
    -- if type(sup_inputs) == 'table' then
      -- sup_inp = table.index(sup_inputs, 1, sup_inds)
    -- else
      sup_inp = sup_inputs:index(1, sup_inds)
    -- end
    -- if type(sup_targets) == 'table' then
      sup_targ = table.index(sup_targets, 1, sup_inds)
    -- else
      -- sup_targ = sup_targets:index(1, sup_inds)
    -- end
    
    ---- enclosure f(x), df(x)/dx
    local feval_supervised = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local out = self.model:forward(sup_inp)
      local err, grad_err = table.criterion(self.criterion, out, sup_targ)
      grad_err = table.mul(grad_err, sup_mult)
      self.model:backward(sup_inp, grad_err)

      sup_total_err = sup_total_err + table.sum(err)
      return err, gradParameters
    end

    self.func(feval_supervised, parameters, sgdState)

  end

  print('<Trainer> Total unsupervised error: ', unsup_total_err, '| Supervised error', sup_total_err)
  return unsup_total_err, sup_total_err
end

function trainer:train_composer(inputs, fixed, multipliers, repeats, batchSize)
  local setSize = inputs:size()[1]

  local iterations = torch.ceil(setSize / batchSize * repeats)
  local total_err = 0

  print('<Trainer> Beginning RMSprop, epoch', epoch)
  for iter = 1, iterations do 

    xlua.progress(iter, iterations)

    local inds = torch.ceil(torch.rand(batchSize)*setSize):long()
    local inp = inputs:index(1, inds)
    local targ = inp:clone()
    -- local targ = targets:index(1, inds)

    ---- enclosure to eval model
    local feval = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local out = self.model:forward(inp)
      local transformed = fixed:forward(out)
      local render = transformed[1]
      local err = self.criterion:forward(render, targ)
      local grad_err = self.criterion:backward(render, targ)
      local grad_trans = {grad_err, zerosSingle}
      local grad = fixed:backward(out, grad_trans)
      assert(#grad == #multipliers, 'Unequal number of gradients and multipliers')
      for i = 1, #grad do
        grad[i] = grad[i] * multipliers[i]
      end
      self.model:backward(inp, grad)

      total_err = total_err + err
      return err, gradParameters
    end
    ----

    self.func(feval, parameters, sgdState)
  end

  print('<Trainer> Total error: ', total_err)
  return total_err
end

function trainer:__forward_large(inputs, model, batchSize)
  local model = model or self.model
  local numInputs
  if type(inputs) == 'table' then
    numInputs = inputs[1]:size()[1]
  else
    numInputs = inputs:size()[1]
  end
  local batchSize = batchSize or 16
  local outputs
  for start = 1, numInputs, batchSize do
    xlua.progress(start, numInputs)
    local finish = math.min(start + batchSize - 1, numInputs)
    -- print(start, finish)
    local inp
    if type(inputs) == 'table' then
      inp = table.span(inputs, start, finish)
      inp = table.cuda(inp)
    else
      inp = inputs[{{start, finish}}]
      inp = inp:cuda()
    end

    local out = model:forward(inp)

    if type(out) == 'table' then
      out = table.clone(out)
      out = table.double(out)
      if outputs then
        outputs = table.cat(outputs, out, 1)
      else
        outputs = out
      end
    else
      out = out:clone():double()
      if outputs then
        outputs = torch.cat(outputs, out, 1)
      else
        outputs = out
      end
    end
  end
  -- finish progress bar
  xlua.progress(numInputs, numInputs)
  return outputs
end

function trainer:validate_composer(inputs, masks, fixed, intrinsics, shading, return_images)
  local return_images = return_images or false
  local criterion = nn.MSECriterion()
  local numInputs = inputs:size()[1]
  print('<Validator> Getting intrinsic predictions')
  local intrinsic_pred = self:__forward_large(inputs)
  local albedo_pred, normals_pred, lights_pred = unpack( self:mask(intrinsic_pred, masks, 1) )
  -- print(albedo_pred:size(), lights_pred:size())
  print('<Validator> Getting reconstructions')
  local fixed_pred = self:__forward_large(intrinsic_pred, fixed)
  -- print('render preds: ', fixed_pred)
  local render_pred, shading_pred = unpack( self:mask(fixed_pred, masks) )
  -- print(render_pred:size(), shading_pred:size())

  local albedo, normals, lights = unpack(intrinsics)
  local albedo_err = criterion:forward(albedo_pred, albedo:double())
  -- local spec_err = criterion:forward(spec_pred, spec:double())
  local normals_err = criterion:forward(normals_pred, normals:double())
  local lights_err = criterion:forward(lights_pred, lights:double())
  local shading_err = criterion:forward(shading_pred, shading:double())
  local render_err = criterion:forward(render_pred, inputs[{{},{1,3}}]:double())

  local errors = {albedo_err, normals_err, lights_err, shading_err, render_err}
  local preds, truth
  if return_images then
    preds = {render_pred, albedo_pred, normals_pred, shading_pred}
    truth = {albedo, normals, shading}
  end
  return errors, preds, truth
end

function trainer:validate_alternate(inputs, masks, fixed, intrinsics, return_images)
  local return_images = return_images or false
  local criterion = nn.MSECriterion()
  local numInputs = inputs:size()[1]
  print('<Validator> Getting intrinsic predictions')
  local intrinsic_pred = self:__forward_large(inputs)
  -- print(intrinsic_pred)
  local albedo_pred, shading_pred = unpack(intrinsic_pred)
  shading_pred = nn.Unsqueeze(2):forward(shading_pred)
  local albedo_pred, shading_pred = unpack( self:mask({albedo_pred, shading_pred}, masks, 0) )
  -- print(albedo_pred:size(), lights_pred:size())
  print('<Validator> Getting reconstructions')
  local fixed_pred = self:__forward_large(intrinsic_pred, fixed)
  -- print('render preds: ', fixed_pred)
  local render_pred = unpack( self:mask({fixed_pred}, masks) )
  -- print(render_pred:size(), shading_pred:size())

  local albedo, shading = unpack(intrinsics)
  local albedo_err = criterion:forward(albedo_pred, albedo:double())
  local shading_err = criterion:forward(shading_pred, shading:double())
  local render_err = criterion:forward(render_pred, inputs[{{},{1,3}}]:double())

  local errors = {albedo_err, shading_err, render_err}
  local preds, truth
  if return_images then
    preds = {render_pred, albedo_pred, shading_pred}
    truth = {albedo, shading}
  end
  return errors, preds, truth
end

function trainer:validate_semisupervised(inputs, masks, fixed, intrinsics, shading, return_images)
  local return_images = return_images or false
  local criterion = nn.MSECriterion()
  local numInputs = inputs:size()[1]
  print('<Validator> Getting intrinsic predictions')
  local outputs = self:__forward_large(inputs)
  local channels = table.select(outputs, {1,2,3,5})
  local lights_pred = outputs[4]
  local render_pred, albedo_pred, normals_pred, shading_pred = unpack( self:mask(channels, masks, 0) )

  -- local albedo_pred, normals_pred, lights_pred = unpack( self:mask(intrinsic_pred, masks, 1) )
  -- print(albedo_pred:size(), lights_pred:size())
  -- print('<Validator> Getting reconstructions')
  -- local fixed_pred = self:__forward_large(intrinsic_pred, fixed)
  -- print('render preds: ', fixed_pred)
  -- local render_pred, shading_pred = unpack( self:mask(fixed_pred, masks) )
  -- print(render_pred:size(), shading_pred:size())

  local albedo, normals, lights = unpack(intrinsics)
  local albedo_err = criterion:forward(albedo_pred, albedo:double())
  -- local spec_err = criterion:forward(spec_pred, spec:double())
  local normals_err = criterion:forward(normals_pred, normals:double())
  local lights_err = criterion:forward(lights_pred, lights:double())
  local shading_err = criterion:forward(shading_pred, shading:double())
  local render_err = criterion:forward(render_pred, inputs[{{},{1,3}}]:double())

  local errors = {albedo_err, normals_err, lights_err, shading_err, render_err}
  local preds, truth
  if return_images then
    preds = {render_pred, albedo_pred, normals_pred, shading_pred}
    truth = {albedo, normals, shading}
  end
  return errors, preds, truth
end

function trainer:log_intrinsics(save_path, errors, labels)
  local labels = labels or {'albedo', 'normals', 'lights', 'shading', 'render'}
  assert(#errors == #labels, 'Unequal number of errors and labels in intrinsic error logger')
  for ind, lab in pairs(labels) do
    local log_path = paths.concat(save_path , lab .. '_err')
    if not self.initialized_loggers then
      logger:init(log_path)
    end
    -- local logger = self.loggers[lab] or optim.Logger(log_path)
    -- self.loggers[lab] = logger
    local err = errors[ind]
    logger:add(log_path, err); 
    -- logger:style{[lab] = '-'}; logger:plot()
  end
  self.initialized_loggers = true
end

function trainer:mask(tbl, mask, exclude)
  local exclude = exclude or 0
  local new = {}
  for ind = 1, #tbl-exclude do
    local tensor = tbl[ind]
    local channels = tensor:size()[2]
    local mask_ch = mask:repeatTensor(1,channels,1,1)
    local masked = torch.cmul(tensor, mask_ch)
    table.insert(new, masked)
  end
  for ind = #tbl-exclude+1, #tbl do
    table.insert(new, tbl[ind])
  end
  return new
end

function trainer:train_lights(inputs, params, targets, repeats, batchSize)
  local setSize = inputs:size()[1]

  local iterations = torch.ceil(setSize / batchSize * repeats)
  local total_err_img = 0
  local total_err_par = 0

  print('<Trainer> Beginning RMSprop, epoch', epoch)
  for iter = 1, iterations do 

    xlua.progress(iter, iterations)

    local inds = torch.ceil(torch.rand(batchSize)*setSize):long()
    local inp = inputs:index(1, inds)
    local par = params:index(1, inds)
    local targ = targets:index(1, inds)

    ---- enclosure to eval model
    local feval = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local out = self.model:forward(inp)
      local images = out[1]
      local lights = out[2]

      local err_img = self.criterion:forward(images, targ)
      local grad_err_img = self.criterion:backward(images, targ):clone()

      local err_par = self.criterion:forward(lights, par)
      local grad_err_par = self.criterion:backward(lights, par):clone()

      local err = err_img + err_par
      self.model:backward(inp, {grad_err_img, grad_err_par})

      total_err_img = total_err_img + err_img
      total_err_par = total_err_par + err_par
      return err, gradParameters
    end
    ----

    self.func(feval, parameters, sgdState)
  end

  print('<Trainer> Total image error: ', total_err_img, 'Total params error: ', total_err_par)
  return total_err
end

function trainer:train_shader(inputs, params, targets, repeats, batchSize)
  local setSize = inputs:size()[1]

  local iterations = torch.ceil(setSize / batchSize * repeats)
  local total_err = 0

  print('<Trainer> Beginning RMSprop, epoch', epoch)
  for iter = 1, iterations do 

    xlua.progress(iter, iterations)

    local inds = torch.ceil(torch.rand(batchSize)*setSize):long()
    local inp = inputs:index(1, inds)
    local par = params:index(1, inds)
    local targ = targets:index(1, inds)

    ---- enclosure to eval model
    local feval = function(x)
      collectgarbage()
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero() 
      local out = self.model:forward({inp, par})
      local err = self.criterion:forward(out, targ)
      local gradErr = self.criterion:backward(out, targ)
      self.model:backward({inp, par}, gradErr)

      total_err = total_err + err
      return err, gradParameters
    end
    ----

    self.func(feval, parameters, sgdState)
  end

  print('<Trainer> Total error: ', total_err)
  return total_err
end

function trainer:__grid(inputs, nrow)
  -- print(inputs)
  local num_img = #inputs
  local size = inputs[1]:size()
  local ch, m, n = size[1], size[2], size[3]
  local grid = torch.Tensor(ch, m*torch.ceil(num_img/nrow), n*nrow )
  for ind = 1, num_img do
    -- print(inputs[ind]:size())

    local i = torch.floor((ind-1) / nrow)
    local j = (ind-1) % nrow
    -- print(grid[{{},{i*m+1,(i+1)*m},{j*n+1,(j+1)*n}}]:size())
    grid[{{},{i*m+1,(i+1)*m},{j*n+1,(j+1)*n}}] = inputs[ind]
  end
  return grid
end

function trainer:visualize(inputs)
  local setSize = inputs:size()[1]
  local outputs = self.model:forward(inputs)
  -- local outDim = outputs:size()[2]
  local plot = {}
  for ind = 1, setSize do
    table.insert(plot, inputs[{{ind},{1,3}}]:squeeze():float())
    local current = 1
    for _, label in pairs(self.labels) do
      local ch
      if label == 'albedo' or label == 'normals' then
        ch = 3
      else
        ch = 1
      end
      -- print('visualizing: ', label, current, current+ch-1)
      local img = outputs[{{ind},{current,current+ch-1}}]
      -- local img = outputs[{{ind},{(l-1)*self.channels+1, math.min(l*self.channels, outDim)},{},{}}]
      if label == 'depth' then
        img = colormap:convert(img)
      elseif label == 'specular' or label == 'shading' then
        img = img:repeatTensor(1,3,1,1)
      end
      table.insert(plot, img:float())
      current = current + ch
    end
  end
  local formatted = self:__grid(plot, (#self.labels+1))
  -- formatted[{{},{1},{1}}]=torch.Tensor{{1,1,1}}
  return formatted
end

function trainer:visualize_semisupervised(inputs, masks, fixed)
  local setSize = inputs:size()[1]
  local outputs = self.model:forward(inputs)

  local rendered = outputs[1]
  local albedo = outputs[2]
  local normals = outputs[3]
  local lights = outputs[4]
  local shading = outputs[5]

  local channels = {rendered, albedo, normals, shading}
  local masked = self:mask( channels, masks, 0 )
  -- local trans = self:mask( fixed:forward(outputs), masks )
  -- local outDim = outputs:size()[2]
  local plot = {}
  for i = 1, setSize do
    table.insert(plot, inputs[{{i},{1,3}}]:squeeze():float())
    for j = 1, #masked do
      local m = masked[j][i]:float()
      if m:size()[1] == 1 then
        m = m:repeatTensor(3,1,1)
      end
      table.insert(plot, m)
    end
  end
  -- inp, albedo, shading, normals, render
  local formatted = self:__grid(plot, 5)
  -- formatted[{{},{1},{1}}]=torch.Tensor{{1,1,1}}
  return formatted
end

function trainer:visualize_composer(inputs, masks, fixed)
  local setSize = inputs:size()[1]
  local outputs = self:mask( self.model:forward(inputs), masks, 1 )
  local trans = self:mask( fixed:forward(outputs), masks )
  -- local outDim = outputs:size()[2]
  local plot = {}
  for i = 1, setSize do
    table.insert(plot, inputs[{{i},{1,3}}]:squeeze():float())
    for j = 1, #outputs-1 do
      local out = outputs[j][i]:float()
      if out:size()[1] == 1 then
        out = out:repeatTensor(3,1,1)
      end
      table.insert(plot, out)
    end
    for j = 1, #trans do
      local tr = trans[j][i]:float()
      if tr:size()[1] == 1 then
        tr = tr:repeatTensor(3,1,1)
      end
      table.insert(plot, tr)
    end
  end
  -- inp, albedo, shading, normals, render
  local formatted = self:__grid(plot, 5)
  -- formatted[{{},{1},{1}}]=torch.Tensor{{1,1,1}}
  return formatted
end

function trainer:visualize_alternate(inputs, masks, fixed)
  local setSize = inputs:size()[1]
  local outputs = self:mask( self.model:forward(inputs), masks, 1 )
  local trans = self:mask( {fixed:forward(outputs)}, masks )
  -- local outDim = outputs:size()[2]
  local plot = {}
  for i = 1, setSize do
    table.insert(plot, inputs[{{i},{1,3}}]:squeeze():float())
    for j = 1, #outputs-1 do
      local out = outputs[j][i]:float()
      if out:size()[1] == 1 then
        out = out:repeatTensor(3,1,1)
      end
      table.insert(plot, out)
    end
    for j = 1, #trans do
      local tr = trans[j][i]:float()
      if tr:size()[1] == 1 then
        tr = tr:repeatTensor(3,1,1)
      end
      table.insert(plot, tr)
    end
  end
  -- inp, albedo, shading, normals, render
  local formatted = self:__grid(plot, 5)
  -- formatted[{{},{1},{1}}]=torch.Tensor{{1,1,1}}
  return formatted
end

function trainer:save_val(save_path, inputs, preds, truth, num) 
  paths.mkdir(save_path)
  local num_save = num_save or 20
  for i = 1, num do
    local plot = {}

    -- render, albedo, spec, normals, shading
    for j = 1, #preds do
      local img = preds[j][i]
      if img:size()[1] == 1 then
        img = img:repeatTensor(3,1,1)
      end
      table.insert(plot, img)
    end

    table.insert(plot, inputs[{{i},{1,3}}]:squeeze())

    for j = 1, #truth  do
      local img = truth[j][i]
      if img:size()[1] == 1 then
        img = img:repeatTensor(3,1,1)
      end
      table.insert(plot, img)
    end
    local formatted = self:__grid(plot, 4)
    image.save( paths.concat(save_path, i .. '.png'), formatted)
  end
end

function trainer:visualize_shader(inputs, params, targets)
  local setSize = inputs:size()[1]
  local outputs = self.model:forward({inputs, params})
  -- local outDim = outputs:size()[2]
  local plot = {}
  for ind = 1, setSize do
    -- table.insert(plot, colormap:convert(inputs[ind]):float())
    table.insert(plot, inputs[ind]:float())
    table.insert(plot, outputs[ind]:repeatTensor(3,1,1):float())
    table.insert(plot, targets[ind]:repeatTensor(3,1,1):float())
  end
  local formatted = self:__grid(plot, 6)
  return formatted
end

function trainer:plot_logs(directory)
  os.execute('th log_figure.lua -directory ' .. directory)
end
