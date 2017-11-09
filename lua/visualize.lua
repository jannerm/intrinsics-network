#!/om/user/janner/torch/install/bin/th 

require 'paths'

-- require paths.thisfile('trainer.lua')
require '../visualizations/Render'

function visualize_channels(channels_net, inputs, targets, save_path)
  local batch_size = 16
  local num_images = inputs:size()[1]
  local channels = trainer:__forward_large(inputs, channels_net, batch_size)
  local albedo_pred = channels[{{},{1,3}}]
  local normals_pred = channels[{{},{4,6}}]
  local albedo_true = targets[{{},{1,3}}]
  local normals_true = targets[{{},{4,6}}]
  
  local masks = normals_true:sum(2):ne(0):double()

  for ind = 1, num_images do
    -- local mask = normals_true[ind]:sum(1):ne(0):double()
    image.save( paths.concat(save_path, ind .. '_observed.png'), torch.cat(inputs[{{ind},{1,3}}]:squeeze():double(), masks[ind], 1) )
    image.save( paths.concat(save_path, ind .. '_albedo_pred.png'), torch.cat(albedo_pred[ind]:double(), masks[ind], 1) )
    image.save( paths.concat(save_path, ind .. '_albedo_true.png'), torch.cat(albedo_true[ind]:double(), masks[ind], 1) )
    image.save( paths.concat(save_path, ind .. '_normals_pred.png'), torch.cat(normals_pred[ind]:double(), masks[ind], 1) )
    image.save( paths.concat(save_path, ind .. '_normals_true.png'), torch.cat(normals_true[ind]:double(), masks[ind], 1) )
  end

  return albedo_pred, normals_pred, masks
end


function visualize_real(channels_net, lights_net, shader_net, inputs, masks, save_path)
  local batch_size = 16
  local num_images = inputs:size()[1]
  local channels = trainer:__forward_large(inputs, channels_net, batch_size)
  local albedo_pred = channels[{{},{1,3}}]
  local normals_pred = channels[{{},{4,6}}]
  local lights_pred = trainer:__forward_large(inputs, lights_net, batch_size)
  local lights_rendered = Render:vis_lights(lights_pred)
  local shading_pred = trainer:__forward_large({normals_pred:cuda(), lights_pred:cuda()}, shader_net, batch_size)
  print('done')
  print(shading_pred:size())
  -- local albedo_true = targets[{{},{1,3}}]
  -- local normals_true = targets[{{},{4,6}}]
  
  -- local masks = normals_true:sum(2):ne(0):double()

  print('Saving to ', save_path)
  for ind = 1, num_images do
    xlua.progress(ind, num_images)
    local obs = inputs[ind]
    local al_pred = torch.cat(albedo_pred[ind]:double(), masks[ind], 1)
    local norm_pred = torch.cat(normals_pred[ind]:double(), masks[ind], 1)
    local li_pred = lights_rendered[ind]
    local shad_pred = torch.cat(shading_pred[ind]:repeatTensor(3,1,1):double(), masks[ind], 1)

    local plot = {obs, al_pred, norm_pred, shad_pred, li_pred}
    -- print(plot)
    local formatted = trainer:__grid(plot, 5)
    image.save( paths.concat(save_path, ind .. '.png'), formatted )
    -- local mask = normals_true[ind]:sum(1):ne(0):double()
    -- image.save( paths.concat(save_path, ind .. '_observed.png'), torch.cat(inputs[{{ind},{1,3}}]:squeeze():double(), masks[ind], 1) )
    -- image.save( paths.concat(save_path, ind .. '_albedo_pred.png'), torch.cat(albedo_pred[ind]:double(), masks[ind], 1) )
    -- image.save( paths.concat(save_path, ind .. '_albedo_true.png'), torch.cat(albedo_true[ind]:double(), masks[ind], 1) )
    -- image.save( paths.concat(save_path, ind .. '_normals_pred.png'), torch.cat(normals_pred[ind]:double(), masks[ind], 1) )
    -- image.save( paths.concat(save_path, ind .. '_normals_true.png'), torch.cat(normals_true[ind]:double(), masks[ind], 1) )
  end

  return albedo_pred, normals_pred, lights_pred, shad_pred
end

function visualize_lights(lights_net, inputs, targets, save_path)
  local batch_size = 16
  local num_images = inputs:size()[1]
  local params = trainer:__forward_large(inputs, lights_net, batch_size)
  local lights_pred = Render:vis_lights(params)
  local lights_true = Render:vis_lights(targets)

  for ind = 1, num_images do
    image.save( paths.concat(save_path, ind .. '_lights_pred.png'), lights_pred[ind])
    image.save( paths.concat(save_path, ind .. '_lights_true.png'), lights_true[ind])
  end

  return params
end

function visualize_shader(shader_net, inputs, masks, targets, save_path)
  local batch_size = 16
  local num_images = inputs[1]:size()[1]
  local shading_pred = trainer:__forward_large(inputs, shader_net, batch_size)
  local shading_true = targets:repeatTensor(1,3,1,1)
  local shading_pred = shading_pred:repeatTensor(1,3,1,1)

  local masks = masks:double()
  local masked_shading_pred = torch.cat(shading_pred, masks, 2)
  local masked_shading_true = torch.cat(shading_true, masks, 2)

  for ind = 1, num_images do
    image.save( paths.concat(save_path, ind .. '_shading_pred.png'), masked_shading_pred[ind])
    image.save( paths.concat(save_path, ind .. '_shading_true.png'), masked_shading_true[ind])
  end

  return shading_pred
end

function visualize_reconstructions(albedo, shading, masks, save_path)
  local num_images = albedo:size()[1]
  local albedo = albedo:double()
  local shading = shading:double()
  local reconstructions = torch.cmul(albedo, shading)
  local masked_recon = torch.cat(reconstructions, masks, 2)

  for ind = 1, num_images do
        image.save( paths.concat(save_path, ind .. '_reconstruction.png'), masked_recon[ind])
  end

  return reconstructions
end

function visualize_composer(model, fixed, inputs, targets, lights_true, save_path, formatted_path, label, save_true)
  local batch_size = 16
  local num_images = inputs:size()[1]
  print('Model outputs')
  local channels = trainer:__forward_large(inputs, model, batch_size)

  local albedo_pred = channels[1]
  local normals_pred = channels[2]
  local lights_pred = channels[3]
  local albedo_true = targets[{{},{1,3}}]
  local normals_true = targets[{{},{4,6}}]
  local shading_true = targets[{{},{7}}]:repeatTensor(1,3,1,1)

  local lights_pred_rendered = Render:vis_lights(lights_pred)
  local lights_true_rendered = Render:vis_lights(lights_true)
  
  print('Fixed outputs')
  local output = trainer:__forward_large(channels, fixed, batch_size)
  local reconstructions = output[1]
  local shading_pred = output[2]:repeatTensor(1,3,1,1)

  local masks = normals_true:sum(2):ne(0):double()

  print('Saving...')
  for ind = 1, num_images do
    xlua.progress(ind, num_images)
    -- local mask = normals_true[ind]:sum(1):ne(0):double()
    local recon = torch.cat(reconstructions[ind]:double(), masks[ind], 1)
    local obs = inputs[ind] --torch.cat(inputs[ind]:double(), masks[ind], 1)

    local al_pred = torch.cat(albedo_pred[ind]:double(), masks[ind], 1)
    local al_true = torch.cat(albedo_true[ind]:double(), masks[ind], 1)
    local norm_pred = torch.cat(normals_pred[ind]:double(), masks[ind], 1)
    local norm_true = torch.cat(normals_true[ind]:double(), masks[ind], 1)
    local shad_pred = torch.cat(shading_pred[ind]:double(), masks[ind], 1)
    local shad_true = torch.cat(shading_true[ind]:double(), masks[ind], 1)
    local light_pred = lights_pred_rendered[ind]
    local light_true = lights_true_rendered[ind]
    
    local plot_pred = trainer:__grid({recon, al_pred, norm_pred, shad_pred, light_pred}, 5)
    image.save( paths.concat(formatted_path, ind .. '_'..label..'.png'), plot_pred ) 

    if save_true then
      local plot_true = trainer:__grid({obs, al_true, norm_true, shad_true, light_true}, 5)
      image.save( paths.concat(formatted_path, ind .. '_true.png'), plot_true )  

      image.save( paths.concat(save_path, ind .. '_observed.png'), obs )
      image.save( paths.concat(save_path, ind .. '_albedo_true.png'), al_true ) 
      image.save( paths.concat(save_path, ind .. '_normals_true.png'), norm_true )
      image.save( paths.concat(save_path, ind .. '_shading_true.png'), shad_true )
      image.save( paths.concat(save_path, ind .. '_lights_true.png'), light_true )
    end

    image.save( paths.concat(save_path, ind .. '_reconstruction_' .. label .. '.png'), recon )
    image.save( paths.concat(save_path, ind .. '_albedo_' .. label .. '.png'), al_pred )
    image.save( paths.concat(save_path, ind .. '_normals_' .. label .. '.png'), norm_pred )
    image.save( paths.concat(save_path, ind .. '_shading_' .. label .. '.png'), shad_pred )
    image.save( paths.concat(save_path, ind .. '_lights_' .. label .. '.png'), light_pred )
    
  end

  -- return albedo_pred, normals_pred, masks

end
