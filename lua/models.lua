require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'

local INTRINSIC_CHANNELS = {albedo=3, depth=1, normals=3, shading=1, specular=1}

function conv(in_channels, out_channels, filt, stride, pad)
    local net = nn.Sequential()
    net:add(nn.SpatialConvolution(in_channels, out_channels, filt, filt, stride, stride, pad, pad))
    net:add(nn.SpatialBatchNormalization(out_channels))
    net:add(nn.LeakyReLU(0.2, true))
    return net
end

function join()
    return nn.JoinTable(1,3)
end

function upsample(scale)
    return nn.SpatialUpSamplingNearest(scale)
end

function normalize()
    print('<Model> Normalizing Normals')
    return function (layer, mask)
        local norm = nn.Power(2)(layer)
        norm = nn.Sum(1,3)(norm)
        norm = nn.Sqrt()(norm)
        norm = nn.Replicate(3,1,2)(norm)
        -- add epsilon and inverse
        norm = nn.AddConstant(0.0001)(norm)
        norm = nn.Power(-1)(norm)
        --
        local normed = nn.CMulTable()({layer, norm})
        local masked = nn.CMulTable()({normed, mask})
        return masked
    end
end

function normalize_loaded(model)
    local input = nn.Identity()()
    -- split 4-channel input
    local split = nn.SplitTable(1,3)(input)
    -- select first three channels
    -- local composite = nn.NarrowTable(1,3)(split)
    -- add back first dim
    -- composite = nn.MapTable(nn.Unsqueeze(1,2))(composite)
    -- join back first three channels
    -- composite = nn.JoinTable(1,3)(composite)
    -- select fourth channel and replicate it for later cmul
    local mask = nn.Replicate(3,2,3)(nn.SelectTable(4)(split))

    local intrinsics = model(input)
    local albedo = nn.SelectTable(1)(intrinsics)
    local normals = nn.SelectTable(2)(intrinsics)
    local lights = nn.SelectTable(3)(intrinsics)

    local normed = normalize()(normals, mask)

    local model = nn.gModule({input}, {albedo, normed, lights})
    return model

end

function decode(channels, filt, pad)
    return function (encoder)
        local decoder = {}
        local joined = {}
        stride = 1
        -- print('\nEncoder --> Decoder', channels[#channels], '\n')
        decoder[1] = conv(channels[#channels], channels[#channels], filt, stride, pad)(encoder[#encoder])
        joined[1] = join()({decoder[1], encoder[#encoder]})

        for ind = #channels-1, 1, -1 do
            local dec_ind = #joined + 1
            local inp_node = joined[dec_ind-1]
            local inp_ch = channels[ind+1] * 2
            local out_ch = channels[ind]
            stride = 1
            local scale = 2
            -- print('Decoder: ', inp_ch, '--> ', out_ch)
            if ind ~= 1 then
                decoder[dec_ind] = upsample(scale)(conv(inp_ch, out_ch, filt, stride, pad)(inp_node))
                joined[dec_ind] = join()({decoder[dec_ind], encoder[#encoder-dec_ind+1]})
            else
                decoder[dec_ind] = conv(inp_ch, out_ch, filt, stride, pad)(inp_node)
            end
        end
        return decoder[#decoder]
    end
end

-- this version outputs images only (no lighting parameters)
function skip_model(labels)
    local input = nn.Identity()()
    -- split 4-channel input
    local split = nn.SplitTable(1,3)(input)
    -- select first three channels
    local composite = nn.NarrowTable(1,3)(split)
    -- add back first dim
    composite = nn.MapTable(nn.Unsqueeze(1,2))(composite)
    -- join back first three channels
    composite = nn.JoinTable(1,3)(composite)
    -- select fourth channel and replicate it for later cmul
    -- local mask
    if table.contains(labels, 'normals') then
        mask = nn.Replicate(3,2,3)(nn.SelectTable(4)(split))
    end

    -- model = nn.gModule({input}, {composite, mask})
    -- inp = torch.randn(5,4,40,40)
    -- out = model:forward(inp)
    -- print(inp:size())
    -- print(out)

    local encoder = {}
    local channels = {3, 16, 32, 64, 128, 256, 256}
    local filt = 3
    local pad = 1
    local stride

    -- encoder[1] = conv(channels[1], channels[2], 3, 3, 1, 1, 1, 1)(input)

    for ind = 1, #channels-1 do
        local inp_node = encoder[ind-1] or composite
        local inp_ch = channels[ind]
        local out_ch = channels[ind+1]
        if ind == 1 then stride = 1 else stride = 2 end
        -- print('Encoder: ', inp_ch, '--> ', out_ch)
        encoder[ind] = conv(inp_ch, out_ch, filt, stride, pad)(inp_node)
    end

    local intrinsics = {}
    for ind, label in pairs(labels) do
        channels[1] = INTRINSIC_CHANNELS[label]
        local deconv = decode(channels, filt, pad)(encoder)
        if label == 'normals' then
            deconv = normalize()(deconv, mask)
        end
        intrinsics[ind] = deconv
    end

    local output = join()(intrinsics)

    local model = nn.gModule({input}, {output})
    return model

end

function lights_model_link(in_channels)
    -- return function (encoded)
    local net = nn.Sequential()
    net:add( conv(in_channels, 3, 3, 1, 1) )
    net:add( conv(3, 1, 3, 1, 1) )
    net:add( nn.View(-1):setNumInputDims(3) )
    net:add( nn.Linear(8*8,4) )
    -- print('net: ', net)
    return net
    -- end
end

function lights_model(param_dim)
    local input = nn.Identity()()
    -- split 4-channel input
    local split = nn.SplitTable(1,3)(input)
    -- select first three channels
    local composite = nn.NarrowTable(1,3)(split)
    -- add back first dim
    composite = nn.MapTable(nn.Unsqueeze(1,2))(composite)
    -- join back first three channels
    composite = nn.JoinTable(1,3)(composite)

    local encoder = {}
    local channels = {3, 16, 3, 1}
    local filt = 5
    local pad = 0
    local stride = 3

    -- encoder[1] = conv(channels[1], channels[2], 3, 3, 1, 1, 1, 1)(input)

    for ind = 1, #channels-1 do
        local inp_node = encoder[ind-1] or composite
        local inp_ch = channels[ind]
        local out_ch = channels[ind+1]
        -- if ind == 1 then stride = 1 else stride = 2 end
        -- print('Encoder: ', inp_ch, '--> ', out_ch)
        encoder[ind] = conv(inp_ch, out_ch, filt, stride, pad)(inp_node)
    end

    local lin = nn.Reshape(8*8)(encoder[#encoder])
    lin = nn.Linear(8*8, param_dim)(lin)

    local model = nn.gModule({input}, {lin})
    return model
end

-- this version outputs images only (no lighting parameters)
function skip_model_lights(labels)
    local input = nn.Identity()()
    local encoder = {}
    local channels = {3, 16, 32, 64, 128, 256, 256}
    local filt = 3
    local pad = 1
    local stride

    -- encoder[1] = conv(channels[1], channels[2], 3, 3, 1, 1, 1, 1)(input)

    for ind = 1, #channels-1 do
        local inp_node = encoder[ind-1] or input
        local inp_ch = channels[ind]
        local out_ch = channels[ind+1]
        if ind == 1 then stride = 1 else stride = 2 end
        -- print('Encoder: ', inp_ch, '--> ', out_ch)
        encoder[ind] = conv(inp_ch, out_ch, filt, stride, pad)(inp_node)
    end

    local intrinsics = {}
    for ind, label in pairs(labels) do
        if label == 'albedo' then
            channels[1] = 3
        else
            channels[1] = 1
        end
        intrinsics[ind] = decode(channels, filt, pad)(encoder)
    end

    local images = join()(intrinsics)
    local params = lights_model_link(channels[#channels])(encoder[#encoder])

    local model = nn.gModule({input}, {images, params})
    return model

end

function shader_model(param_dim, expand_dim)
    local input = nn.Identity()()
    local depth = nn.SelectTable(1)(input)
    local params = nn.SelectTable(2)(input)

    local encoder = {}
    local channels = {3, 16, 32, 64, 128, 256, 256}
    local filt = 3
    local pad = 1
    local stride

    -- encoder[1] = conv(channels[1], channels[2], 3, 3, 1, 1, 1, 1)(input)

    for ind = 1, #channels-1 do
        local inp_node = encoder[ind-1] or depth
        local inp_ch = channels[ind]
        local out_ch = channels[ind+1]
        if ind == 1 then stride = 1 else stride = 2 end
        -- print('Encoder: ', inp_ch, '--> ', out_ch)
        encoder[ind] = conv(inp_ch, out_ch, filt, stride, pad)(inp_node)
    end

    -- map params to a channel the same size as final encoder layer
    local expanded = nn.Linear(param_dim, expand_dim*expand_dim)(params)
    expanded = nn.Reshape(1, expand_dim, expand_dim)(expanded)

    -- join the final encoder layer with the expanded params
    -- increase channels[-1] to account for the extra layer
    encoder[#encoder] = join()({encoder[#encoder], expanded})
    channels[#channels] = channels[#channels] + 1

    -- output single-channeled shading
    channels[1] = 1

    local shading = decode(channels, filt, pad)(encoder)

    local model = nn.gModule({input}, {shading})
    return model
end

function semisupervised_model(channels, lights, shader)
    local inp = nn.Identity()()
    local ch = channels(inp)
    local li = lights(inp)

    local split = nn.SplitTable(2)(ch)

    local albedo = nn.NarrowTable(1,3)(split)
    albedo = nn.MapTable():add(nn.Unsqueeze(2))(albedo)
    albedo = nn.JoinTable(1,3)(albedo)

    local normals = nn.NarrowTable(4,3)(split)
    normals = nn.MapTable():add(nn.Unsqueeze(2))(normals)
    normals = nn.JoinTable(1,3)(normals)

    local shading = shader({normals, li})
    shading_rep = nn.Squeeze()(shading)
    shading_rep = nn.Replicate(3,2)(shading_rep)

    local render = nn.CMulTable()({albedo, shading_rep})

    local model = nn.gModule({inp}, {render, albedo, normals, li, shading})
    return model
end

function composer_model(channels, lights, shader)
    local learn = __composer_learn(channels, lights)
    local fixed = __composer_fixed(shader)
    return learn, fixed
end

function __composer_learn(channels, lights)
    local inp = nn.Identity()()
    local ch = channels(inp)
    local li = lights(inp)

    local split = nn.SplitTable(2)(ch)

    local albedo = nn.NarrowTable(1,3)(split)
    albedo = nn.MapTable():add(nn.Unsqueeze(2))(albedo)
    albedo = nn.JoinTable(1,3)(albedo)

    -- local specular = nn.NarrowTable(4,1)(split)
    -- specular = nn.JoinTable(1,3)(specular)
    -- specular = nn.Unsqueeze(2)(specular)

    local normals = nn.NarrowTable(4,3)(split)
    normals = nn.MapTable():add(nn.Unsqueeze(2))(normals)
    normals = nn.JoinTable(1,3)(normals)

    local model = nn.gModule({inp}, {albedo, normals, li})
    return model
end

function __composer_fixed(shader)
    local inp = nn.Identity()()
    local albedo = nn.SelectTable(1)(inp)
    -- local specular = nn.SelectTable(2)(inp)
    local normals = nn.SelectTable(2)(inp)
    local li = nn.SelectTable(3)(inp)

    -- specular = nn.Squeeze()(specular)
    -- specular = nn.Replicate(3,2)(specular)

    local shading = shader({normals, li})
    shading_rep = nn.Squeeze()(shading)
    shading_rep = nn.Replicate(3,2)(shading_rep)

    local render = nn.CMulTable()({albedo, shading_rep})
    -- render = nn.CAddTable()({render, specular})
    -- albedo, specular, normals, lights, shading
    local model = nn.gModule({inp}, {render, shading})
    return model
end

function alternate_composer(channels)
    local learn = __alternate_learn(channels)
    local fixed = __alternate_fixed()
    return learn, fixed
end

function __alternate_learn(channels)
    local inp = nn.Identity()()
    local ch = channels(inp)

    local split = nn.SplitTable(2)(ch)

    local albedo = nn.NarrowTable(1,3)(split)
    albedo = nn.MapTable():add(nn.Unsqueeze(2))(albedo)
    albedo = nn.JoinTable(1,3)(albedo)

    local shading = nn.NarrowTable(4,1)(split)
    shading = nn.SelectTable(1)(shading)
    -- shading = nn.Unsqueeze(2)(shading)

    local model = nn.gModule({inp}, {albedo, shading})
    return model
end

function __alternate_fixed()
    local inp = nn.Identity()()
    local albedo = nn.SelectTable(1)(inp)
    local shading = nn.SelectTable(2)(inp)
    shading = nn.Replicate(3,2)(shading)

    local render = nn.CMulTable()({albedo, shading})

    local model = nn.gModule({inp}, {render})
    return model
end

-- model = lights_model(4):cuda()
-- inp = torch.randn(5,3,256,256):cuda()
-- -- -- params = torch.randn(5,4):cuda()
-- out = model:forward(inp)
-- print(inp:size(), out:size())



