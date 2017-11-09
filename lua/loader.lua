require 'paths'
npy = require 'npy4th'

local INTRINSIC_CHANNELS = {albedo=3, depth=1, normals=3, shading=1, specular=1}

function datasets_to_table(path, datasets)
    local tbl = {}
    for _, set in pairs(string.split(datasets, ',')) do
        table.insert(tbl, paths.concat(path, set) )
    end
    return tbl
end

function parse_multipliers(multipliers)
    local separated = string.split(multipliers, '-')
    local curriculum = {}
    local total_length = 0
    for ind, sep in pairs(separated) do
        local period = string.split(sep, ':')
        local duration, mults = period[1], period[2]
        curriculum[ind] = {duration, table.tonumber(string.split(mults, ','))}
        total_length = total_length + duration
    end
    return curriculum
end

function choose_mults(multipliers, epoch)
    local ind = 1
    local remaining_length = epoch
    local next_length = tonumber(multipliers[(ind-1) % #multipliers+1][1])
    while remaining_length > next_length do
        remaining_length = remaining_length - multipliers[(ind-1) % #multipliers+1][1]
        ind = ind + 1
        next_length = tonumber(multipliers[(ind-1) % #multipliers+1][1])
    end
    return multipliers[(ind-1) % #multipliers+1][2]
end

function load_sequential(image_path, array_path, labels, testSize, channels, dim_x, dim_y, param_dim, verbose, rep)
    local verbose = verbose or false
    local repeats = rep or 1
    if verbose then
        print('<Loader>', 'Loading testing subset of size', testSize, '(load_sequential)')
        print('<Loader>', image_path, array_path)
    end

    local outDim = 0
    for _, label in pairs(labels) do
        outDim = outDim + INTRINSIC_CHANNELS[label]
    end
    local targets = torch.Tensor(testSize, outDim, dim_x, dim_y)
    local params = torch.Tensor(testSize, param_dim)
    local masks = torch.Tensor(testSize, 1, dim_x, dim_y)
    local tensor = npy.loadnpy(array_path)

    local shading = torch.Tensor(testSize, 1, dim_x, dim_y)

    for ind = 1, testSize do

        if verbose then
            xlua.progress(ind, testSize)
        end

        local img_num = ind * repeats

        local current = 1
        for _, name in pairs(labels) do
            local img = image.load( paths.concat(image_path, img_num .. '_' .. name .. '.png') )
            local ch = INTRINSIC_CHANNELS[name]
            img = img[{{1,ch}}]
            targets[{{ind},{current,current+ch-1}}] = img
            current = current + ch
        end
        shading[ind] = image.load( paths.concat(image_path, img_num .. '_shading.png') )[{{1}}]
        local par = tensor[img_num+1]
        params[ind] = par
        masks[ind] = image.load( paths.concat(image_path, img_num .. '_mask.png') )[{{1}}]:gt(0)
    end

    assert(labels[1] == 'albedo', 'Expecting first output to be albedo. Are labels rearranged?')

    -- composite = albedo .* shading 
    local composite = torch.cmul(targets[{{},{1, channels},{},{}}], shading:repeatTensor(1,3,1,1))
    local inputs = torch.cat(composite, masks, 2)
    print('inputs size:')
    print(inputs:size())

    return inputs, params, targets, masks
end

function load_inds(image_paths, array_path, labels, inds, channels, dim_x, dim_y, param_dim, verbose)
    local verbose = verbose or false
    local size = inds:numel()
    if verbose then
        print('<Loader>', 'Loading testing subset of size', size, '(load_inds)')
        print('<Loader>', image_path, array_path)
    end

    local outDim = 0
    for _, label in pairs(labels) do
        outDim = outDim + INTRINSIC_CHANNELS[label]
    end
    local targets = torch.Tensor(size, outDim, dim_x, dim_y)
    local params = torch.Tensor(size, param_dim)
    local masks = torch.Tensor(size, 1, dim_x, dim_y)
    local tensor = npy.loadnpy(array_path)

    local shading = torch.Tensor(size, 1, dim_x, dim_y)

    local repeats = 4
    for i = 1, size do

        if verbose then
            xlua.progress(i, size)
        end

        -- lua tables are 1-indexed, hence ceiling
        local path_num = torch.ceil(torch.rand(1)*#image_paths)[1]
        local path = image_paths[path_num]

        local img_num = inds[i]

        local current = 1
        for _, name in pairs(labels) do
            local img = image.load( paths.concat(path, img_num .. '_' .. name .. '.png') )
            local ch = INTRINSIC_CHANNELS[name]
            img = img[{{1,ch}}]
            targets[{{i},{current,current+ch-1}}] = img
            current = current + ch
        end
        shading[i] = image.load( paths.concat(path, img_num .. '_shading.png') )[{{1}}]
        local par = tensor[img_num+1]
        params[i] = par
        masks[i] = image.load( paths.concat(path, img_num .. '_mask.png') )[{{1}}]:gt(0)
    end

    assert(labels[1] == 'albedo', 'Expecting first output to be albedo. Are labels rearranged?')

    -- composite = albedo .* shading 
    local composite = torch.cmul(targets[{{},{1, channels},{},{}}], shading:repeatTensor(1,3,1,1))
    local inputs = torch.cat(composite, masks, 2)

    return inputs, params, targets, masks
end

function load(image_paths, array_path, labels, setSize, selectionSize, channels, dim_x, dim_y, param_dim, verbose)    
    local verbose = verbose or false
    if verbose then
        print('<Loader>', 'Loading training subset of size', selectionSize, 'from total set of size', setSize, '(load)')
        print('<Loader>', image_paths, array_path)
    end

    -- local labels = {'albedo', 'depth', 'shading', 'specular'}
    local outDim = 0
    for _, label in pairs(labels) do
        outDim = outDim + INTRINSIC_CHANNELS[label]
    end
    local targets = torch.Tensor(selectionSize, outDim, dim_x, dim_y)
    local params = torch.Tensor(selectionSize, param_dim)
    local masks = torch.Tensor(selectionSize, 1, dim_x, dim_y)
    local tensor = npy.loadnpy(array_path)

    local shading = torch.Tensor(selectionSize, 1, dim_x, dim_y)

    for ind = 1, selectionSize do

        if verbose then
            xlua.progress(ind, selectionSize)
        end

        -- lua tables are 1-indexed, hence ceiling
        local path_num = torch.ceil(torch.rand(1)*#image_paths)[1]
        local path = image_paths[path_num]

        local img_num = torch.floor(torch.rand(1)*setSize)[1]

        local current = 1
        for _, name in pairs(labels) do
            local img = image.load( paths.concat(path, img_num .. '_' .. name .. '.png') )
            local ch = INTRINSIC_CHANNELS[name]
            -- if name == 'albedo' or name == 'normals' then
            --     ch = 3
            --     -- img = img[{{1,3}}]
            -- else 
            --     ch = 1
            --     -- img = img[1]
            --     -- channel = channel
            -- end
            img = img[{{1,ch}}]
            targets[{{ind},{current,current+ch-1}}] = img
            current = current + ch
            -- targets[{{ind},{(ch-1)*channels+1,(math.min(ch*channels,outDim))},{},{}}] = img
        end
        shading[ind] = image.load( paths.concat(path, img_num .. '_shading.png') )[{{1}}]
        local par = tensor[img_num+1]
        params[ind] = par
        masks[ind] = image.load( paths.concat(path, img_num .. '_mask.png') )[{{1}}]:gt(0)
    end

    -- depth = 1 - depth
    assert(#labels==0 or labels[1] == 'albedo', 'Expecting first output to be albedo. Are labels rearranged?')
    -- assert(labels[2] == 'shading', 'Expecting second output to be shading. Are labels rearranged?')
    -- assert(labels[3] == 'specular', 'Expecting third output to be specular. Are labels rearranged?')
    -- assert(labels[4] == 'depth', 'Expecting fourth output to be depth. Are labels rearranged?')
    -- assert(labels[5] == 'normals', 'Expecting fifth output to be normals. Are labels rearranged?')
    -- targets[{{},{channels+3},{},{}}] = 1 - targets[{{},{channels+3},{},{}}]

    -- composite = albedo .* shading 
    local composite = torch.cmul(targets[{{},{1, channels},{},{}}], shading:repeatTensor(1,3,1,1))
    -- + specular
    -- inputs = torch.add(inputs, targets[{{},{channels+1},{},{}}]:repeatTensor(1,3,1,1))
    local inputs = torch.cat(composite, masks, 2)

    return inputs, params, targets, masks

end

function load_real(image_path, selectionSize, channels, dim_x, dim_y, verbose)    
    local verbose = verbose or false
    if verbose then
        print('<Loader>', 'Loading real subset of size', selectionSize)
        print('<Loader>', image_path)
    end

    local images = torch.Tensor(selectionSize, channels, dim_x, dim_y)
    local masks = torch.Tensor(selectionSize, 1, dim_x, dim_y)

    for ind = 1, selectionSize do

        if verbose then
            xlua.progress(ind, selectionSize)
        end

        local img = image.load( paths.concat(image_path, ind .. '.png') )
        -- img = image.scale(img, dim_x, dim_y)
        local mask = img:eq(0):sum(1):ne(3)
        -- local mask = torch.ones(1, dim_x, dim_y)

        images[ind] = img
        masks[ind] = mask
    end

    local inputs = torch.cat(images, masks, 2)

    return inputs, masks
end

function load_real_inds(image_path, inds, channels, dim_x, dim_y, verbose)    
    local verbose = verbose or false
    if verbose then
        print('<Loader>', 'Loading real subset of size', num_imgs, 'by indices')
        print('<Loader>', image_path)
    end

    local num_imgs = #inds

    local images = torch.Tensor(num_imgs, channels, dim_x, dim_y)
    local masks = torch.Tensor(num_imgs, 1, dim_x, dim_y)

    for ind = 1, num_imgs do

        if verbose then
            xlua.progress(ind, num_imgs)
        end

        local img_num = inds[ind]

        local img = image.load( paths.concat(image_path, img_num .. '.png') )
        -- img = image.scale(img, dim_x, dim_y)
        local mask = img:eq(0):sum(1):ne(3)
        -- local mask = torch.ones(1, dim_x, dim_y)

        images[ind] = img
        masks[ind] = mask
    end

    local inputs = torch.cat(images, masks, 2)

    return inputs, masks
end

-- learn_output, fixed_output = convert_intrinsics_composer(inputs, params, targets)
function convert_intrinsics_composer(inputs, params, targets)
    local albedo = targets[{{},{1,3}}]
    -- local specular = targets[{{},{4}}]
    local normals = targets[{{},{4,6}}]
    local shading = targets[{{},{7}}]
    local learn_output = {albedo, normals, params}
    local fixed_output = {inputs[{{},{1,3}}], shading}
    return learn_output, fixed_output
end

-- learn_output, fixed_output = convert_intrinsics_composer(inputs, params, targets)
function convert_intrinsics_alternate(inputs, targets)
    local albedo = targets[{{},{1,3}}]
    local shading = targets[{{},{4}}]
    local learn_output = {albedo, shading}
    local fixed_output = inputs[{{},{1,3}}]
    return learn_output, fixed_output
end

function convert_semisupervised(inputs, channels, params)
    local albedo = channels[{{},{1,3}}]
    local normals = channels[{{},{4,6}}]
    local shading = channels[{{},{7}}]
    -- {render, albedo, normals, li, shading}
    return {inputs, albedo, normals, params, shading}
end

--albedo, specular, normals, lights, shading
function convert_for_val(targets, params)
    --albedo, specular, normals, lights
    local albedo = targets[{{},{1,3}}]
    -- local specular = targets[{{},{4}}]
    local normals = targets[{{},{4,6}}]
    local shading = targets[{{},{7}}]
    return {albedo, normals, params}, shading
end

function convert_for_alt(targets)
    local albedo = targets[{{},{1,3}}]
    local shading = targets[{{},{4}}]
    return {albedo, shading}
end

function load_shader(image_paths, array_path, setSize, selectionSize, channels, dim_x, dim_y, param_dim, verbose)    
    local verbose = verbose or false
    if verbose then
        print('<Loader>', 'Loading training subset of size', selectionSize, 'from total set of size', setSize)
        print('<Loader>', image_paths, array_path)
    end

    local inputs = torch.Tensor(selectionSize, channels, dim_x, dim_y)
    local targets = torch.Tensor(selectionSize, 1, dim_x, dim_y)
    local params = torch.Tensor(selectionSize, param_dim)
    local tensor = npy.loadnpy(array_path)

    for ind = 1, selectionSize do

        if verbose then
            xlua.progress(ind, selectionSize)
        end

        -- lua tables are 1-indexed, hence ceiling
        local path_num = torch.ceil(torch.rand(1)*#image_paths)[1]
        local path = image_paths[path_num]

        -- datasets are created with Python, so are 0-indexed, hence floor
        local img_num = torch.floor(torch.rand(1)*setSize)[1]

        -- local depth = 1 - image.load( paths.concat(path, img_num .. '_depth.png') )[1]
        local normals = image.load( paths.concat(path, img_num .. '_normals.png') )[{{1,3}}]
        local shad = image.load( paths.concat(path, img_num .. '_shading.png') )[1]
        local par = tensor[img_num+1]

        inputs[ind] = normals
        targets[ind] = shad
        params[ind] = par
    end

    return inputs, params, targets

end

-- function to_gpu(gpu, obj)
--     if gpu >= 0 then
--         obj = obj:cuda()
--     end
--     return obj
-- end