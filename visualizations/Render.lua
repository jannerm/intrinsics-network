require 'os'
require 'paths'
require 'image'
npy = require 'npy4th'

Render = {}

function Render:vis_lights(lights, verbose)
    local verbose = verbose or false
    local write_path = 'temp_lights_' .. torch.random()
    local lights_path = paths.concat(write_path, 'lights.npy')
    paths.mkdir(write_path)
    npy.savenpy(lights_path, lights)

    local num_lights = lights:size()[1]
    print('\nRendering ' .. num_lights .. '...')
    self:__blender(lights_path, write_path, verbose)

    local images = self:__read_images(write_path, num_lights)
    print('Deleting ' .. lights_path .. '...\n')
    os.execute('rm -r '.. write_path)
    -- paths.rmall(write_path, 'yes')
    return images
end

function Render:__blender(lights_path, write_path, verbose)
    local script_path = '../dataset/vis_lights.py'
    local command = '/om/user/janner/blender-2.76b-linux-glibc211-x86_64/blender ' .. 
                    '--background -noaudio --python ' .. script_path .. ' -- ' ..
                    '--lights_path ' .. lights_path .. ' --save_path ' .. write_path
    if not verbose then
        local log_file = paths.concat(write_path, 'log.txt')
        command = command .. ' > ' .. log_file
    end

    print(command)
    os.execute(command)
end

function Render:__read_images(load_path, num_lights)
    -- 0-indexing in python
    local img = image.load( paths.concat(load_path, '0.png') )
    local size = img:size()
    local images = torch.Tensor(num_lights, size[1], size[2], size[3])
    
    for ind = 0, num_lights - 1 do 
        local img = image.load( paths.concat(load_path, ind .. '.png'))
        images[ind+1] = img
    end

    return images
end

