require 'paths'
require 'image'

set_path = 'output/car_normals_emit_0.05'
set_size = 20000

for ind = 0, set_size-1 do
    xlua.progress(ind, set_size-1)
    local img = image.load( paths.concat(set_path, ind .. '_normals.png'))[{{1,3}}]
    -- local mask = img:sum(1):le(.001):repeatTensor(3,1,1)
    local mask = img:eq(0):sum(1):gt(0):repeatTensor(3,1,1)
    local norm = img:pow(2):sum(1):squeeze():repeatTensor(3,1,1)
    -- print(img:size())
    -- print(norm:size())
    local normed = img:cdiv(norm) * 1.5
    normed[mask] = 0
    image.save( paths.concat(set_path, ind .. '_normals_norm.png'), normed )
end