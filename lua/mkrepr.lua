require 'paths'
require 'image'

channels = 3
dim = 256

savedir = 'saved/run_0.1/'

images = {{235, 4}, {247, 9}, {233, 4}, {237, 5}, {243, 2}, {246, 4}, {236, 2}, {253, 4}, {252, 8}, {252, 4}}

for ind, tbl in pairs(images) do
    local epoch = tbl[1]
    local pos = tbl[2]
    local file = paths.concat(savedir, epoch .. '.png')
    local img = image.load(file)
    print(img:size())
    print(epoch, dim)
    local selection = img[{{},{(pos-1)*dim+1,pos*dim},{}}]
    grid = grid or torch.Tensor(channels, dim*#images, img:size()[3])
    grid[{{},{(ind-1)*dim+1,ind*dim},{}}] = selection
end

image.save('saved/grid.png', grid)