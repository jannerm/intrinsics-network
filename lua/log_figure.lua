require 'paths'
require 'lfs'
require 'optim'

cmd = torch.CmdLine()
cmd:option('-directory', '')
cmd:option('-verbose', false)
opt = cmd:parse(arg)

function find_logs(directory)
    local logs = {}
    for file in lfs.dir(directory) do
        if string.find(file, '.log') ~= nil and string.find(file, '.eps') == nil and string.find(file, '.copy') == nil then
            -- print(file)
            table.insert(logs, file)
        end
    end
    return logs
end

function plot_log(directory, filename)
    local original = paths.concat(directory, filename)
    local modified = paths.concat(directory, filename .. '.copy')
    local logger = optim.Logger(modified)
    local f = io.open(original, 'r')
    local label = f:read()
    local val = f:read()
    while val ~= nil do
        logger:add{[label] = val}
        val = f:read()
    end
    f:close()
    logger:style{[label] = '-'} 
    logger:plot()
end

function plot_all(directory, verbose)
    local logs = find_logs(directory)
    if opt.verbose then
        print(logs)
    end
    for _, log in pairs(logs) do
        plot_log(directory, log)
    end
end

if #opt.directory == 0 then
    error('No directory specified')
end
-- local base = 'saved/mask_0,0,1,.1_0.001/'
plot_all(opt.directory, opt.verbose) 
-- local logs = find_logs(base)
-- print(logs)

-- for _, log in pairs(logs) do
--     -- local path = paths.concat(base, log)
--     -- print(path)
--     plot_log(base, log)
-- end