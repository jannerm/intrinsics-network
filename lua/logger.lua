require 'paths'

logger = {}

function logger:init(filename)
    local file = io.open(filename .. '.log', 'w')
    file:write(filename .. '\n')
    file:close()
end

function logger:add(filename, val)
    local file = io.open(filename .. '.log', 'a')
    file:write(val .. '\n')
    file:close()
end

-- logger:init('test')
-- logger:add('test', 1)
-- logger:add('test', 2)
-- logger:add('test', 3)