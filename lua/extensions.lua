function string.split(str, sep)
    local sep, fields = sep or ":", {}
    local pattern = string.format("([^%s]+)", sep)
    str:gsub(pattern, function(c) fields[#fields+1] = c end)
    return fields
end

function table.join(tbl1, tbl2)
    local joined = {}
    table.insert_all(joined, tbl1)
    table.insert_all(joined, tbl2)
    return joined
end

-- inserts all in tbl2 to tbl1
function table.insert_all(tbl1, tbl2)
    for _, val in tbl2 do
        table.insert(tbl1, val)
    end
    return tbl1
end

function table.tonumber(tbl)
    local new = {}
    for key, val in pairs(tbl) do
        new[key] = tonumber(val)
    end
    return new
end

function table.sum(tbl)
    local sum = 0
    for _, val in pairs(tbl) do
        sum = sum + val
    end
    return sum
end

function table.cmul(tbl1, tbl2)
    for key, _ in pairs(tbl1) do
        tbl1[key] = torch.cmul(tbl1[key], tbl2[key])
    end
    return tbl1
end

function table.fill(tbl, start, finish, item)
    for key, value in pairs(tbl) do
        value[{{start, finish}}]:fill(item)
    end
    return tbl
end

function table.constmul(tbl1, tbl2)
    for key, val in pairs(tbl1) do
        tbl1[key] = val*tbl2[key]
    end
    return tbl1
end

function table.cat(tbl1, tbl2, index)
    local new = {}
    for key, _ in pairs(tbl1) do
        new[key] = torch.cat(tbl1[key], tbl2[key], index)
    end
    return new
end

function table.span(tbl, start, finish)
    local new = {}
    for key, val in pairs(tbl) do
        new[key] = val[{{start, finish}}]
    end
    return new
end

function table.index(tbl, dim, inds)
    local new = {}
    for key, val in pairs(tbl) do
        new[key] = val:index(dim, inds)
    end
    return new
end

function table.clone(tbl)
    local new = {}
    for key, val in pairs(tbl) do
        new[key] = val:clone()
    end
    return new
end

function table.double(tbl)
    local new = {}
    for key, val in pairs(tbl) do
        new[key] = val:double()
    end
    return new
end

function table.cuda(tbl)
    local new = {}
    for key, val in pairs(tbl) do
        new[key] = val:cuda()
    end
    return new
end

function table.criterion(criterion, inputs, targets)
    assert(#inputs == #targets, 'Unequal number of inputs and targets')
    local errors = {}
    local gradients = {}
    for i = 1, #inputs do
        local inp = inputs[i]
        local targ = targets[i]
        local err = criterion:forward(inp, targ)
        local grad = criterion:backward(inp, targ):clone()
        table.insert(errors, err)
        table.insert(gradients, grad)
    end
    return errors, gradients
end

function table.select(tbl, inds)
    local new = {}
    for i, ind in pairs(inds) do
        new[i] = tbl[ind]
    end
    return new
end

function table.mul(tbl, scalar)
    for key, val in pairs(tbl) do
        tbl[key] = val*scalar
    end
    return tbl
end

function table.slice(tbl, start, finish)
    local new = {}
    for ind = start, finish do
        table.insert(new, tbl[ind])
    end
    return new
end

function table.contains(tbl, needle)
    for _, val in pairs(tbl) do
        if val == needle then
            return true
        end
    end
    return false
end

random = {}

function random.choice(tensor, selection_size)
    local size = tensor:size()[1]
    assert(selection_size <= size, 'random.choice: Selecting more elements than are in input')
    local inds = torch.randperm(size)[{{1,selection_size}}]:long()
    local selection = tensor:index(1, inds)
    return selection
end



