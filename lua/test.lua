cmd = torch.CmdLine()
cmd:option('-options', '')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- loadstring('opt.options='..opt.options)()
print(opt)
-- print(opt.options)

function justWords(str)
  local t = {}
  local function helper(word) table.insert(t, word) return "" end
  if not str:gsub("%w+", helper):find"%S" then return t end
end

function string.split(str, sep)
        local sep, fields = sep or ":", {}
        local pattern = string.format("([^%s]+)", sep)
        str:gsub(pattern, function(c) fields[#fields+1] = c end)
        return fields
end

print(string.split('a,b,c', ',') )
