function make_grid(inputs, nrow)
  local inputs = inputs
  if type(inputs) ~= 'table' then
    inputs = tensorToTable(inputs)
  end
  local num_img = #inputs
  local size = inputs[1]:size()
  local ch, m, n = size[1], size[2], size[3]
  local grid = torch.Tensor(ch, m*torch.ceil(num_img/nrow), n*nrow )
  for ind = 1, num_img do
    local i = torch.floor((ind-1) / nrow)
    local j = (ind-1) % nrow
    grid[{{},{i*m+1,(i+1)*m},{j*n+1,(j+1)*n}}] = inputs[ind]
  end
  return grid
end

function tensorToTable(tensor)
  local plot = {}
  for i = 1, tensor:size()[1] do
      local slice = tensor[i]:float()
      table.insert(plot, slice)
  end
  return plot
end