require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'

local Render, parent = torch.class("nn.Render", "nn.Module")

function Render:__init(channels, lights, shader)
   parent.__init(self)
   self.channels = channels
   self.lights = lights
   self.shader = shader

   self.composer = self:__compose(channels, lights, shader)
end

function Render:__compose(channels, lights, shader)
    local inp = nn.Identity()()
    local ch = channels(inp)
    local split = nn.SplitTable(2)(ch)

    local albedo = nn.NarrowTable(1,3)(split)
    albedo = nn.MapTable():add(nn.Unsqueeze(2))(albedo)
    albedo = nn.JoinTable(1,3)(albedo)

    local specular = nn.NarrowTable(4,1)(split)
    -- specular = nn.MapTable():add(nn.Unsqueeze(2))(specular)
    specular = nn.JoinTable(1,3)(specular)
    specular = nn.Replicate(3,2)(specular)

    local normals = nn.NarrowTable(5,3)(split)
    normals = nn.MapTable():add(nn.Unsqueeze(2))(normals)
    normals = nn.JoinTable(1,3)(normals)

    local lights = lights(inp)

    local shading = shader({normals, lights})
    shading = nn.Squeeze()(shading)
    shading = nn.Replicate(3,2)(shading)

    local render = nn.CMulTable()({albedo, shading})
    render = nn.CAddTable()({render, specular})
    -- albedo, specular, normals, lights, shading
    local model = nn.gModule({inp}, {render, albedo, specular, normals, shading})
    return model
end

function Render:updateOutput(input)
    return self.composer:forward(input)
end
function Render:updateGradInput(input, gradOutput)
    self.composer:updateGradInput(input, gradOutput)
end

-- function CAdd:updateOutput(input)
--     local ch = self.channels:forward(input)
--     local li = self.lights:forward(input)

--    self._output = self._output or input.new()
--    self._bias = self._bias or input.new()
--    self._expand = self._expand or input.new()
--    self._repeat = self._repeat or input.new()

--    self.output:resizeAs(input):copy(input)
--    if input:nElement() == self.bias:nElement() then
--       self.output:add(self.bias)
--    else
--       if self.bias:dim() == input:dim() then
--          self._output:set(self.output)
--          self._bias:set(self.bias)
--       else
--          local batchSize = input:size(1)
--          self._output:view(self.output, batchSize, -1)
--          self._bias:view(self.bias, 1, -1)
--       end

--       self._expand:expandAs(self._bias, self._output)

--       --expandAs uses stride 0 and self._expand is not contiguous
--       --cuda ops may assume contiguous input
--       if torch.type(input) == 'torch.CudaTensor' then
--          self._repeat:resizeAs(self._expand):copy(self._expand)
--          self._output:add(self._repeat)
--       else
--          self._output:add(self._expand)
--       end
--    end

--    return self.output
-- end

-- function CAdd:updateGradInput(input, gradOutput)
--    self.gradInput = self.gradInput or input.new()
--    self.gradInput:resizeAs(gradOutput):copy(gradOutput)

--    return self.gradInput
-- end

-- function CAdd:accGradParameters(input, gradOutput, scale)
--    scale = scale or 1

--    self._gradBias = self._gradBias or gradOutput.new()
--    self._gradOutput = self._gradOutput or gradOutput.new()
--    self._repeat = self._repeat or gradOutput.new()

--    if self.bias:nElement() == gradOutput:nElement() then
--       self.gradBias:add(scale, gradOutput)
--    else
--       if self.bias:dim() == gradOutput:dim() then
--          self._gradBias:set(self.gradBias)
--          self._gradOutput:set(gradOutput)
--       else
--          local batchSize = input:size(1)
--          self._gradBias:view(self.gradBias, 1, -1)
--          self._gradOutput:view(gradOutput, batchSize, -1)
--       end

--       self._gradBias:expandAs(self._gradBias, self._gradOutput)

--       --expandAs uses stride 0 and self._gradBias is not contiguous
--       --cuda ops may assume contiguous input
--       if torch.type(self._gradBias) == 'torch.CudaTensor' then
--          self._repeat:resizeAs(self._gradBias):copy(self._gradBias)
--          self._repeat:add(scale, self._gradOutput)
--          self._gradBias:copy(self._repeat)
--       else
--          self._gradBias:add(scale, self._gradOutput)
--       end
--    end
-- end


-- local inp = nn.Identity()()
-- local conv = nn.SpatialConvolution(3,3,3,3,1,1)(inp)
-- local model = nn.gModule({inp}, {conv})

-- input = torch.randn(5,3,10,10)
-- output = model:forward(input)
-- print(input:size(), output:size())

-- local i2 = nn.Identity()()
-- local out = model(i2)
-- local m2 = nn.gModule({i2}, {out})

-- output = m2:forward(input)
-- print(output:size())


