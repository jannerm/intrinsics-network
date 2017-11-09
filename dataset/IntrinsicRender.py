import sys, os, math, time, random, subprocess
import numpy as np
import bpy

class IntrinsicRender:

    def __init__(self, x_res, y_res, use_nodes=True):
        self.__toggleNodes(use_nodes)
        bpy.data.scenes['Scene'].render.layers['RenderLayer'].use_pass_normal = True
        if use_nodes:
            self.tree = bpy.context.scene.node_tree
            self.__addNode('CompositorNodeNormalize')
            self.changeResolution(x_res, y_res)
            self.__initNormals()
            self.__initNormalsMaterial()
            self.__initBackground()

    def __initNormals(self):
        render = self.tree.nodes['Render Layers']
        mix1 = self.__addNode('CompositorNodeMixRGB')
        mix2 = self.__addNode('CompositorNodeMixRGB')
        invert = self.__addNode('CompositorNodeInvert')
        mix1.blend_type = 'MULTIPLY'
        mix2.blend_type = 'ADD'
        mix1.inputs[2].default_value = [.5, .5, .5, 1.]
        mix2.inputs[2].default_value = [.5, .5, .5, 1.]
        self.__linkNodes(render, mix1, out=3, inp=1)
        self.__linkNodes(mix1, mix2, inp=1)
        self.__linkNodes(mix2, invert, inp=1)

    def __initNormalsMaterial(self, img_path = 'materials/normals_vector.png'):
        folder = os.path.dirname(os.path.realpath(__file__))
        fullpath = os.path.join(folder, img_path)
        # img_path = os.path.join(os.path.dirname)
        # print('PATH:', os.path.realpath(__file__))
        print('PATH: ', fullpath)
        mat = bpy.data.materials.new('Normals')
        mat.use_shadeless = True
        tex = bpy.data.textures.new('Normals', type='IMAGE')
        img = bpy.data.images.load(fullpath)
        tex.image = img
        slot = mat.texture_slots.add()
        slot.texture = tex
        slot.texture_coords = 'NORMAL'


    def __initBackground(self):
        alpha = self.__addNode('CompositorNodeAlphaOver')
        background = self.__addNode('CompositorNodeMixRGB')
        render = self.tree.nodes['Render Layers']
        invert = self.tree.nodes['Invert']
        self.__linkNodes(render, alpha, out=1, inp=2)
        self.__linkNodes(invert, alpha, inp=1)
        self.__linkNodes(alpha, background)
        self.__linkNodes(invert, background, inp=2)
        background.inputs[1].default_value = [0., 0., 0., 1.]
        # bpy.data.scenes['Scene'].render.alpha_mode = 'TRANSPARENT'

    def changeResolution(self, x_res, y_res):
        bpy.data.scenes['Scene'].render.resolution_x = x_res
        bpy.data.scenes['Scene'].render.resolution_y = y_res

    # CompositorNodeNormalize
    def __addNode(self, name):
        node = self.tree.nodes.new(name)
        return node

    def __linkNodes(self, node1, node2, out=0, inp=0):
        tree = bpy.context.scene.node_tree
        tree.links.new(node1.outputs[out], node2.inputs[inp])

    def __toggleNodes(self, boolean):
        bpy.data.scenes['Scene'].use_nodes = boolean

    def __shadeless(self, boolean):
        for mat in bpy.data.materials:
            mat.use_shadeless = boolean

    def changeMode(self, mode, name='shape'):
        if mode == 'composite':
            self.composite()
        elif mode == 'albedo':
            self.albedo()
        elif mode == 'depth':
            self.depth()
            bpy.data.objects['Plane'].hide_render = False
        elif mode == 'depth_hires':
            self.depth(normalize=False)
            self.__filetype('exr')
        elif mode == 'normals':
            self.normals(name)
            bpy.data.objects['Plane'].hide_render = True
            self.__filetype('png')
        elif mode == 'shading':
            self.shading(name = name)
        elif mode == 'mask':
            self.mask(name = name)
        elif mode == 'specular':
            self.specular(name = name)
        elif mode == 'lights':
            self.lights(name)
        else:
            RuntimeError('Mode not recognized: ' + mode)

    def composite(self):
        bpy.data.scenes['Scene'].render.alpha_mode = 'SKY'
        self.__original(True)
        self.__shadeless(False)
        render = self.tree.nodes['Render Layers']
        composite = self.tree.nodes['Composite']
        self.__linkNodes(render, composite)

    def albedo(self):
        self.__original(True)
        self.__shadeless(True)
        render = self.tree.nodes['Render Layers']
        composite = self.tree.nodes['Composite']
        self.__linkNodes(render, composite)

    def __filetype(self, extension):
        if extension == 'png':
            bpy.context.scene.render.image_settings.file_format = 'PNG'
        elif extension == 'exr':
            bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
            bpy.context.scene.render.image_settings.color_depth='32'
        else:
            raise RuntimeError('Unrecognized filetpye: ', extension)

    def depth(self, normalize=True):
        self.__original(True)
        render = self.tree.nodes['Render Layers']
        composite = self.tree.nodes['Composite']
        if normalize:
            norm = self.tree.nodes['Normalize']
            self.__linkNodes(render, norm, out=2)
            self.__linkNodes(norm, composite)
        else:
            self.__linkNodes(render, composite, out=2)

    # def normals(self):
    #     self.__original(True)
    #     self.__normalsPass(True)

    def normals(self, name):
        bpy.data.objects[name].hide_render = True
        bpy.data.objects[name+'_shading'].hide_render = True
        bpy.data.objects[name+'_normals'].hide_render = False
        if name != 'sphere':
            bpy.data.objects['sphere'].hide_render = True
        # self.__original(True)
        self.__normalsPass(False)
        mat = bpy.data.materials['Normals']
        obj = bpy.data.objects[name+'_normals']
        for ind in range(len(obj.data.materials)):
            obj.data.materials[ind] = mat



    def __normalsPass(self, boolean):
        render = self.tree.nodes['Render Layers']
        background = self.tree.nodes['Mix.002']
        # invert = self.tree.nodes['Invert']
        composite = self.tree.nodes['Composite']
        if boolean:
            bpy.data.scenes['Scene'].render.alpha_mode = 'TRANSPARENT'
            self.__linkNodes(background, composite)
        else:
            bpy.data.scenes['Scene'].render.alpha_mode = 'SKY'
            self.__linkNodes(render, composite)

    def lights(self, name):
        if name in bpy.data.objects:
            bpy.data.objects[name].hide_render = True
            bpy.data.objects[name + '_shading'].hide_render = True
            bpy.data.objects[name + '_normals'].hide_render = True
        bpy.data.objects['sphere'].hide_render = False
        # bpy.data.materials['sphere'].use_shadeless = False
        self.__color( label='lights', obj_name='sphere')
        self.__normalsPass(False)
        bpy.data.scenes['Scene'].render.alpha_mode = 'TRANSPARENT'


    def shading(self, name='shape'):
        self.__original(False, name=name)
        self.__normalsPass(False)
        # self.__shadeless(False)
        render = self.tree.nodes['Render Layers']
        composite = self.tree.nodes['Composite']
        self.__linkNodes(render, composite)
        self.__color( label = 'shading', obj_name = name + '_shading' )

    def mask(self, name='shape'):
        self.__original(False, name=name)
        self.__normalsPass(False)
        # self.__shadeless(True)
        render = self.tree.nodes['Render Layers']
        composite = self.tree.nodes['Composite']
        self.__linkNodes(render, composite)
        self.__color( label='mask', obj_name=name+'_shading', shadeless=True, preserve_transparency=False )

    def specular(self, name='shape'):
        self.__original(False, name=name)
        self.__color( label='specular', obj_name=name+'_shading', diffuse_intensity=0, specular_intensity=1.0, emit = 0)

    def __color(self, label='shading', obj_name='shape_shading', diffuse_intensity=0.8, specular_intensity=0.5, emit = 0.05, shadeless=False, preserve_transparency=True):
        obj = bpy.data.objects[obj_name]
        for ind in range(len(obj.data.materials)):
            old = obj.data.materials[ind]
            trans = old.use_transparency and preserve_transparency
            mat = bpy.data.materials.new(label + '_' + str(ind))
            mat.diffuse_intensity = diffuse_intensity
            mat.specular_intensity = specular_intensity
            mat.diffuse_color = (1,1,1)
            mat.specular_color = (1,1,1)
            mat.emit = emit
            mat.use_shadeless = shadeless
            mat.use_transparency = trans
            if trans:
                mat.alpha = old.alpha
            obj.data.materials[ind] = mat


    ''' Show original mesh 
    or duplicated mesh '''
    def __original(self, boolean, name='shape'):
        if boolean:
            obj = bpy.data.objects[name].hide_render = False
            obj = bpy.data.objects[name + '_shading'].hide_render = True
            obj = bpy.data.objects[name + '_normals'].hide_render = True
            obj = bpy.data.objects['sphere'].hide_render = True
        else:
            obj = bpy.data.objects[name].hide_render = True
            obj = bpy.data.objects[name + '_shading'].hide_render = False
            obj = bpy.data.objects[name + '_normals'].hide_render = True
            obj = bpy.data.objects['sphere'].hide_render = True