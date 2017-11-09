import sys, os, math, time, random, subprocess
import numpy as np
import bpy

class ShapeNetRender:

    def __init__(self, shapenet_path, staging_path, write_path, create=False):
        self.shapenet_path = shapenet_path
        self.staging_path = staging_path
        self.write_path = write_path
        if create:
            subprocess.call(['mkdir', staging_path])
            subprocess.call(['mkdir', write_path])


    #### Loading        
    def load(self, category, name='shape'):
        if category == 'random':
            category = self.__getSubdirectories(self.shapenet_path)
        category_path = os.path.join(self.shapenet_path, category)
        imported = False
        while not imported:
            subprocess.call(['mkdir', self.staging_path])
            shape = self.__choose(category_path)
            self.__copy( category_path, shape, self.staging_path )
            self.__import( os.path.join(self.staging_path, shape, 'model.obj') )
            imported = self.__join() ## returns True only if join was successful
            subprocess.call(['rm', '-r', self.staging_path])
        self.__subsurf()
        self.__rename('shape', name)
        print('DONE RENAMING')

    def __rename(self, old, new):
        # for obj in bpy.data.objects:
            # print(obj.name)
        bpy.data.objects[old].name = new

    def __getSubdirectories(self, base):
        return [folder for folder in os.listdir(base) if os.path.isdir(os.path.join(base, folder))]

    ## category is absolute path to shapenet class 
    def __choose(self, category):
        models = self.__getSubdirectories(category)
        selection = random.choice(models)
        print(os.path.join(self.shapenet_path, selection, 'model.obj'))
        return selection if os.path.exists(os.path.join(category, selection, 'model.obj')) else self.__choose(category)

    def __copy(self, category_path, shape, newPath):
        print('COPING NEWPATH')
        # print(oldPath)
        print(newPath)
        subprocess.call(['cp', '-r', os.path.join(category_path, shape), newPath])
        textures = [i for i in os.listdir(os.path.join(newPath, shape)) if '.mtl' in i]
        print(newPath)
        for tex in textures:
            print('MOVING ', tex)
            subprocess.call(['mv', os.path.join(newPath, shape, tex), os.path.join(newPath, shape, tex.split('_tmp')[0])])

    def __import(self, path):
        bpy.ops.import_scene.obj(filepath = path)

    def __subsurf(self):
        self.__select(lambda x: x.name == 'shape')
        bpy.ops.object.modifier_add(type='SUBSURF')
        bpy.data.objects['shape'].modifiers['Subsurf'].levels = 1
        bpy.data.objects['shape'].modifiers['Subsurf'].render_levels = 1
        bpy.data.objects['shape'].modifiers['Subsurf'].subdivision_type = 'SIMPLE'

    def __select(self, function):
        for obj in bpy.data.objects:
            if function(obj):
                obj.select = True
            else:
                obj.select = False


    #### Modifying
    def __join(self):
        for obj in bpy.data.objects:
            if 'mesh' in obj.name or 'Mesh' in obj.name:
                obj.select = True
                bpy.context.scene.objects.active = obj
                obj.name = 'mesh'
            else:
                obj.select = False
        try: 
            bpy.ops.object.join()
            bpy.data.objects['mesh'].name = 'shape'
            return True
        except RuntimeError:
            return False

    def __orient(self, size, coords, rotation):
        obj = bpy.data.objects['shape']
        largest_dim = max(obj.dimensions)
        scale = size / largest_dim
        for dim in range(3):
            obj.scale[dim] = scale
            obj.location[dim] = coords[dim] 
            obj.rotation_euler[dim] = toRadians(rotation[dim])

    def __validate(self, scale, coords, rotation):
        for obj in bpy.data.objects:
            if 'mesh' in obj.name:
                for dim in range(3):
                    if obj.dimensions[dim] > 1:
                        __orient(scale / obj.dimensions[dim], coords, rotation)

    #### Deleting
    def __deleteSubShapes(self):
        for obj in bpy.data.objects:
            if 'mesh' in obj.name or 'Mesh' in obj.name:
                # if random.random() < 0.25:
                obj.select = True
                bpy.context.scene.objects.active = obj
                obj.name = 'mesh'
            else:
                obj.select = False