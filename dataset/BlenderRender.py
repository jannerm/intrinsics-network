import sys, os, math, time, random, subprocess
import numpy as np
import bpy

class BlenderRender:

    def __init__(self, gpu, spotlight=False):
        if gpu:
            self.switchToGPU()
        self.delete(lambda x: x.name != 'Camera')
        self.translate('Camera', [0, -8.5, 5])
        self.rotate('Camera', [60, 0, 0])
        if spotlight:
            bpy.ops.object.lamp_add(type = 'SPOT')
            bpy.data.lamps['Spot'].energy = 5
            bpy.data.lamps['Spot'].spot_size = self.__toRadians(90)
        else:
            bpy.ops.object.lamp_add(type='POINT')
            self.translate('Point', [0, -10, 3])
            bpy.data.lamps['Point'].use_specular = True
        bpy.data.worlds['World'].horizon_color = (0, 0, 0)
        bpy.data.scenes['Scene'].render.resolution_percentage = 100
        self.__wall()

    def write(self, path, name, extension = 'png'):
        bpy.context.scene.render.filepath = os.path.join(path, name + '.' + extension)
        bpy.ops.render.render(write_still = True)

    def switchToGPU(self, vebose = True):
        if verbose:
            print('before changing settings: ', bpy.context.scene.cycles.device)
        bpy.data.scenes['Scene'].cycles.samples=20
        bpy.data.scenes["Scene"].render.tile_x=256
        bpy.data.scenes["Scene"].render.tile_y=256
        bpy.data.scenes['Scene'].cycles.max_bounces=5
        bpy.data.scenes['Scene'].cycles.caustics_reflective=False
        bpy.data.scenes['Scene'].cycles.caustics_refractive=False
        if verbose:
            print('max bounces: ', bpy.data.scenes['Scene'].cycles.max_bounces)
            print('Samples: ', bpy.data.scenes['Scene'].cycles.samples)
        bpy.data.scenes["Scene"].render.engine='CYCLES'
        bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        bpy.data.scenes["Scene"].cycles.device='GPU'

    def __wall(self):
        bpy.ops.mesh.primitive_plane_add()
        bpy.data.objects['Plane'].location = [0,10,0]
        bpy.data.objects['Plane'].rotation_euler = [math.pi/3.,0,0]
        bpy.data.objects['Plane'].scale = [20,20,20]
        bpy.data.objects['Plane'].hide_render = True

    def hideAll(self):
        for obj in bpy.data.objects:
            obj.hide_render = True

    def __scale(self, name, size):
        obj = bpy.data.objects[name]
        for dim in range(3):
            obj.scale[dim] = size

    def __ensureList(self, inp):
        if type(inp) != list:
            inp = [inp]
        return inp

    def resize(self, names, size, dim=0):
        names = self.__ensureList(names)
        if type(names) == str:
            names = [names]
        for name in names:
            obj = bpy.data.objects[name]
            obj.dimensions[dim] = size
            scale = obj.scale[dim]
            self.__scale(name, scale)

    def translate(self, names, coords):
        names = self.__ensureList(names)
        for name in names:
            obj = bpy.data.objects[name]
            for dim in range(3):
                obj.location[dim] = coords[dim]

    def rotate(self, names, angles):
        names = self.__ensureList(names)
        for name in names:
            obj = bpy.data.objects[name]
            for dim in range(3):
                obj.rotation_euler[dim] = self.__toRadians(angles[dim])

    def delete(self, function):
        for obj in bpy.data.objects:
            if function(obj):
                obj.select = True
            else:
                obj.select = False
        bpy.ops.object.delete()

    def __toRadians(self, degree):
        return degree * math.pi / 180.

    def random(self, low, high):
        if type(high) == list:
            params = [np.random.uniform(low=low[ind], high=high[ind]) for ind in range(len(high))]
            return params
        else:
            return np.random.uniform(low=low, high=high)

    def light_spherical(self, rho, phi, theta):
        x = rho * math.sin(phi) * math.cos(theta)
        y = rho * math.sin(phi) * math.sin(theta)
        z = rho * math.cos(phi)
        self.translate('Point', [x, y, z])

    def light(self, energy, pos):
        bpy.data.lamps['Point'].energy = energy
        self.translate('Point', pos)

    def __select(self, function):
        for obj in bpy.data.objects:
            if function(obj):
                obj.select = True
            else:
                obj.select = False

    def duplicate(self, old, new):
        self.__select(lambda x: x.name == old)
        bpy.ops.object.duplicate()
        obj = bpy.data.objects[old + '.001']
        obj.name = new

    def sphere(self, location, scale, label = 'shape'):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=200, ring_count=200, location=location, size=scale)
        bpy.data.objects['Sphere'].name = label
        if 'sphere' not in bpy.data.materials:
            bpy.data.materials.new('sphere')
        bpy.ops.object.material_slot_add()
        bpy.data.objects[label].data.materials[0] = bpy.data.materials['sphere']

    def spotlight(self, x, y, z, rot_x, rot_y, rot_z):
        spot = bpy.data.objects['Spot']
        spot.location = [x, y, z]
        spot.rotation_euler = [self.__toRadians(rot_x), self.__toRadians(rot_y), self.__toRadians(rot_z)]




