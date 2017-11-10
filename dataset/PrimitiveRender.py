import bpy, numpy as np

class PrimitiveRender:

    def __init__(self):
        self.categories = ['cube', 'sphere', 'cylinder', 'cone', 'torus', 'suzanne', 'bunny', 'teapot']

    def load(self, category):
        assert category in self.categories

        if category == 'cube':
            bpy.ops.mesh.primitive_cube_add()
            bpy.data.objects['Cube'].name = 'shape'
        elif category == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(ring_count=128,segments=256)
            bpy.data.objects['Sphere'].name = 'shape'
        elif category == 'cylinder':
            bpy.ops.mesh.primitive_cylinder_add(vertices=256)
            bpy.data.objects['Cylinder'].name = 'shape'
        elif category == 'cone':
            bpy.ops.mesh.primitive_cone_add(vertices=256)
            bpy.data.objects['Cone'].name = 'shape'
        elif category == 'torus':
            bpy.ops.mesh.primitive_torus_add(major_segments=512, minor_segments=512)
            for obj in bpy.data.objects:
                name = obj.name
                if 'Torus' in name:
                    print('NAME: ', name)
                    bpy.data.objects[name].name = 'shape'
        elif category == 'suzanne':
            bpy.ops.mesh.primitive_monkey_add()
            bpy.data.objects['Suzanne'].name = 'shape'
            bpy.ops.object.shade_smooth()
        elif category == 'bunny':
            bpy.ops.import_mesh.stl(filepath='objects/bunny.stl')
            for obj in bpy.data.objects:
                name = obj.name
                print('NAME: ', name)
                if 'Bunny' in name:
                    bpy.data.objects[name].name = 'shape'
        elif category == 'teapot':
            # bpy.ops.import_scene.obj(filepath='objects/teapot.obj')
            bpy.ops.import_mesh.stl(filepath='objects/teapot.stl')
            bpy.data.objects['Teapot'].name = 'shape'
            bpy.ops.object.shade_smooth()
        else:
            raise RuntimeError

        if 'mat' not in bpy.data.materials:
            bpy.data.materials.new('mat')
        bpy.data.materials['mat'].diffuse_color = np.random.rand(3).tolist()
        # while len(bpy.data.objects['shape'].data.materials) == 0:
            # print('ADDING MATERIAL')
        # bpy.data.objects['shape'].select=True
        bpy.ops.object.material_slot_add()
        # if len(bpy.data.objects['shape'].data.materials) == 0:
            # bpy.data.objects['shape'].data.materials.append(bpy.data.materials['mat'])
        # else:
        # bpy.ops.material.new()
        # for mat in bpy.data.objects['shape'].data.materials:
            # print('MAT: ', mat)
        bpy.data.objects['shape'].data.materials[0] = bpy.data.materials['mat']

