import os, shutil, subprocess, numpy as np, scipy.misc, torch
from torch.autograd import Variable

class Render:

    def vis_lights(self, lights, verbose = False):
        if type(lights) != np.ndarray:
            lights = lights.data.cpu().numpy() 

        write_path = 'temp_path_' + str(np.random.rand())
        lights_path = os.path.join(write_path, 'lights.npy')
        self.__mkdir(write_path)
        np.save(lights_path, lights)

        num_lights = lights.shape[0]
        print 'Rendering {} lights...'.format(num_lights)
        self.__blender(lights_path, write_path, verbose)

        images = self.__read_images(write_path, num_lights)
        if verbose:
            print 'Deleting {}\n'.format(write_path)
        self.__rmdir(write_path)
        return Variable( torch.Tensor(images.transpose(0,3,1,2)[:,:3]) )

    def __mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def __rmdir(self, path):
        subprocess.Popen(['rm', '-r', path])

    def __blender(self, lights_path, write_path, verbose):
        # script_path = '../dataset/vis_lights.py'
        script_path = '/om/user/janner/mit/urop/intrinsic/dataset/vis_lights.py'
        command =   [   '/om/user/janner/blender-2.76b-linux-glibc211-x86_64/blender', 
                        '--background', '-noaudio', '--python', script_path, '--', \
                        '--lights_path', lights_path, '--save_path', write_path]
        if verbose:
            stdout = None
            print command
        else:
            stdout = open( os.path.join(write_path, 'log.txt'), 'w')

        subprocess.call(command, stdout=stdout)

    def __read_images(self, load_path, num_lights):
        img = scipy.misc.imread( os.path.join(load_path, '0.png') )
        M, N, C = img.shape
        images = np.zeros( (num_lights, M, N, C) )

        for ind in range(num_lights):
            img = scipy.misc.imread( os.path.join(load_path, str(ind) + '.png') ) / 255.
            images[ind] = img

        return images




