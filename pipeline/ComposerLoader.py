import os, torch, torch.utils.data, scipy.misc, numpy as np, pdb
import utils

class ComposerDataset(torch.utils.data.Dataset):
    def __init__(self, directory, unlabeled_datasets, labeled_datasets, size_per_dataset=10000, inds=None, unlabeled_array='shader2', labeled_array='shader2'):
        self.directory = directory

        self.size_per_dataset = size_per_dataset
        self.unlabeled_array = unlabeled_array
        self.labeled_array = labeled_array

        self.unlabeled_datasets = [os.path.join(directory, dataset) for dataset in unlabeled_datasets.split(',')]
        self.labeled_datasets = [os.path.join(directory, dataset) for dataset in labeled_datasets.split(',')]

        print '<Loader> Unlabeled: ', self.unlabeled_datasets
        print '<Loader> Labeled: ', self.labeled_datasets

        self.unlabeled_selections = ['input', 'mask']
        self.labeled_selections = ['input', 'mask', 'albedo', 'depth', 'normals', 'lights', 'shading']

        self.unlabeled_data_files = self.__find_dataset(self.unlabeled_datasets, self.unlabeled_selections, size_per_dataset, inds, self.unlabeled_array)
        self.labeled_data_files = self.__find_dataset(self.labeled_datasets, self.labeled_selections, size_per_dataset, inds, self.labeled_array)
        
        self.labeled_size = self.__selection_len(self.labeled_data_files)


    def __find_dataset(self, datasets, selections, size_per_dataset, inds, array):
        data_files = []
        for dataset in datasets:
            self.set_specific = []
            for sel in selections:
                if sel == 'lights':
                    files = np.load('../dataset/arrays/' + array + '.npy')[:size_per_dataset,:]
                    assert files.shape[0] == size_per_dataset
                ## reflectance, shading
                elif sel == 'input':
                    files = {}
                    files['reflectance'] = self.__find_sort_files(dataset, 'albedo')[:size_per_dataset]
                    files['shading'] = self.__find_sort_files(dataset, 'shading')[:size_per_dataset]
                    assert len(files['reflectance']) == size_per_dataset and len(files['shading']) == size_per_dataset
                else:
                    files = self.__find_sort_files(dataset, sel)[:size_per_dataset] 
                    assert len(files) == size_per_dataset        
                self.set_specific.append(files)
            data_files.append(self.set_specific)
        
        ## merge set lists
        data_files = [self.__merge(lists) for lists in zip(*data_files)]

        if inds:
            inds = [i+(offset*self.size_per_dataset) for offset in range(len(datasets)) for i in inds]
            print '<Loader> Indices: ', inds
            for ind, sel in enumerate(selections):
                files = data_files[ind]
                if sel == 'lights':
                    data_files[ind] = files[inds]
                elif sel == 'input':
                    data_files[ind] = {key: [files[key][i] for i in inds] for key in data_files[ind].keys()}
                else:
                    data_files[ind] = [files[i] for i in inds]

        return data_files

    def __find_sort_files(self, dataset, sel):
        files = [fname for fname in os.listdir(dataset) if sel + '.png' in fname and 'sphere' not in fname]
        files = sorted(files, key=lambda fname: int(fname.split('_')[0]) ) 
        files = [os.path.join(dataset, fname) for fname in files]  
        return files

    ## merge selection from different datasets,
    ## e.g., merge all reflectance images from 
    ## motorbikes, airplanes, and bottles

    ## inputs are represented as dicts with reflectance and shading filenames
    ## lights are represented as np arrays
    ## other intrinsic images as list of filenames 
    def __merge(self, lists):
        if type(lists[0]) == np.ndarray:
            return np.concatenate(lists, 0)
        elif type(lists[0]) == dict:
            merged = {}
            for key in lists[0].keys():
                merged[key] = [i for subdict in lists for i in subdict[key]]
            return merged
        else:
            return [i for sublist in lists for i in sublist]

    ## read image as C x M x N array in range [0, 1]
    def __read_image(self, path):
        img = scipy.misc.imread(path)[:,:,:-1].transpose(2,0,1) / 255.
        return img

    def __read_data_files(self, data_files, selections, idx):
        outputs = []
        for ind, sel in enumerate(selections):
            
            ## lights : simply index into array
            if sel == 'lights':
                out = data_files[ind][idx,:]

            ## composite : pointwise multiplication between reflectance and shading
            elif sel == 'input':
                reflectance_path = data_files[ind]['reflectance'][idx]
                shading_path = data_files[ind]['shading'][idx]
                reflectance = self.__read_image(reflectance_path)
                shading = self.__read_image(shading_path)
                out = reflectance * shading
            
            ## other intrinsic images : read image from disk
            else:
                path = data_files[ind][idx]
                out = self.__read_image(path)
                if sel == 'shading':
                    out = out[0,:,:]
                if sel == 'depth':
                    out = out[0,:,:]
                    out = 1 - out
                if sel == 'normals':
                    out = utils.image_to_vector(out)
            outputs.append(out)
        return outputs

    def __selection_len(self, data_files):
        files = data_files[0]
        if type(files) == np.ndarray:
            return files.shape[0]
        elif type(files) == dict:
            return len(files['reflectance'])
        else:
            return len(files)

    def __getitem__(self, unlabeled_idx):
        # print idx
        unlabeled_outputs = self.__read_data_files(self.unlabeled_data_files, self.unlabeled_selections, unlabeled_idx)

        ## just grab a random labeled example
        labeled_idx = np.random.randint(self.labeled_size)
        labeled_outputs = self.__read_data_files(self.labeled_data_files, self.labeled_selections, labeled_idx)
        return unlabeled_outputs, labeled_outputs

    def __len__(self):
        return self.__selection_len(self.unlabeled_data_files)



if __name__ == '__main__':
    import time
    directory = '../../dataset/output/'
    unlabeled_datasets = 'car_normalized'
    labeled_datasets = 'airplane_normalized,boat_normalized'
    # selections = ['input', 'mask', 'albedo', 'normals', 'lights']
    # selection_fn = lambda fname: 'shading' in fname
    dset = ComposerDataset(directory, unlabeled_datasets, labeled_datasets)
    loader = torch.utils.data.DataLoader(dset, batch_size=16, num_workers=4)
    print 'done init'
    time_0 = time.time()
    for i, inp in enumerate(loader):
        # pdb.set_trace()
        print i, [[t.size() for t in sublist] for sublist in inp], len(inp), len(inp[0]), len(inp[1])
    print 'total time: ', time.time() - time_0

