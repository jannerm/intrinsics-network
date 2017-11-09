import os, numpy as np, matplotlib, pdb
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Logger:
    def __init__(self, labels, savepath):
        self.labels = labels
        self.savepath = savepath
        self.data = [ [[],[]] for i in range(len(labels)) ]

    def update(self, train, val):
        # pdb.set_trace()
        assert len(train) == len(self.labels)
        # assert len(val) + 1 >= len(train)
        for ind in range(len(train)):
            self.data[ind][0].append(train[ind])
            if len(val) > ind:
                self.data[ind][1].append(val[ind])
        self.__plot()

    def __plot(self):
        for ind, lab in enumerate(self.labels):
            fullpath = os.path.join(self.savepath, '_log_' + lab + '.png')
            plt.clf()
            plt.plot(self.data[ind][0], label='train', color='g')
            plt.plot(self.data[ind][1], label='val', color='b')
            plt.legend()
            plt.title(lab)
            plt.ylabel('Error')
            plt.xlabel('Epoch')
            plt.savefig(fullpath)

            trainpath = os.path.join(self.savepath, '_log_' + lab + '_train.txt')
            valpath = os.path.join(self.savepath, '_log_' + lab + '_val.txt')
            trainfile = open(trainpath, 'w')
            valfile = open(valpath, 'w')
            
            trainfile.write( '\n'.join([str(i) for i in self.data[ind][0]]) )
            valfile.write( '\n'.join([str(i) for i in self.data[ind][1]]) )
            trainfile.close()
            valfile.close()


