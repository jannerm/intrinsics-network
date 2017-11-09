class ParamUpdater:

    ## 10-normals,shader_10-shader
    def __init__(self, transfer):
        epoch_sep = transfer.split('_')
        self.num_sep = [i.split('-') for i in epoch_sep]
        self.num_sep = [[int(i[0]), i[1]] for i in self.num_sep]
        
        current = 0
        self.time_steps = [current]
        for i in self.num_sep:
            current += i[0]
            self.time_steps.append(current)

        # print self.time_steps
        # param_sep = [[i[0], i[1].split('_')] for i in num_sep]
        # print epoch_sep
        # print self.num_sep
        # print num_sep2
        # print param_sep

    def check(self, epoch):
        if epoch in self.time_steps:
            return True
        else:
            return False
        # pass

    def refresh(self, epoch):
        ind = self.time_steps.index(epoch)
        ind = min(ind, len(self.time_steps)-2)
        # print ind
        transfer = self.num_sep[ind][1]
        return transfer
        # pass


if __name__ == '__main__':
    transfer = '10-normals,shader_10-shader'
    p = ParamUpdater(transfer)

    for epoch in range(400):
        if p.check(epoch):
            print epoch, p.refresh(epoch)
    # print p.refresh(0)
    # print p.refresh(10)
    # print p.refresh(20)