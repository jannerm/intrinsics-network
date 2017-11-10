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


    def check(self, epoch):
        if epoch in self.time_steps:
            return True
        else:
            return False

    def refresh(self, epoch):
        ind = self.time_steps.index(epoch)
        ind = min(ind, len(self.time_steps)-2)
        transfer = self.num_sep[ind][1]
        return transfer


if __name__ == '__main__':
    transfer = '10-normals,shader_10-shader'
    p = ParamUpdater(transfer)

    for epoch in range(400):
        if p.check(epoch):
            print epoch, p.refresh(epoch)