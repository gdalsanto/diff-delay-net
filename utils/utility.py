import torch

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.tensor(float('inf'), device=get_device())

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_device():
    '''set device according to cuda availablilty'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_steps(dataloader):
    '''get number of training steps required to process one epoch
    dataloader: instance of torch.utils.data.DataLoader '''
    return len(dataloader)/dataloader.batch_size()

def get_str_results(epoch=None, train_loss=None, valid_loss=None, time=None, lossF = None, lossT = None, lr=None):
    '''construct the string that has to be print at the end of the epoch'''
    to_print=''

    if epoch is not None:
        to_print += 'epoch: {:3d} '.format(epoch)
    
    if train_loss is not None:
        to_print += '- train_loss: {:6.4f} '.format(train_loss[-1])
                        
    if valid_loss is not None:
        to_print += '- test_loss: {:6.4f} '.format(valid_loss[-1])

    if time is not None:
        to_print += '- time: {:6.4f} s'.format(time)

    if lossF is not None:
        to_print += '- lossF: {:6.4f}'.format(lossF) 

    if lossT is not None:
        to_print += '- lossT: {:6.4f}'.format(lossT) 

    if lr is not None:
        to_print += '- lr: {:6.4f}'.format(lr) 

    return to_print