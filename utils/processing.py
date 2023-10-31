import torch 
import numpy as np 
from utils.utility import get_device


def normalize_energy(x):
    ''' normalize energy of x to 1 '''
    energy = np.sum(np.power(np.abs(x),2))
    return np.divide(x , np.power(energy, 1/2))
    
def normalize_energy_torch(x):
    ''' normalize energy of x to 1 '''
    energy = torch.sum(torch.pow(torch.abs(x),2))
    return torch.divide(x , torch.pow(energy, 1/2))

def find_onset(rir):
    ''' find onset in input RIR by extracting a local energy envelope of the 
    RIR then finding its maximum point'''
    # extract local energy envelope 
    win_len = 64
    overlap = 0.75
    win = np.hanning(win_len)
    
    # pad rir 
    rir = np.pad(rir, (int(win_len * overlap), int(win_len * overlap)))
    hop = (1 - overlap)
    n_wins = np.floor(rir.shape[0] / (win_len * hop )- 1/2/hop ) 
    

    local_energy = []
    for i in range(1,int(n_wins - 1)):
        local_energy.append(
                np.sum( 
                    (rir[(i-1)*int(win_len*hop):(i-1)*int(win_len*hop) + win_len] ** 2) * win)
            )
    # discard trailing points 
    # remove (1/2/hop) to avoid map to negative time (center of window) 
    n_win_discard = (overlap/hop) - (1/2/hop) 

    local_energy = local_energy[int(n_win_discard):]
    return int(win_len * hop * ( np.argmax(local_energy) - \
        1 ))    # one hopsize as safety margin 

def augment_direct_gain(rir, low = 10**(-12/20), high=10**(3/20), sr=48000):
    ''' multiply first 5ms of rir with random gain sampled from uniform 
    distribution in [low, high] '''
    direct =int(0.005*sr) # take first 5 ms 
    gain = (high-low)*np.random.rand(1) + low
    rir[:direct] = rir[:direct]*gain
    return rir



def get_frequency_samples(num):
    '''
    get frequency samples (in radians) sampled at linearly spaced points along the unit circle
    Args    num (int): number of frequency samples
    Output  frequency samples in radians between [0, pi]
    '''
    # angle = torch.arange(0, 1+(1/num)/2, 1/num)
    # abs = torch.ones(num+1)
    angle = torch.linspace(0, 1-1/num, num, device=get_device())
    abs = torch.ones(num, device=get_device())
    return torch.polar(abs, angle * np.pi)    

def householder_matrix(N):
    ''' 
    get NxN householder matrix from random vector of length N 
    '''
    u = np.random.rand(N,1)
    u = u / np.sqrt(np.sum(np.power(u, 2))) 
    return (np.identity(N) - 2*u*np.transpose(u)).astype('float32')