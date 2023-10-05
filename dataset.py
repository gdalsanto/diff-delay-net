import random
import torch
import os
import soundfile as sf
from time import time
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from utils.processing import *
from utils.utility import *

class rirDataset(Dataset):

    def __init__(self, args):
        # make list of all filenames enclosed in args.path
        pathlist = [y for x in os.walk(args.ds_path) for y in glob(os.path.join(x[0], '*.wav'))]
        self.pathlist = pathlist 
        print("Loading RIRs to {}".format(get_device()))
        self.data_loaded = []
        st = time()
        for i in tqdm(range(0, 100)): #len(pathlist))):

            rir, samplerate = sf.read(pathlist[i], dtype='float32')
            if samplerate!=args.sr:
                raise ValueError('Wrong samplerate: detected {} - required {}'.format(samplerate, args.sr))
            # if multichannel, take only the first channel
            if len(rir.shape)>1:
                print('Converting to mono by taking only the first channel')
                rir = rir[0, :]
            # adjust length 
            rir_len_samples = int(args.rir_length*args.sr)
            if rir.shape[0] > rir_len_samples:
                rir = rir[:rir_len_samples]
            elif rir.shape[0] < rir_len_samples:
                rir = np.pad(rir, 
                ((0, rir_len_samples - rir.shape[0])),
                mode = 'constant')

            # --------------- PREPROCESSING --------------- #
            # remove onset 
            onset = find_onset(rir)
            rir = np.pad(rir[onset:],(0, onset))
            # multply random gain to direct sound 
            rir = augment_direct_gain(rir, sr=args.sr)
            # nornalize 
            rir = normalize_energy(rir)
            # --------------------------------------------- #
            
            # convert to tensor and move to device 
            self.data_loaded.append(torch.tensor(rir).to(get_device()))
            del rir
        et = time()
        print('Finished loading RIRs in {:.3f} seconds'.format(et-st))
        
    def __len__(self):
        return len(self.data_loaded)

    def __getitem__(self, index):
        return self.data_loaded[index]
    
    def get_pathlist(self):
        return self.pathlist


def split_dataset(dataset, split):
    ''' randomly split a dataset into non-overlapping new datasets of 
    sizes given in 'split' argument'''
    # use split % of dataset for validation 
    train_set_size = int(len(dataset) * split)
    valid_set_size = int(len(dataset) - train_set_size)
    print('Device {}'.format((next(iter(dataset))).get_device()))
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(
        dataset, 
        [train_set_size, valid_set_size], 
        generator=seed)

    return train_set, valid_set

def get_dataloader(dataset, batch_size, shuffle=True):
    ''' create torch dataloader form given dataset '''
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator(),
        drop_last = True
    )
    return dataloader

