import torch 
import os
import numpy as np
import soundfile as sf
from utils.utility import *
import time 
import re
from glob import glob


def save_checkpoint(args, net, epoch):
    ''' save checkpoint of the network 
    Args:    
        args    parsed arguments
        net     model whose parameters have to che saved in a checkpoint
        epoch   number of epochs 
    '''
    if args.out_path is None:
    # make directory where to store checkpoints, outputs, and log files
        args.out_path = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.out_path)
        if args.checkpoint_path is None:
            args.checkpoint_path = os.path.join(args.out_path, "checkpoint")
            os.makedirs(args.checkpoint_path)

    filename = "ASPestNet-weights-{:04d}.pt".format(epoch)
    torch.save(net.state_dict(), os.path.join(args.checkpoint_path, filename))
    return args
    
def restore_checkpoint(args, net, epoch = None, pattern = r'\d{4}'):
    ''' restore model from checkpoint of the network
    Args:    
        net     model to load the checkpoint onto
        epoch   epoch to be restored. If None, the latest checkpoint will be used
    '''    
    if not args.restore_checkpoint or args.checkpoint_path is None:
        print('Training a newly initialized network')
        args = save_checkpoint(args, net, 0)
        return args, net, epoch
    list_paths = glob(args.checkpoint_path)
    list_ids = [int(re.findall(pattern, weight_path)[-1])
                for weight_path in list_paths]
    if epoch is None:
        # find latest epoch
        filename = list_paths[max(list_ids)]
        epoch = max(list_ids)
    else:
        filename = list_paths[list_ids.index(epoch)]

    checkpoint = torch.load(
        filename, map_location=args.device)
    net.load_state_dict(checkpoint)
    return args, net, epoch

def write_audio(x, dir_path, filename='ir.wav', sr=44100):
    ''' save tensor as wave file
    Args:
        x           tensor to be saved 
        dir_path    path to directory where to save the wave file
        sr          sample rate  
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    x = x.cpu().numpy()
    # normalize to avoid clipping
    x = x / np.abs(np.max(x))
    sf.write(os.path.join(dir_path, filename), x, sr)