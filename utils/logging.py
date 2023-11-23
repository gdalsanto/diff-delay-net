import torch 
import os
import numpy as np
import soundfile as sf
from utils.utility import *
import time 
import re
from glob import glob


def save_checkpoint(args, net, optimizer, scheduler, epoch):
    ''' save checkpoint of the network 
    Args:    
        args    parsed arguments
        net     model whose parameters have to che saved in a checkpoint
        optimizer 
        scheduler
        epoch   number of epochs 

    '''
    if args.out_path is None:
    # make directory where to store checkpoints, outputs, and log files
        args.out_path = os.path.join('output', time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(args.out_path)
        if args.checkpoint_path is None:
            args.checkpoint_path = os.path.join(args.out_path, "checkpoint")
            os.makedirs(args.checkpoint_path)

    # create a dictionary to store all the state dictionaries
    state_dicts = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    # generate the filename for the combined state file
    filename = "ASPestNet-states-{:04d}.pt".format(epoch)

    # save the combined state dictionary to a single file
    torch.save(state_dicts, os.path.join(args.checkpoint_path, filename))
    return args
    
def restore_checkpoint(args, net, optimizer = None, scheduler = None, epoch = None, pattern = r'\d{4}'):
    ''' restore model from checkpoint of the network
    Args:    
        net     model to load the checkpoint onto
        epoch   epoch to be restored. If None, the latest checkpoint will be used
    '''    
    if not args.restore_checkpoint or args.checkpoint_path is None:
        print('Training a newly initialized network')
        args = save_checkpoint(args, net, optimizer, scheduler, 0)
        epoch = 0
        return args, net, optimizer, scheduler, epoch

    list_paths = glob(os.path.join(args.checkpoint_path, '*'))
    list_ids = [int(re.findall(pattern, weight_path)[-1])
                for weight_path in list_paths]
    if epoch is None:
        # find latest epoch
        filename = list_paths[list_ids.index(max(list_ids))]
        epoch = max(list_ids)
    else:
        filename = list_paths[list_ids.index(epoch)]

    checkpoint = torch.load(
        filename, map_location=args.device)

    # access individual state dictionaries
    net_state_dict = checkpoint['net']
    optimizer_state_dict = checkpoint['optimizer']
    scheduler_state_dict = checkpoint['scheduler']

    # load the state dictionaries into your PyTorch model, optimizer, and scheduler
    if net.training == False:
        net.load_state_dict(net_state_dict)
        return args, net.eval(), epoch

    net.load_state_dict(net_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)
    args.out_path = os.path.dirname(args.checkpoint_path).replace('/checkpoint', '')
    return args, net, optimizer, scheduler, epoch+1

def write_audio(x, dir_path, filename='ir.wav', sr=48000):
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