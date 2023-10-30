# Differentiable Artificial Reverberation 
# Replica of Sungho Lee et al paper

import argparse
import os
import time
from tqdm import tqdm
from dataset import *
from utils.logging import *
from model import *
from diff_dsp import *

from losses import *

def load_dataset(args):
    # get training and valitation dataset
    dataset = rirDataset(args)
    # split data into training and validation set 
    train_set, valid_set = split_dataset(
        dataset, args.split)

    # dataloaders
    train_loader = get_dataloader(
        train_set,
        batch_size=args.batch_size,
        shuffle = args.shuffle,
    )
    
    valid_loader = get_dataloader(
        valid_set,
        batch_size=args.batch_size,
        shuffle = args.shuffle,
    )
    return train_loader, valid_loader 

def train(args, train_dataset, valid_dataset):

    if (get_device == 'cuda') & torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = 'cpu'
    # initialize network
    net = ASPestNet()
    args.device = get_device()
    net = net.to(args.device )
    
    # check if checkpoint is available
    # arg.out_path will be updated
    args, net, _ = restore_checkpoint(args, net)
    # save arguments 
    with open(os.path.join(args.out_path, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # ----------- TRAINING CONFIGURATIONS ----------- # 
    # optimizer 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # loss
    criterion = MSSpectralLoss()
    # learning rate scheduler 
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size = 50000,
        gamma = 10**(-0.2)
    ) 
    # early stopping 
    early_stop = EarlyStopper(
        patience=50000, 
        min_delta=1e-4)

    # frequency samples to oveluate the transfer funciton on 
    # args.num is the length of the impulse response. We compute the transfer 
    # funciton on [0, fs/2] 
    x = get_frequency_samples(args.num//2+1)     
    args.steps = 0
    train_loss, valid_loss = [], []

    # sample one test example from validation set
    test_batch = next(iter(valid_dataset))
    write_audio(
        test_batch[0,:], 
        os.path.join(args.out_path, 'audio_output'),
        'target_ir.wav')

    for epoch in range(args.max_epochs):
        epoch_loss = 0
        grad_norm = 0
        st = time.time()
        # -------- TRAINING
        for i, data in enumerate(tqdm(train_dataset)):
            input = data
            target = input.clone()
            optimizer.zero_grad()
            estimate, _, _ = net(input, x)
            # apply loss
            loss = criterion(estimate, target)
            epoch_loss += loss.item()
            loss.backward()
            # clip gridients
            grad_norm += nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)
            # update the wieghts
            optimizer.step()
            # update scheduler
            if args.steps >= args.scheduler_steps:
                scheduler.step()

            args.steps += 1
        
        train_loss.append(epoch_loss/len(train_dataset))

        # --------- VALIDATION
        epoch_loss = 0
        for i, data in enumerate(tqdm(valid_dataset)):
            input = data
            target = input.clone()
            optimizer.zero_grad()
            estimate, _, _ = net(input, x)
            # apply loss
            loss = criterion(estimate, target)
            epoch_loss += loss.item()  

        valid_loss.append(epoch_loss/len(valid_dataset))          
        
        et = time.time()
        to_print = get_str_results(
            epoch=epoch, 
            train_loss=train_loss, 
            valid_loss=valid_loss, 
            time=et-st,
            lr = scheduler.get_last_lr()[0])
        print(to_print)

        # LOGGING every 10 epochs 
        # if (epoch % 10) == 0:
        save_checkpoint(args, net, epoch)

        test_ir_out, _, _ = net(test_batch, x)
        write_audio(
            test_ir_out[0,:].detach(), 
            os.path.join(args.out_path, 'audio_output'),
            'e{:04d}-output-ir-loss{:.3f}.wav'.format(epoch, criterion(test_ir_out[0,:], test_batch[0,:])))
        
        with open(os.path.join(args.out_path, 'log.txt'), "a") as file:
            file.write("epoch: {:04d} train loss: {:6.4f} valid loss: {:6.4f}\n".format(
                epoch, train_loss[-1], valid_loss[-1]))
        
        if early_stop.early_stop(valid_loss[-1]):
            return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    dset_parser = parser.add_argument_group('dset', 'dataset sepcific args')

    dset_parser.add_argument('--sr', default=48000,
        help='sample rate')
    dset_parser.add_argument('--ds_path', '-p', 
        help='directly point to dataset path')
    dset_parser.add_argument('--rir_length', type=float, default=2.,
        help='rir length in seconds')
    dset_parser.add_argument('--split', type=float, default=0.8,
        help='training / validation split')
    dset_parser.add_argument('--shuffle', action='store_false',
        help='if true, shuffle the data in the dataset at every epoch')
    dset_parser.add_argument('--batch_size', type=int, default=4,
        help='batch size')
    
    train_parser = parser.add_argument_group('train', 'training sepcific args')

    train_parser.add_argument('--num', default=120000, 
        help='frequency-sampling points') 
    train_parser.add_argument('--lr', default=10e-5,
        help='learning rate')
    train_parser.add_argument('--clip_max_norm', default=10, 
        help='gradient clipping maximum gradient norm')
    train_parser.add_argument('--max_epochs', default=10000, 
        help='max number of epochs')
    train_parser.add_argument('--scheduler_steps', default=250000,
        help='number of training steps needed before activating the lr scheduler')
    train_parser.add_argument('--log', action='store_false', 
        help='turn off logging')
    train_parser.add_argument('--out_path', 
        help='path to output directory')
    train_parser.add_argument('--restore_checkpoint', action='store_true',
        help='if true restore checkpoint')
    train_parser.add_argument('--checkpoint_path',
        help='path to checkpoints directory')
    train_parser.add_argument('--norm_h0', action='store_true',
        help='If true, match the energy of h0 to that of the remaining ir')
    args = parser.parse_args()


    train_dataset, valid_dataset = load_dataset(args)
    
    train(args, train_dataset, valid_dataset)
