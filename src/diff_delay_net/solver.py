# Differentiable Artificial Reverberation 
# Replica of Sungho Lee et al paper

import argparse
import os
import pickle
import time
from torch import Tensor
from tqdm import tqdm
from dataset import *
from utils.logging import *
from model import *
from diff_dsp import *
from torch.utils.tensorboard.writer import SummaryWriter
from utils.metrics import compute_speech2fdn_metrics
from losses import *
import warnings
from scipy.signal import oaconvolve

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

def save_batch(step_dict, out_dir, batch_idx):
    dry = step_dict["dry"].squeeze()
    wet = step_dict["wet"].squeeze()
    wet_fdn = step_dict["wet_fdn"].squeeze()
    rir = step_dict["rir"].squeeze()
    rir_fdn = step_dict["rir_fdn"].squeeze()

    batch_size = dry.shape[0]
    for i in range(batch_size):
        save_audio(
            os.path.join(out_dir, f"{(batch_idx):03d}_{i:03d}_dry.wav"),
            dry[i],
            48000,
        )
        save_audio(
            os.path.join(out_dir, f"{(batch_idx):03d}_{i:03d}_wet.wav"),
            wet[i],
            48000,          
        )
        save_audio(
            os.path.join(out_dir, f"{(batch_idx):03d}_{i:03d}_wet_fdn.wav"),
            wet_fdn[i],
            48000,
        )
        save_audio(
            os.path.join(out_dir, f"{(batch_idx):03d}_{i:03d}_rir.wav"),
            rir[i],
            48000,          
        )
        save_audio(
            os.path.join(out_dir, f"{(batch_idx):03d}_{i:03d}_rir_fdn.wav"),
            rir_fdn[i],
            48000,
        )        

def step(model, batch, freqs, device):
    dry, wet, rir, wetspec = batch
    rir = rir.to(device)
    dry = dry.to(device)[:, :, :int(2.5*48000)]  # use only first 2.5 seconds
    # wet = wet.to(device)[:, :, :int(2.5*48000)] 
    rir_norm = rir / torch.sqrt(torch.sum(rir**2, dim=-1, keepdim=True))
    wet = torch.tensor(oaconvolve(dry.cpu(), rir_norm.cpu(), mode="full", axes=-1)).to(device)[:, :, :int(2.5*48000)] 
    wetspec = wetspec.to(device)
    
    wet_fdn, rir_fdn, ext_params, z = model(wet, dry, freqs)

    return {
        "dry": dry,
        "wet": wet,
        "z": z,
        "wet_fdn": wet_fdn,
        "rir_fdn": rir_fdn,
        "ext_params": ext_params,
        "rir": rir,
    }

def load_dataset(args):
    # get training and valitation dataset
    if args.mode == 'speech':
        dataset = speechDataset(args)
        return dataset
    else:
        dataset = rirDataset(args)
        # split data into training and validation set
        train_set, valid_set = split_dataset(dataset, args.split)
        # dataloaders
        train_loader = get_dataloader(  train_set,
                                        batch_size=args.batch_size,
                                        shuffle = args.shuffle,) 
        valid_loader = get_dataloader(  valid_set,
                                        batch_size=args.batch_size,
                                        shuffle = args.shuffle,)
        return train_loader, valid_loader 

def train(args, dataset):

    # set device and tensor tyoe
    args.device = get_device()
    # make log directory inside the output path 
    log_dir = os.path.join(args.out_path, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if (args.device == 'cuda') & torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    print("Device: "+str(args.device))

    # initialize network
    model = ASPestNet()
    model = model.to(args.device)

    # ----------- TRAINING CONFIGURATIONS ----------- # 
    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # loss
    signal_losses = [MSSpectralLoss()]
    weights = [1.0]
    # learning rate scheduler 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size = 50000,
                                                gamma = 10**(-0.2)) 

    # save arguments 
    with open(os.path.join(args.out_path, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))

    # frequency samples to evaluate the transfer function on
    # args.num is the length of the impulse response. We compute the transfer
    # function on [0, fs/2]
    freqs = get_frequency_samples(args.num//2+1)

    # logging and early stopping stuff
    logger = SummaryWriter(log_dir)
    step_idx, stag_ct, best_val_loss = 0, 0, 1e10
    best_state = model.state_dict()

    for epoch in range(args.max_epochs):

        logger.add_scalar("epoch", epoch, step_idx)

        # training loop
        batch_idx, train_loss = 0, 0
        model.train()
        with tqdm(
            dataset.train_loader,
            desc=f"Epoch {epoch + 1}/{args.max_epochs}",
            leave=False,
        ) as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                step_dict = step(model, batch, freqs, args.device)
                
                target = step_dict["rir"][:, 0, :]
                pred = step_dict["rir_fdn"][:, :target.shape[-1]]
                # compute losses
                signal_loss = 0
                for loss, weight in zip(signal_losses, weights):
                    name = loss.__class__.__name__
                    tmp = loss(pred, target)
                    signal_loss += weight * tmp
                    logger.add_scalar(f"loss/train_{name}", tmp.item(), step_idx)
                    if tmp.isnan():
                        raise ValueError(f"Loss {name} is NaN.")

                loss = signal_loss

                loss.backward()
                optimizer.step()

                logger.add_scalar("loss/train", loss.item(), step_idx)

                train_loss += loss.item()
                step_idx += 1
                batch_idx += 1

            train_loss /= batch_idx
            logger.add_scalar("loss/train_epoch", train_loss, step_idx)

            # validation loop
            model.eval()
            batch_idx, val_loss = 0, 0
            pbar.set_description("Validating...")
            pbar.refresh()
            for batch in dataset.valid_loader:
                with torch.no_grad():
                    step_dict = step(model, batch, freqs, args.device)
                    target = step_dict["rir"][:, 0, :]
                    pred = step_dict["rir_fdn"][:, :target.shape[-1]]

                    # compute losses
                    signal_loss = 0
                    for loss, weight in zip(signal_losses, weights):
                        name = loss.__class__.__name__
                        tmp = loss(pred, target)
                        signal_loss += weight * tmp
                        logger.add_scalar(
                            f"loss/valid_{name}", tmp.item(), step_idx
                        )
                        if tmp.isnan():
                            raise ValueError(f"Loss {name} is NaN.")
                        
                    loss = signal_loss

                    val_loss += loss.item()
                    batch_idx += 1

            val_loss /= batch_idx
            logger.add_scalar("loss/valid_epoch", val_loss, step_idx)

        # update scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

        torch.cuda.empty_cache()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            stag_ct = 0
        else:
            stag_ct += 1

        if (
            stag_ct > args.patience or epoch == args.max_epochs - 1
        ) and epoch >= 1:
            print(f"Stopping after epoch {epoch}.")
            break

    print("Start evaluating results")
    try:
        epoch_loss, batch_idx = 0, 0
        outputs = []
        for batch in tqdm(dataset.test_loader):
            with torch.no_grad():
                step_dict = step(model, batch, freqs, args.device)
                # save_batch(step_dict, log_dir, batch_idx)
                out_dict = {
                    "dry": step_dict["dry"][:, 0, :].cpu().numpy().squeeze(),
                    "wet": step_dict["wet"][:, 0, :].cpu().numpy().squeeze(),
                    "wet_fdn": step_dict["wet_fdn"].cpu().numpy().squeeze(),
                    "rir": step_dict["rir"][:, 0, :].cpu().numpy().squeeze(),
                    "rir_fdn": step_dict["rir_fdn"].cpu().numpy().squeeze(),
                    "ext_params": step_dict["ext_params"],
                }
                target = step_dict["rir"][:, 0, :]
                pred = step_dict["rir_fdn"][:, :target.shape[-1]]

                # compute losses
                signal_loss = 0
                for loss, weight in zip(signal_losses, weights):
                    name = loss.__class__.__name__
                    tmp = loss(pred, target)
                    signal_loss += weight * tmp
                    logger.add_scalar(f"loss/test_{name}", tmp.item(), step_idx)
                    if tmp.isnan():
                        raise ValueError(f"Loss {name} is NaN.")

                loss = signal_loss 
                outputs.append(out_dict)

                epoch_loss += loss.item()
                batch_idx += 1

    except KeyboardInterrupt:
        batch_idx += 1
        print("\nTesting aborted by user, attempting to compute metrics.")

    compute_speech2fdn_metrics(outputs, log_dir)

    epoch_loss /= max(1, batch_idx)
    logger.add_scalar("loss/test", epoch_loss)
    print(f"Test loss: {epoch_loss}")

    # save the model state
    model_path = os.path.join(log_dir, "model.pth")
    torch.save(model.cpu().state_dict(), model_path)
    print(f"Model saved to {model_path}")
    with open(log_dir + "/outputs.pkl", "wb") as f:
        pickle.dump(outputs, f)

    logger.close()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    dset_parser = parser.add_argument_group('dset', 'dataset sepcific args')

    dset_parser.add_argument('--sr', default=48000,
        help='sample rate')
    dset_parser.add_argument('--ds_path', '-p', 
        help='directly point to dataset path')
    dset_parser.add_argument('--rir_length', type=float, default=1.8,
        help='rir length in seconds')
    dset_parser.add_argument('--split', type=float, default=0.8,
        help='training / validation split')
    dset_parser.add_argument('--shuffle', action='store_false',
        help='if true, shuffle the data in the dataset at every epoch')
    dset_parser.add_argument('--batch_size', type=int, default=4,
        help='batch size')
    dset_parser.add_argument('--len_dataset', type=int,
        help='number of elements to select from dataset')       
    dset_parser.add_argument('--num_workers', type=int, default=4,
        help='number of workers for data loading')
    dset_parser.add_argument('--pin_memory', action='store_true',
        help='if true, the data loader will copy Tensors into CUDA pinned memory before returning them')
    train_parser = parser.add_argument_group('train', 'training sepcific args')

    train_parser.add_argument('--mode', default="fdn", choices=["fdn", "speech"],
        help='mode of training') 
    train_parser.add_argument('--num', default=120000, 
        help='frequency-sampling points') 
    train_parser.add_argument('--lr', type = float, default=10e-5,
        help='learning rate')
    train_parser.add_argument('--clip_max_norm', default=10, 
        help='gradient clipping maximum gradient norm')
    train_parser.add_argument('--max_epochs', type=int, default=512, 
        help='max number of epochs')
    train_parser.add_argument('--patience', default=12,
        help='number of training epochs needed before activating the lr scheduler')
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

    if args.out_path is None:
        base_dir = 'output'
    else: 
        base_dir = args.out_path
    # make directory where to store checkpoints, outputs, and log files
    args.out_path = os.path.join(base_dir, time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(args.out_path)
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(args.out_path, "checkpoint")
        os.makedirs(args.checkpoint_path)

    dataset = load_dataset(args)
    
    train(args, dataset)
