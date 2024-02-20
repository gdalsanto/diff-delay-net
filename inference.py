import os
import torch 
import argparse
import soundfile as sf
from utils.utility import *
from utils.logging import *
from utils.processing import *
from model import ASPestNet
from losses import MSSpectralLoss
import scipy
import pandas as pd
import torchaudio.transforms as T

def inference(args): 

    if (get_device == 'cuda') & torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = 'cpu'
    
    net = ASPestNet()
    args.device = get_device()
    net = net.to(args.device)

    # load checkpoint
    args.restore_checkpoint = True
    net.eval()
    args, net, epoch = restore_checkpoint(args, net, epoch=6)
    net.train() # if in eval mode the decay is 0 (inf rt) TODO check why
    x = get_frequency_samples(int(np.floor(args.num/2+1)))

    # load input rir
    rir_len_samples = int(args.rir_length*args.sr)

    if args.rand_input: 
        input = torch.rand(4, rir_len_samples).to(get_device())
    else:
        # loop over the RIRs in the ds_inf_path
        if os.path.splitext(os.path.basename(args.ds_inf_path))[-1] == '.csv':
            df = pd.read_csv(args.ds_inf_path)
            pathlist = [filename for filename in df['filename']]
        else: 
            pathlist = [y for x in os.walk(args.ds_inf_path) for y in glob(os.path.join(x[0], '*.wav'))]
        with torch.no_grad():
            for i, filepath in enumerate(pathlist):
                rir, samplerate = sf.read(filepath, dtype='float32')
                if len(rir.shape) > 1:
                    rir = rir[:,-1]
                if samplerate!=args.sr:
                    rir = torch.tensor(rir)
                    resampler = T.Resample(samplerate, args.sr, dtype=rir.dtype)
                    rir = resampler(rir)
                    rir = rir.numpy()
                    # raise ValueError('Wrong samplerate: detected {} - required {}'.format(samplerate, args.sr))
                    Warning('Wrong samplerate: detected {} - required {}'.format(samplerate, args.sr))

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
                # nornalize 
                rir = normalize_energy(rir)        
                input = torch.unsqueeze(torch.tensor(rir).to(get_device()), 0)

                rir_estimated = net(input, x)[0]
                # compute loss
                mss_loss = MSSpectralLoss()
                loss = mss_loss(rir_estimated, torch.tensor(rir).unsqueeze(0))
                print(loss)
                # save results 
                if not os.path.exists(args.out_dir):
                    os.makedirs(args.out_dir)
                filepath = os.path.splitext(os.path.basename(filepath))[0] + '_estimated.wav'
                
                rir_estimated = rir_estimated.detach().numpy()
                # rir_estimated = rir_estimated / np.max(np.abs(rir_estimated))
                sf.write(os.path.join(args.out_dir, filepath), rir_estimated[0], args.sr)
                # save filters
                if args.save_filters:
                    # get filter's parameters and frequency response 
                    parameters, filters_tf = net.get_filters(input, x)
                    # make dir 
                    os.makedirs(os.path.join(args.out_dir, 'filters'),  exist_ok=True)
                    # save coefficents in .mat format 
                    scipy.io.savemat(os.path.join(args.out_dir, 'filters','parameters_'+'epoch{:04d}'.format(epoch)+'.mat'), parameters)
                    scipy.io.savemat(os.path.join(args.out_dir, 'filters','filters_tf_'+'epoch{:04d}'.format(epoch)+'.mat'), filters_tf)
   

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()

    parser.add_argument('--sr', default=48000,
        help='sample rate')
    parser.add_argument('--results_dir',
        help = 'Path to results folder. NOTE: checkpoint folder is expected to be contained by results_dir')
    parser.add_argument('--ds_inf_path', 
        help = 'Path to the inference dataset folder')
    parser.add_argument('--num', default=120000, 
        help='frequency-sampling points') 
    parser.add_argument('--rir_length', type=float, default=1.8,
        help='rir length in seconds')
    parser.add_argument('--save_filters', action='store_true',
        help='If true, save the coefficents of the filters')
    parser.add_argument('--rand_input', action='store_true', 
        help='If true, use tensor of random values as input')
    
    args = parser.parse_args()
    args.checkpoint_path = os.path.join(args.results_dir, 'checkpoint')
    args.out_dir = os.path.join(args.results_dir, 'inference')
    inference(args)