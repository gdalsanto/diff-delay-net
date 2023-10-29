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
    args, net, epoch = restore_checkpoint(args, net)

    x = get_frequency_samples(int(np.floor(args.num/2+1)))

    # load input rir
    rir, samplerate = sf.read(args.filepath, dtype='float32')
    if samplerate!=args.sr:
        raise ValueError('Wrong samplerate: detected {} - required {}'.format(samplerate, args.sr))
    
    rir_len_samples = int(args.rir_length*args.sr)
    if rir.shape[0] > rir_len_samples:
        rir = rir[:rir_len_samples]
    elif rir.shape[0] < rir_len_samples:
        rir = np.pad(rir, 
        ((0, rir_len_samples - rir.shape[0])),
        mode = 'constant')
    rir = normalize_energy(rir)
    input = torch.unsqueeze(torch.tensor(rir).to(get_device()), 0)

    rir_estimated, _, _ = net(input, x)
    # compute loss
    mss_loss = MSSpectralLoss()
    loss = mss_loss(rir_estimated, input)
    print(loss)
    # save results 
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    filepath = os.path.splitext(os.path.basename(args.filepath))[0] + '_estimated.wav'
    
    rir_estimated = rir_estimated.detach().numpy()
    rir_estimated = rir_estimated / np.max(np.abs(rir_estimated))
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
    parser.add_argument('--checkpoint_path',
        help = 'Path to checkpoint to use')
    parser.add_argument('--filepath', 
        help = 'Path to the input file')
    parser.add_argument('--out_dir', default = '/Users/dalsag1/Dropbox (Aalto)/aalto/projects/diff-delay-net/inference', 
        help = 'Path to the output folder')
    parser.add_argument('--num', default=120000, 
        help='frequency-sampling points') 
    parser.add_argument('--rir_length', type=float, default=1.8,
        help='rir length in seconds')
    parser.add_argument('--save_filters', action='store_true',
        help='If true, save the coefficents of the filters')

    args = parser.parse_args()
    
    inference(args)