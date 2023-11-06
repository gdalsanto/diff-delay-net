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

def load_rir(sr, rir_length, rir_path):
    rir, samplerate = sf.read(rir_path, dtype='float32')
    if samplerate!=sr:
        raise ValueError('Wrong samplerate: detected {} - required {}'.format(samplerate, sr))
    # if multichannel, take only the first channel
    if len(rir.shape)>1:
        print('Converting to mono by taking only the first channel')
        rir = rir[0, :]
    # adjust length 
    rir_len_samples = int(rir_length*sr)
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
    rir = augment_direct_gain(rir, sr=sr)
    # nornalize 
    rir = normalize_energy(rir)
    return rir
    
def inference(args): 

    if (get_device == 'cuda') & torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = 'cpu'
    
    net = ASPestNet()
    net.eval()
    args.device = get_device()
    net = net.to(args.device)

    # load checkpoint
    args.restore_checkpoint = True
    args, net, epoch = restore_checkpoint(args, net)

    x = get_frequency_samples(int(np.floor(args.num/2+1)))

    # load input rir
    rir_len_samples = int(args.rir_length*args.sr)

    if args.rand_input: 
        input = torch.rand(4, rir_len_samples).to(get_device())
    else:
        input = torch.tensor(load_rir(args.sr, args.rir_length, args.filepath))

    '''
    with torch.no_grad():
        _,ir_late, h0 = net(input.unsqueeze(0), x) 
        # compute the energy of early ir and late ir
        energy_h0 = torch.mean(torch.pow(torch.abs(h0),2), dim=1)
        energy_late = torch.mean(torch.pow(torch.abs(ir_late),2), dim=1)
        
        # match energy of early part to that of late part 
        # TODO This should be changes with a more meaningful scaling
        net.h0_norm.data.copy_(torch.div(
            net.h0_norm, torch.pow( torch.min(
                energy_h0/energy_late), 1/2)))

        # normalize energy of ir to equal 1 
        ir ,_, _ = net(input.unsqueeze(0), x)   
        energy = torch.mean(torch.pow(torch.abs(ir),2), dim=1)
        net.ir_norm.data.copy_(torch.div(net.ir_norm, torch.pow( torch.max(energy), 1/2)))
    '''
    target = input.clone()
    rir_estimated, _, _ = net(input.unsqueeze(0), x)
    # compute loss
    mss_loss = MSSpectralLoss()
    loss = mss_loss(rir_estimated, target)
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
    parser.add_argument('--rand_input', action='store_true', 
        help='If true, use tensor of random values as input')
    args = parser.parse_args()
    
    inference(args)