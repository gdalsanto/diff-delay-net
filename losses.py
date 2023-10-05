import torch 
import torch.nn as nn 
import torch.nn.functional as F
from nnAudio import features
from utils.utility import get_device

class MSSpectralLoss(nn.Module):
    '''multi scale spectral loss'''
    def __init__(self):
        super().__init__()
        # fft sizes
        self.n_fft = [256, 512, 1024, 2048, 4096]
        self.hop_size = 0.25
        self.l1loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        loss_match = 0 # initialize match loss
        for i, n_fft in enumerate(self.n_fft):
            # initialize stft function with new nfft 
            stft = features.stft.STFT(
                n_fft = n_fft,
                hop_length = int(n_fft*self.hop_size),
                window = 'hann',
                freq_scale = 'log',
                sr = 44100,
                output_format = 'Magnitude',
                verbose=False,
                fmin = 20,
                fmax = 12000,   # TODO ask or check on this 
            )
            stft = stft.to(get_device())

            Y_pred = stft(y_pred)
            Y_true = stft(y_true)
            # update match loss
            loss_match += self.l1loss(Y_pred, Y_true)
        return loss_match
