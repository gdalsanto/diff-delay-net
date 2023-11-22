import torch 
import torch.nn as nn 
import torch.nn.functional as F
from nnAudio import features
from utils.utility import get_device

class MSSpectralLoss(nn.Module):
    '''multi scale spectral loss'''
    def __init__(self, sr = 48000):
        super().__init__()
        # fft sizes
        self.n_fft = [256, 512, 1024, 2048, 4096]
        self.overlap = 0.875
        self.sr = sr
        self.l1loss = nn.L1Loss()

    def forward(self, y_pred, y_true):
        loss_match = 0 # initialize match loss
        for i, n_fft in enumerate(self.n_fft):
            # initialize stft function with new nfft 
            hop_length = int(n_fft*(1-self.overlap))
            stft = features.stft.STFT(
                n_fft = n_fft,
                hop_length = hop_length,
                window = 'hann',
                freq_scale = 'log',
                sr = self.sr,
                fmin = 20,
                fmax = self.sr // 2,
                output_format = 'Magnitude',
                verbose=False
            )
            stft = stft.to(get_device())

            Y_pred = stft(y_pred)
            Y_true = stft(y_true)
            # update match loss
            loss_match += self.l1loss(Y_pred, Y_true)
        return loss_match
