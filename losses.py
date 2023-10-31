import torch 
import torch.nn as nn 
import torch.nn.functional as F
from nnAudio import features
from utils.utility import get_device
from utils.processing import normalize_energy_torch

class MSSpectralLoss(nn.Module):
    '''multi scale spectral loss'''
    def __init__(self, sr=48000, norm_peak=False):
        super().__init__()
        # fft sizes
        self.n_fft = [256, 512, 1024, 2048, 4096]
        self.hop_size = 0.25
        self.l1loss = nn.L1Loss()
        self.sr = sr
        self.norm_peak = norm_peak

    def forward(self, y_pred, y_true):
        loss_match = 0 # initialize match loss
        for i, n_fft in enumerate(self.n_fft):
            # initialize stft function with new nfft 
            stft = features.stft.STFT(
                n_fft = n_fft,
                hop_length = int(n_fft*self.hop_size),
                window = 'hann',
                freq_scale = 'log',
                sr = self.sr,
                output_format = 'Magnitude',
                verbose=False,
                fmin = 20,
                fmax = self.sr/2, 
            )
            stft = stft.to(get_device())
            if self.norm_peak:
                y_pred = y_pred/torch.max(torch.abs(y_pred))
                y_true = y_true/torch.max(torch.abs(y_true))
            Y_pred = stft(y_pred)
            Y_true = stft(y_true)
            # update match loss
            loss_match += self.l1loss(Y_pred, Y_true)
        return loss_match

if __name__ == "__main__":
    import soundfile as sf 
    ir_true, _ = sf.read("/Users/dalsag1/Dropbox (Aalto)/aalto/projects/diff-delay-net/shungo-dar/dar-main/as_2_rir.wav")
    ir_pred, _ = sf.read("/Users/dalsag1/Dropbox (Aalto)/aalto/projects/diff-delay-net/shungo-dar/dar-main/as_2_freq_sampled_estimation.wav")
    ir_true, ir_pred = map(lambda x: torch.tensor(x[:12000], dtype=torch.float).view(1, -1), (ir_true, ir_pred))
    ir_true = normalize_energy_torch(ir_true)
    ir_pred = normalize_energy_torch(ir_pred)
    mss_loss = MSSpectralLoss(norm_peak=True)
    loss = mss_loss(ir_pred, ir_true)
    print(loss)