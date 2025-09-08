from typing import List, Tuple, Union
import torch
import os
import numpy as np
from scipy.signal import butter, sosfilt, zpk2sos
from numpy import ndarray
import soundfile as sf
from scipy.stats import linregress

def save_audio(filepath, x, fs=48000):
    # check if the folder exists
    folder = os.path.dirname(filepath)
    if not os.path.exists(folder):
        os.makedirs(folder)
    sf.write(filepath, x.detach().cpu().numpy(), fs)

def get_onset(rir: ndarray, thresh: float = 0.1) -> int:
    assert thresh >= 0.0 and thresh <= 1.0, "Threshold must be in [0, 1]"
    # get first index of exceeding a fraction of the peak
    return np.where(np.abs(rir) >= np.max(np.abs(rir)) * thresh)[0][0]


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.tensor(float('inf'), device=get_device())

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def get_device():
    '''set device according to cuda availablilty'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_steps(dataloader):
    '''get number of training steps required to process one epoch
    dataloader: instance of torch.utils.data.DataLoader '''
    return len(dataloader)/dataloader.batch_size()

def get_str_results(epoch=None, train_loss=None, valid_loss=None, time=None, lossF = None, lossT = None):
    '''construct the string that has to be print at the end of the epoch'''
    to_print=''

    if epoch is not None:
        to_print += 'epoch: {:3d} '.format(epoch)
    
    if train_loss is not None:
        to_print += '- train_loss: {:6.4f} '.format(train_loss[-1])
                        
    if valid_loss is not None:
        to_print += '- test_loss: {:6.4f} '.format(valid_loss[-1])

    if time is not None:
        to_print += '- time: {:6.4f} s'.format(time)

    if lossF is not None:
        to_print += '- lossF: {:6.4f}'.format(lossF) 

    if lossT is not None:
        to_print += '- lossT: {:6.4f}'.format(lossT) 

    return to_print





def get_edc(h: ndarray) -> ndarray:
    """
    Returns the normalized energy decay curve (EDC) in dB for a 1D impulse response.
    """
    assert h.ndim == 1
    edc = np.cumsum((np.abs(h) ** 2)[::-1])[::-1]
    edc /= np.max(edc)
    return 10 * np.log10(edc + 1e-10)


def decaytime_from_slopes(
    dfn_output: List[ndarray],
    thresh: float = -60,
) -> Tuple[ndarray, ndarray]:
    """
    Combines multiple decay slopes into a T60 estimate and interpolated EDC.
    """
    t = np.linspace(0, 4.0, 40001)
    c = 10 * np.log10(dfn_output[1])
    m = -60 / dfn_output[0]
    edcs = m * t[:, None, None] + c
    edcs_lin = np.power(10, edcs * 0.1)
    edc_sum = 10 * np.log10(np.sum(edcs_lin, axis=-1) + 1e-10)
    edc_sum_n = edc_sum - edc_sum.max(axis=0, keepdims=True)
    t_out = np.linspace(0, 1.96162498, 100)
    edc_interp = np.stack([np.interp(t_out, t, ee) for ee in edc_sum_n.T])
    t60 = np.argmin(edc_sum_n > thresh, axis=0) * 0.0001

    # # DBG plot
    # import matplotlib.pyplot as plt

    # # two subplots
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(t, edc_sum_n)
    # plt.subplot(1, 2, 2)
    # plt.plot(t_out, edc_interp.T)
    # plt.savefig("plots/dummy.png")
    # plt.close()

    return t60, edc_interp


def decaytime_from_edc(h: ndarray, thresh: float = -30, fs: int = 16000) -> float:
    """
    Returns decay time (s) from the EDC crossing a threshold.
    """
    edc = get_edc(h)
    return np.argmin(edc >= thresh) / fs

def decaytime_from_edc(
    h: ndarray, 
    fs: int, 
    decay_start_db: float = -5, 
    decay_end_db: float = -35
) -> float:
    """
    Estimate the reverberation time (RT60) from an Energy Decay Curve (EDC) using linear regression.
    
    RT60 is the time required for the sound pressure level to decrease by 60 dB.
    This function estimates RT60 by fitting a linear regression to the decay portion
    of the energy decay curve.
    """
    edc_db = get_edc(h)
    time = np.arange(len(edc_db)) / fs
    valid_range = (edc_db < decay_start_db) & (edc_db > decay_end_db)
    
    if not np.any(valid_range):
        return float('inf'), 0.0, 0.0, valid_range
    
    time_valid = time[valid_range.squeeze()]
    edc_valid = edc_db[valid_range]

    slope, intercept, *_ = linregress(time_valid, edc_valid)
    rt60 = -60 / slope if slope != 0 else float('inf')
    
    return rt60

class RIRParameters:
    """
    Analyze Room Impulse Responses (RIRs) and extract acoustic parameters.
    """

    def __init__(
        self,
        fs: int = 16000,
        center_freqs: List[float] = [125, 250, 500, 1000, 2000, 4000, 8000],
    ) -> None:
        """
        fs: Sampling rate. center_freqs: Octave band centers.
        """
        self.fs = fs
        self.center_freqs = center_freqs

        # filterbank freqs
        self.oct_freqs = generate_fractional_octaves(
            f_min=125,
            f_max=8000,
            fraction=1,
        )
        self.third_oct_freqs = generate_fractional_octaves(
            f_min=62.5,
            f_max=8000,
            fraction=3,
        )

        # filterbanks
        self.oct_fb = OctaveFilterbank(
            fs=self.fs,
            center_freqs=self.oct_freqs,
        )
        self.third_oct_fb = OctaveFilterbank(
            fs=self.fs,
            center_freqs=self.third_oct_freqs,
        )

        # store 1k indices for relative magnitude
        self.ind_1k_oct = 3
        self.ind_1k_third_oct = 12

    def compute_c50(self, h: ndarray) -> float:
        """
        Returns clarity index C50 (dB) for an impulse response.
        """
        ind_dir = get_onset(rir=h)
        ind_50 = min(max(ind_dir + int(np.round(self.fs * 0.05)), 0), h.shape[0])
        c50 = np.sum(h[ind_dir:ind_50] ** 2) / (1e-12 + np.sum(h[ind_50:] ** 2))
        c50 = 10 * np.log10(c50)
        return c50

    def compute_edt(self, dfn_params: List[ndarray]) -> ndarray:
        """
        Returns Early Decay Time (EDT) from decay fit network parameters.
        """
        t1 = decaytime_from_slopes(dfn_params, -1)[0]
        t11 = decaytime_from_slopes(dfn_params, -11)[0]
        return 6 * (t11 - t1)

    def compute_drr(self, h: ndarray, fs: int = 48000) -> float:
        """
        Computes the Direct-to-Reverberant Ratio (DRR) of an impulse response.
        """

        onset = max(0, get_onset(h))
        guard = int(fs / 1000)

        direct_energy = np.sum(h[: onset + guard] ** 2)
        reverberant_energy = np.sum(h[onset + guard :] ** 2)
        drr = 10 * np.log10(direct_energy / reverberant_energy)
        return drr

    def analyze(self, h: ndarray) -> dict:
        """
        Returns octave-band C50, EDT, T60, T30, and normalized band energies.
        """
        assert h.ndim == 1, "Impulse response must be 1D."

        h_oct = self.oct_fb(h)
        h_third_oct = self.third_oct_fb(h)

        t30 = np.array([decaytime_from_edc(band, self.fs, -5, -35) for band in h_oct])

        c50 = np.array([self.compute_c50(band) for band in h_oct])

        # relative magnitudes
        mag_oct = 10 * np.log10(np.sum(h_oct**2, axis=1))
        mag_oct = -(mag_oct - mag_oct[self.ind_1k_oct])
        mag_third_oct = 10 * np.log10(np.sum(h_third_oct**2, axis=1))
        mag_third_oct = -(mag_third_oct - mag_third_oct[self.ind_1k_third_oct])

        # drr
        drr = self.compute_drr(h, fs=self.fs)
        return {
            "c50": c50,
            "t30": t30,
            "drr": drr,
            "mag_oct": mag_oct,
            "mag_third_oct": mag_third_oct,
            "freq_oct": self.oct_freqs,
            "freq_third_oct": self.third_oct_freqs,
        }


class OctaveFilterbank:
    def __init__(
        self,
        fs: float,
        center_freqs: List[float],
        fraction: int = 1,
        order: int = 3,
    ) -> None:
        """
        Butterworth filterbank for 1/fraction-octave bands.
        """
        self.fs = fs
        self.center_freqs = center_freqs
        self.fraction = fraction
        self.order = order
        self.sos = []
        self.band_edges = []
        half_band_exp = 1.0 / (2.0 * fraction)
        for freq_c in center_freqs:
            flims = freq_c * 2 ** (np.array([-half_band_exp, +half_band_exp]))
            flims[0] = max(flims[0], 0)
            flims[1] = min(flims[1], fs / 2)
            if flims[1] <= 0:
                raise ValueError(
                    f"Invalid band edges [{flims[0]:.2f}, {flims[1]:.2f}] for center freq {freq_c:.2f} Hz. "
                    f"Check fraction={fraction} and fs={fs}."
                )
            if flims[0] <= 0 and flims[1] < fs / 2:
                z, p, k = butter(
                    self.order, 2 * flims[1] / fs, btype="low", output="zpk"
                )
            elif flims[1] >= fs / 2 and flims[0] > 0:
                z, p, k = butter(
                    self.order, 2 * flims[0] / fs, btype="high", output="zpk"
                )
            else:
                z, p, k = butter(self.order, 2 * flims / fs, btype="band", output="zpk")
            self.sos.append(zpk2sos(z, p, k))
            self.band_edges.append((flims[0], flims[1]))

    def __call__(self, signal: ndarray) -> ndarray:
        """
        Returns stacked filtered signals for all bands.
        """
        return np.stack([sosfilt(sos, signal) for sos in self.sos])


def normalize_data_at_freq(data, freqs, ref_freq=1000, in_dB=False):
    """
    Normalizes 'data' so its value at ref_freq is 0 dB (if in_dB) or 1.0 (if not).
    """
    data = np.asarray(data)
    freqs = np.asarray(freqs)
    ref_value = np.interp(ref_freq, freqs, data)
    if in_dB:
        data_norm = data - ref_value
    else:
        if ref_value < 1e-20:
            ref_value = 1e-20
        data_norm = data / ref_value
    return data_norm


def compute_band_energy(
    h: ndarray,
    fs: float,
    center_freqs: List[float],
    order: int = 2,
    normalize_by_bandwidth: bool = True,
    ref_freq=1e3,
) -> ndarray:
    """
    Returns normalized band energies for h, optionally per Hz.
    """
    fb = OctaveFilterbank(fs, center_freqs, order=order)
    filtered_bands = fb(h)
    energies = np.array([np.sum(band**2) for band in filtered_bands])
    if normalize_by_bandwidth:
        bandwidths = []
        for f_low, f_high in fb.band_edges:
            if f_high > f_low:
                bandwidths.append(f_high - f_low)
            else:
                bandwidths.append(1.0)
        energies = energies / np.array(bandwidths)
    energies = normalize_data_at_freq(energies, center_freqs, ref_freq=ref_freq)
    return energies


def generate_fractional_octaves(f_min=125, f_max=8000, fraction=1):
    """
    Returns center frequencies for 1/fraction-octave bands from f_min to f_max.
    """
    center_freqs = []
    f = f_min
    band_step = 2 ** (1.0 / fraction)
    while round(f, 6) <= f_max * 1.01:
        center_freqs.append(f)
        f *= band_step
    return np.array(center_freqs)


def fractional_octave_smoothing(
    H_dB: ndarray, freq: ndarray, fraction: int = 3
) -> ndarray:
    """
    Smooths H_dB by averaging over 1/fraction-octave bands.
    """
    smoothed = np.zeros_like(H_dB)
    for i, f_center in enumerate(freq):
        if f_center <= 0:
            smoothed[i] = H_dB[i]
            continue
        factor = 2 ** (1 / (2 * fraction))
        f_low = f_center / factor
        f_high = f_center * factor
        idx = np.where((freq >= f_low) & (freq <= f_high))[0]
        smoothed[i] = np.mean(H_dB[idx]) if len(idx) > 0 else H_dB[i]
    return smoothed
