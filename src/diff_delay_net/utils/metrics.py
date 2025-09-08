import json
from typing import List, Dict
from pathlib import Path

import numpy as np
import torch
import pandas as pd
from scipy.linalg import sqrtm

from torchaudio.transforms import Resample
from torchaudio.prototype.pipelines import VGGISH

from utils.utility import RIRParameters

def compute_fad(outputs: List[dict]) -> float:

    # Initialize VGGISH components
    input_sr = VGGISH.sample_rate  # 16kHz
    input_proc = VGGISH.get_input_processor()
    model = VGGISH.get_model()
    model.eval()

    # Resampler from 48kHz to 16kHz
    resample = Resample(orig_freq=48000, new_freq=input_sr)

    tru_embs, fdn_embs = [], []

    for step in outputs:
        # resample to 16kHz and flatten for VGGISH
        wet = resample(torch.tensor(step["wet"])).reshape(-1)
        fdn = resample(torch.tensor(step["wet_fdn"])).reshape(-1)

        tru_embs.append(model(input_proc(wet)).detach().cpu())
        fdn_embs.append(model(input_proc(fdn)).detach().cpu())

    tru_embs = torch.cat(tru_embs, dim=0).numpy()
    fdn_embs = torch.cat(fdn_embs, dim=0).numpy()

    # Compute mean and covariance of real and synthetic embeddings
    mu_tru, sigma_tru = np.mean(tru_embs, axis=0), np.cov(tru_embs, rowvar=False)
    mu_fdn, sigma_fdn = np.mean(fdn_embs, axis=0), np.cov(fdn_embs, rowvar=False)

    # compute fid
    diff = mu_tru - mu_fdn
    covmean = sqrtm(sigma_tru @ sigma_fdn)
    # covmean = covmean.real

    fad = diff @ diff + np.trace(sigma_tru + sigma_fdn - 2 * covmean)
    return float(fad)


def compute_rir_metrics(outputs: List[dict]) -> Dict:

    rir_params = RIRParameters(fs=48000)
    params, params_fdn = [], []

    # ensure all outputs have the same shape
    for output in outputs:
        if output["rir"].ndim == 1:
            output["rir"] = output["rir"][None, ...]
        if output["rir_fdn"].ndim == 1:
            output["rir_fdn"] = output["rir_fdn"][None, ...]

        for rir, rir_fdn in zip(output["rir"], output["rir_fdn"]):
            params_fdn.append(rir_params.analyze(rir_fdn))
            params.append(rir_params.analyze(rir))

    t30 = np.array([p["t30"] for p in params])
    t30_fdn = np.array([p["t30"] for p in params_fdn])
    c50 = np.array([p["c50"] for p in params])
    c50_fdn = np.array([p["c50"] for p in params_fdn])

    # compute mape for t30
    t30_mape = np.mean(np.abs(t30 - t30_fdn) / (np.abs(t30_fdn) + 1e-6), axis=0) * 100
    # compute mae for c50
    c50_mae = np.mean(np.abs(c50 - c50_fdn), axis=0)

    # compute pearson correlation
    t30_corr = np.array(
        [np.corrcoef(tru, pred)[0, 1] for tru, pred in zip(t30.T, t30_fdn.T)]
    )
    c50_corr = np.array(
        [np.corrcoef(tru, pred)[0, 1] for tru, pred in zip(c50.T, c50_fdn.T)]
    )

    return {
        "t30_mape": t30_mape.tolist(),
        "c50_mae": c50_mae.tolist(),
        "t30_corr": t30_corr.tolist(),
        "c50_corr": c50_corr.tolist(),
    }

def compute_speech2fdn_metrics(outputs: List[dict], out_dir: Path):
    # fad = compute_fad(outputs)
    rir_metrics = compute_rir_metrics(outputs)
    with open(out_dir + "/metrics.json", "w") as f:
        # json.dump({"fad": fad, **rir_metrics}, f, indent=2)
        json.dump({**rir_metrics}, f, indent=2)
    print(f"Metrics saved to {out_dir + '/metrics.json'}")
