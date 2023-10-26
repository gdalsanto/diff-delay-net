# differentiable digital signal processing blocks
# comput
import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.utility import get_device

def SVF(x, f, R, mLP, mBP, mHP):
    ''' differentiable implementation of iir filter as serialy cascaded 
    state variable filters (SVRs)
    x: frequency sampling points
    f: cut off frequency 
    R: resonance 
    mLP, mBP, mHP: mixing coefficents'''
    K = f.size(1) # number of cascaded filters 
    bs = f.size(0) # batch size 
    # compute biquad filter parameters  
    beta = torch.zeros((bs, 3, K), device=get_device())     
    alpha = torch.zeros((bs, 3, K), device=get_device())  

    beta[:,0,:] = (f**2) * mLP + f * mBP + mHP
    beta[:,1,:] = 2*(f**2)*mLP - 2*mHP
    beta[:,2,:] = (f**2) * mLP - f * mBP + mHP

    alpha[:,0,:] = f**2 + 2*R*f + 1
    alpha[:,1,:] = 2*f**2 - 2
    alpha[:,2,:] = f**2 - 2*R*f + 1
    # convert parameters to complex 
    beta = torch.complex(beta, torch.zeros(beta.size(), device=get_device()))
    alpha = torch.complex(alpha, torch.zeros(alpha.size(), device=get_device()))

    H = biquad_to_tf(x, beta[:,:,0], alpha[:,:,0])
    for k in range(1, K):
        Hi = biquad_to_tf(x, beta[:,:,k], alpha[:,:,k])
        H = torch.mul(H, Hi)  
    return H 

def SAP(x, m, gamma,  eps = 1e-10):
    ''' differentiable Schoreder allpass filter 
    x: frequency sampling points 
    m: delay lengths
    gamma: feed-forward/back gains'''
    M = m.size(0) # number of channels
    K = m.size(1) # number of cascaded filters 
    bs = gamma.size(0) # batch size
    num = (x.size(0) - 1) * 2 # TODO to be fixed
    # compute transfer function of first filter
    # zK = torch.pow(x.expand(M,-1).permute(1, 0),-m[:,0]).expand(bs, -1, -1)
    zK = get_delays(num, m[:,0].reshape(M, 1))
    gammaK = (gamma[:,:,0].expand(x.size(0), -1, -1)).permute(1, 0, 2)
    H = torch.div(
        (gammaK + zK), 
        (1 + gammaK * zK) + eps)
    # compute all the other SAP filters in series 
    for k in range(1, K):
        # zK = torch.pow(x.expand(M,-1).permute(1, 0),-m[:,k]).expand(bs, -1, -1)
        zK = get_delays(num, m[:,k].reshape(M, 1))
        gammaK = (gamma[:,:,k].expand(x.size(0), -1, -1)).permute(1, 0, 2)
        Hi = torch.div(
            (gammaK + zK),
            (1 + gammaK * zK) + eps)
        # element-wise mul to compute overall system's transfer function
        H = torch.mul(H, Hi)    
    return H 

def get_delays(num, m):

    idxs = torch.arange(num // 2 + 1, device=get_device()).unsqueeze(-2)
    phase = m.unsqueeze(-1) * idxs / num * 2 * np.pi
    delay = torch.view_as_complex(torch.stack([torch.cos(phase), -torch.sin(phase)], -1))
    return delay.permute(1, 2, 0)

def PEQ(x, f, R, G):
    ''' differentiable parameteric equilizer
    x: frequency sampling points
    f: cut off frequency 
    R: resonance 
    G: component gain
    f entries at initialization must be in ascending order '''
    K = f.size(1)   # number of filters in series
    bs = f.size(0)   # number of channels
    # TODO: it seems that low values of R give better looking respononses (low: close to 1/sqrt(2))
    # maybe the activation should not push R to high values 
    # Watch out that for Cdelta the gain is <0db so the filters are "fipped vertically"
    betaLP, alphaLP = compute_biquad_coeff(
        f[:,0], R[:,0], torch.tensor(1, device=get_device()), 2 * R[:,0] * torch.sqrt(G[:,0]), G[:,0] )
        # f[:,0], R[:,0], G[:,0], 2 * R[:,0] * torch.sqrt(G[:,0]), torch.tensor(1, device=get_device()) )
    HHP = biquad_to_tf(x, betaLP[:,:,0], alphaLP[:,:,0]) 
    H = HHP 

    # high shelf filter (as flipped low shelf)
    
    betaHP, alphaHP = compute_biquad_coeff(
        f[:,-1], R[:,-1], G[:,-1], 2 * R[:,-1] * torch.sqrt(G[:,-1]), torch.tensor(1, device=get_device()) )
        # f[:,-1], R[:,-1], torch.tensor(1, device=get_device()), 2 * R[:,-1] * torch.sqrt(G[:,-1]), G[:,-1] )  
    HLP = biquad_to_tf(x, betaHP[:,:,0], alphaHP[:,:,0]) 
    H = H*HLP

    # K - 2 peaking filter 
    for k in range(1, K-1):
        beta, alpha = compute_biquad_coeff(
            f[:,k], R[:,k], torch.tensor(1, device=get_device()), 2*R[:,k]*G[:,k], torch.tensor(1, device=get_device()))
        Hp = biquad_to_tf(x, beta[:,:,0], alpha[:,:,0])
        H = H*Hp
    return H

def biquad_to_tf(x, beta, alpha):
    # TODO: too many transpose operations. they can be removed
        H = torch.div(
                torch.matmul(
                    torch.pow(x.expand((3 ,-1)).transpose(1, 0), 
                        torch.tensor([0, -1, -2], device=get_device())),
                    beta.transpose(1,0)),
                torch.matmul(
                    torch.pow(x.expand((3 ,-1)).transpose(1, 0), 
                        torch.tensor([0, -1, -2], device=get_device())),
                    alpha.transpose(1,0))
            )
        return H.transpose(1, 0) 

def compute_biquad_coeff(f, R, mLP, mBP, mHP): #ok
    if f.dim() == 1:
        f = torch.unsqueeze(f, dim=-1)
        R = torch.unsqueeze(R, dim=-1)
        mLP = torch.unsqueeze(mLP, dim=-1)
        mBP = torch.unsqueeze(mBP, dim=-1)
        mHP = torch.unsqueeze(mHP, dim=-1)

    K = f.size(1) # number of cascaded filters
    bs = f.size(0) # batch size  
    beta = torch.zeros(bs, 3, K, device=get_device())     
    alpha = torch.zeros(bs, 3, K, device=get_device())  

    beta[:,0,:] = (f**2) * mLP + f * mBP + mHP
    beta[:,1,:] = 2*(f**2) * mLP - 2 * mHP
    beta[:,2,:] = (f**2) * mLP - f * mBP + mHP
    alpha[:,0,:] = f**2 + 2*R*f + 1
    alpha[:,1,:] = 2* (f**2) - 2
    alpha[:,2,:] = f**2 - 2*R*f + 1  

    beta = torch.complex(beta, torch.zeros(beta.size(), device=get_device()))
    alpha = torch.complex(alpha, torch.zeros(alpha.size(), device=get_device()))
    return beta, alpha