# differentiable digital signal processing blocks
# comput
import torch 
import torch.nn as nn
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

def SAP(x, m, gamma):
    ''' differentiable Schoreder allpass filter 
    x: frequency sampling points 
    m: delay lengths
    gamma: feed-forward/back gains'''
    M = m.size(0) # number of channels
    K = m.size(1) # number of cascaded filters 
    bs = gamma.size(0) # batch size
    # compute transfer function of first filter
    zK = torch.pow(x.expand(M,-1).permute(1, 0),-m[:,0]).expand(bs, -1, -1)     # this step instroudces some numerical errors so that the absolute value is no longer <= 1
    gammaK = (gamma[:,:,0].expand(x.size(0), -1, -1)).permute(1, 0, 2)
    H = torch.div(
        (gammaK + zK), 
        (1 + gammaK * zK))
    # compute all the other SAP filters in series 
    for k in range(1, K):
        zK = torch.pow(x.expand(M,-1).permute(1, 0),-m[:,k]).expand(bs, -1, -1)
        gammaK = (gamma[:,:,k].expand(x.size(0), -1, -1)).permute(1, 0, 2)
        Hi = torch.div(
            (gammaK + zK),
            (1 + gammaK * zK))
        # element-wise mul to compute overall system's transfer function
        H = torch.mul(H, Hi)    
    return H 

def PEQ(x, f, R, G):
    ''' differentiable parameteric equilizer
    x: frequency sampling points
    f: cut off frequency 
    R: resonance 
    G: component gain
    f entries at initialization must be in ascending order '''
    # TODO: for some reason only when coefficents alpha and beta are swapped 
    # we obtain the proper response

    K = f.size(1)   # number of filters in series
    bs = f.size(0)   # number of channels
    # prevent the shelf filters magniture response to spkie over 1
    R[:,0] = R[:,0] + 1/torch.sqrt(torch.tensor(2, device=get_device())) 
    R[:,-1] = R[:,-1] + 1/torch.sqrt(torch.tensor(2, device=get_device())) 
    # TODO: it seems that low values of R give better looking respononses (low: close to 1/sqrt(2))
    # maybe the activation should not push R to high values 
    # Watch out that the formula used for the  biquad coeff of LP and HP filters are swapped 
    # low shelf filter (as flipped high shelf)
    # G[:,0] = G[:,0] - torch.tensor(0.001, device=get_device())   # this is needed to prevent the low freq from being aplifyed
    betaLP, alphaLP = compute_biquad_coeff(
        f[:,0], R[:,0], torch.tensor(1, device=get_device()), 2 * R[:,0] * torch.sqrt(G[:,0]), G[:,0] )
        # f[:,0], R[:,0], G[:,0], 2 * R[:,0] * torch.sqrt(G[:,0]), torch.tensor(1, device=get_device()) )
    HHP = biquad_to_tf(x, betaLP[:,:,0], alphaLP[:,:,0]) - 1e-5
    H = HHP 

    # high shelf filter (as flipped low shelf)
    
    betaHP, alphaHP = compute_biquad_coeff(
        f[:,-1], R[:,-1], G[:,-1], 2 * R[:,0] * torch.sqrt(G[:,-1]), torch.tensor(1, device=get_device()) )
        # f[:,-1], R[:,-1], torch.tensor(1, device=get_device()), 2 * R[:,-1] * torch.sqrt(G[:,-1]), G[:,-1] )  
    HLP = biquad_to_tf(x, betaHP[:,:,0], alphaHP[:,:,0]) - 1e-5
    H = H*HLP

    # K - 2 peaking filter 
    for k in range(1, K-1):
        beta, alpha = compute_biquad_coeff(
            f[:,k], R[:,k], torch.tensor(1, device=get_device()), 2*R[:,k]*G[:,k], torch.tensor(1, device=get_device()))
        Hp = biquad_to_tf(x, beta[:,:,0], alpha[:,:,0]) - 1e-5
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

def compute_biquad_coeff(f, R, mLP, mBP, mHP):
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