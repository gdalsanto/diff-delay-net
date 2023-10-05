# differentiable delay network 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils.utility import get_device


class SVF(nn.Module):
    def __init__(self, K):
        super().__init__() 
        ''' differentiable implementation of iir filter as serialy cascaded 
        state variable filters (SVRs)
        K: number of filters '''
        self.K = K
        self.f = nn.Parameter(torch.randn(1, self.K))
        self.R = nn.Parameter(torch.randn(1, self.K))
        self.mLP = nn.Parameter(torch.randn(1, self.K))
        self.mBP = nn.Parameter(torch.randn(1, self.K))
        self.mHP = nn.Parameter(torch.randn(1, self.K))

    def forward(self, x):
        ''' x: sampling points '''
        # compute biquad filter parameters  
        beta = torch.zeros((3, self.K), device=get_device())     
        alpha = torch.zeros((3, self.K), device=get_device())  
        beta[0,:] = (self.f**2) * self.mLP + \
            self.f * self.mBP + self.mHP
        beta[1,:] = 2*(self.f**2)*self.mLP - 2*self.mHP
        beta[2,:] = (self.f**2) * self.mLP - \
            self.f * self.mBP + self.mHP
        alpha[0,:] = self.f**2 + 2*self.R*self.f + 1
        alpha[1,:] = 2*self.f**2 - 2
        alpha[2,:] = self.f**2 - 2*self.R*self.f + 1
        # convert parameters to complex 
        beta = torch.complex(beta, torch.zeros((3, self.K), device=get_device()))
        alpha = torch.complex(alpha, torch.zeros((3, self.K), device=get_device()))
        H = torch.complex(torch.ones(x.size(), device=get_device()),torch.ones(x.size(), device=get_device()))
        for k in range(self.K):
            Hi = torch.div(
                    torch.matmul(
                        torch.pow(x.expand((3 ,-1)).transpose(1, 0), 
                        torch.tensor([0, -1, -2], device=get_device())),
                        beta[:,k]),
                    torch.matmul(
                        torch.pow(x.expand((3 ,-1)).transpose(1, 0), 
                        torch.tensor([0, -1, -2], device=get_device())),
                        alpha[:,k])
            )
            H = H*Hi
        return H


class SAP(nn.Module):
    def __init__(self, m):
        super().__init__()
        ''' differentiable Schoreder allpass filter 
            m: delay lengths'''
        self.m = m 
        self.K = m.size(0)

        self.gamma = nn.Parameter(torch.randn(self.K))

    def forward(self, x):

        H = torch.complex(torch.ones(x.size(), device=get_device()),torch.ones(x.size(), device=get_device()))
        for k in range(self.K):
            Hi = torch.div(
                (1 + self.gamma[k] * x ** (-self.m[k])),
                (self.gamma[k] + x ** (-self.m[k]))
            )
            H = H*Hi
        return H

class PEQ(nn.Module):
    def __init__(self, K):
        super().__init__()
        ''' differentiable parametric equilizer 
            K: number of filters'''
        
        self.K = K
        self.f = nn.Parameter(torch.randn(self.K))
        self.R = nn.Parameter(torch.randn(self.K))
        self.G = nn.Parameter(torch.randn(self.K))

    def forward(self, x):
        ''' x: sampling points '''

        H = torch.complex(torch.ones(x.size(), device=get_device()),torch.ones(x.size(), device=get_device()))
        # low shelf filter 
        betaLP, alphaLP = self.compute_biquad_coeff(
            self.f[0], self.R[0], self.G[0], 2 * self.R[0] * torch.sqrt(self.G[0]), 1)  
        HLP = self.compute_tf(x, betaLP, alphaLP)
        H = H*HLP

        # high shelf filter 
        betaHP, alphaHP = self.compute_biquad_coeff(
            self.f[1], self.R[1], 1, 2 * self.R[1] * torch.sqrt(self.G[1]), self.G[1])
        HHP = self.compute_tf(x, betaHP, alphaHP)
        H = H*HHP

        # K - 2 peaking filter 
        for k in range(2, self.K):
            beta, alpha = self.compute_biquad_coeff(
                self.f[k], self.R[k], 1, 2*self.R[k]*self.G[k], 1)
            Hp = self.compute_tf(x, beta, alpha)
            H = H*Hp
        return H

    def compute_biquad_coeff(self, f, R, mLP, mBP, mHP):
        beta = torch.zeros(3, device=get_device())     
        alpha = torch.zeros(3, device=get_device())  
        beta[0] = (f**2) * mLP + f * mBP + mHP
        beta[1] = 2*(f**2) * mLP - 2 * mHP
        beta[2] = (f**2) * mLP - f * mBP + mHP
        alpha[0] = f**2 + 2*R*f + 1
        alpha[1] = 2* (f**2) - 2
        alpha[2] = f**2 - 2*R*f + 1  
        beta = torch.complex(beta, torch.zeros(3, device=get_device()))
        alpha = torch.complex(alpha, torch.zeros(3, device=get_device()))     
        return beta, alpha

    def compute_tf(self, x, beta, alpha):
        H = torch.div(
                    torch.matmul(
                        torch.pow(x.expand((3 ,-1)).transpose(1, 0), 
                        torch.tensor([0, -1, -2], device=get_device())),
                        beta),
                    torch.matmul(
                        torch.pow(x.expand((3 ,-1)).transpose(1, 0), 
                        torch.tensor([0, -1, -2], device=get_device())),
                        alpha)
            )
        return H