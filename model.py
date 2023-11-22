import torch
import torch.nn as nn 
import torch.nn.functional as F
from nnAudio import features
from diff_dsp import * 
from utils.processing import householder_matrix as householder
from utils.utility import get_device
from einops import rearrange

# MODEL/TASK_AGNOSTIC ENCODER 
class Encoder(nn.Module):
    def __init__(self, n_fft=1024, sr=48000, overlap=0.875):
        super().__init__()
        ''' model/task agnostic decoder '''
        hop_length = int(n_fft*(1-overlap))
        self.stft = features.stft.STFT(
            n_fft = n_fft,
            hop_length = hop_length,
            window = 'hann',
            freq_scale = 'log',
            sr = sr,
            fmin = 20,
            fmax = 12000, #sr // 2,
            output_format = 'Magnitude',
            verbose=False
        )
        self.conv_depth = 5
        self.chn_out = [64, 128, 128, 128, 128]  
        self.kernel = [(7,5), (5,5), (5,5), (5,5), (5,5)]
        self.strides = [(1,2), (2,1), (2,2), (2,2), (1,1)]

        self.conv_list = nn.ModuleList([])
        for i in range(self.conv_depth):
            if i == 0:
                chn_in = 1
            else:
                chn_in = self.chn_out[i-1]
            
            self.conv_list.append(nn.ModuleList(
                [conv2d_block(chn_in, self.chn_out[i], self.kernel[i], self.strides[i])]
            ))
        
        self.gru1 = nn.GRU(input_size=128, num_layers=2, hidden_size=64, 
            batch_first = True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=7168, num_layers=1, hidden_size=128,
            batch_first = True, bidirectional=True)

        self.lin_depth = 2
        in_feat, out_feat = 256, 256
        self.lin_list = nn.ModuleList([])
        for i in range(self.lin_depth):
            self.lin_list.append(nn.ModuleList(
                [linear_block(in_feat, out_feat)]
            ))
    
    def forward(self, x):
        b = x.shape[0]
        # convert to log-freq log-mag stft 
        x = torch.log(self.stft(x) + 1e-7)
        # add channel dimension 
        x = torch.unsqueeze(x, 1)

        # 1. stack of 5 convolutional layer + relu
        for i, module in enumerate(self.conv_list):
            x = module[0](x)
        
        # 2. GRUs
        x = rearrange(x, 'b c f t -> (b t) f c')
        x = rearrange(self.gru1(x)[0], '(b t) f c -> b t (f c)', b=b)
        x = self.gru2(x)[0]

        # 3. stack of 2 linear layaer + layernorm + relu
        for i, module in enumerate(self.lin_list):
            x = module[0](x)          
        return x 
            

class conv2d_block(nn.Module):
    def __init__(self, chn_in, chn_out, kernel, stride):
        super().__init__()
        ''' 2D convolutional layer + relu activation '''
        self.conv2d = nn.Conv2d(chn_in, chn_out, kernel, stride)

    def forward(self, x):
        x = self.conv2d(x)
        y = F.relu(x)
        return y 

class linear_block(nn.Module):
    def __init__(self, in_feat, out_feat):
        super().__init__()
        ''' Linear + layerNorm + relu activation '''
        self.linear = nn.Linear(in_feat, out_feat)
        self.layer_norm = nn.LayerNorm(out_feat)

    def forward(self, x):
        x = self.linear(x)
        x = self.layer_norm(x)
        y = F.relu(x)
        return y 

class ProjectionLayer(nn.Module):
    def __init__(self, in_feats, z1=1, z2=8, bias=None, activation=None):
        super().__init__()

        self.linear1 = nn.Linear(in_feats[0], z1)
        self.linear2 = nn.Linear(in_feats[1], z2)
        # initialize bias 
        if bias is not None:
            self.linear2.bias.data.fill_(0)
            self.linear2.bias = nn.Parameter(bias)
        else:
            self.linear1.bias.data.fill_(0)
            self.linear2.bias.data.fill_(0)
        self.activation = activation

    def forward(self, x):
        x = torch.transpose(x, 2, 1)
        x = self.linear1(x)
        x = torch.transpose(x, 2, 1)
        y = self.linear2(x)
        # nonlinear activation
        if self.activation is not None:
            y = self.activation(y)
        return y


# ARP ESTIMATION NETWORK 
class ASPestNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.sigmoid = nn.Sigmoid()
        z1, z2 = 1, 8
        self.sr = 48000
        omega_min, omega_max = 40, 12000 
        
        k = torch.arange(1,z2+1,1)
        # initial cutoff freqs such that the freqs are equally spaced in the logarithmic scale 
        omegaK = 2*(omega_min*(omega_max/omega_min)**((k-1)/(z2-1)))/self.sr 
        # inverse of the sigmoid function 
        bias_f = lambda x: torch.log(x/(1-x))

        #  investigate on the weight initialization. using the default pytorch init 
        # might shuffles the frequencies.
        self.fC1ProjLayer = ProjectionLayer(
            (76, 256), 1, 8, 
            bias = bias_f(omegaK),
            activation = lambda x: torch.tan(torch.pi * self.sigmoid(x)/2))
            # activation = lambda x: torch.tan(torch.pi * self.sigmoid(
            #   torch.tan(torch.pi * x / 44100 )) / 2))
        self.fCdeltaProjLayer = ProjectionLayer(
            (76, 256), 1, 8, 
            bias = bias_f(omegaK),
            activation = lambda x: torch.tan(torch.pi * self.sigmoid(x)/2))           

        self.RC1ProjLayer = ProjectionLayer(
            (76, 256), 1, 8,
            activation = lambda x: torch.log(1+torch.exp(x)) / torch.log(torch.tensor(2,  device=get_device())))
        bias = torch.ones((3, 8), device=get_device())
        bias[1, :] = 2*torch.ones((1, 8), device=get_device())
        self.mC1ProjLayer = ProjectionLayer(
            (76, 256), 3, 8,
            bias = bias)

        self.GCdeltaProjLayer = ProjectionLayer(
            (76, 256), 1, 8, 
            bias = -10*torch.ones((z1, z2), device=get_device()), 
            activation = lambda x: 10**(-torch.log(1+torch.exp(x)) / torch.log(torch.tensor(2,  device=get_device()))))
        self.RCdeltaProjLayer = ProjectionLayer(
            (76, 256), 1, 8,
            activation = lambda x: torch.log(1+torch.exp(x))  / torch.log(torch.tensor(2,  device=get_device()))) 
            

        self.SAProjLayer = ProjectionLayer(
            (76, 256), 6, 4, 
            activation = self.sigmoid)
        self.bcProjLayer = ProjectionLayer(
            (76, 256), 2, 6)
        self.hProjLayer = ProjectionLayer(
            (76, 256), 1, 232)  # in the original paper it was 100 

        # delay lengths
        self.d = torch.tensor([233, 311, 421, 461, 587, 613],  device=get_device())
        self.M = self.d.size(0)
        self.Q0 = torch.tensor(householder(self.M), device=get_device())  # TODO: when to change Houlsehoöder matrix ? 

        # all pass filter delay lengths
        self.dAP = torch.tensor([
            [131, 151, 337, 353], 
            [103, 173, 331,373], 
            [89, 181, 307, 401], 
            [79, 197, 281, 419], 
            [61, 211, 257, 431], 
            [47, 229, 251, 443]],  device=get_device())   
        # length of IR  
        self.ir_length = int(1.8*self.sr)

    def forward(self, x, z):
        bs = x.size(0)  # batch size
        # STFT 
        # x = torch.stft(x, n_fft=1024, hop_length=128, window=torch.hann_window(1024), return_complex = True)
        # Model/Task-Agnostic Encoder
        x = self.encoder(x) # out: [bs, 109, 256] or [bs, 76, 256]
        # ARP-Groupwise Projection Laters
        bc = self.bcProjLayer(x)
        # bc = bc * self.bc_norm  # apply normalization term 
        b, c = bc[:, 0, :], bc[:, 1, :]
        b = torch.complex(b, torch.zeros(b.size(), device=get_device()))
        c = torch.complex(c, torch.zeros(c.size(),  device=get_device()))
        h0 = self.hProjLayer(x).squeeze(dim=1)
        # common post filter
        fC1 = self.fC1ProjLayer(x).squeeze(dim=1)
        RC1 = self.RC1ProjLayer(x).squeeze(dim=1)
        mC1 = self.mC1ProjLayer(x)
        C1 = SVF(z, fC1, RC1, mC1[:, 0, :], mC1[:, 1, :], mC1[:, 2, :])
        # common parallel delta-coloration filters 
        fCdelta = self.fCdeltaProjLayer(x).squeeze(dim=1)
        GCdelta = self.GCdeltaProjLayer(x).squeeze(dim=1)
        RCdelta = self.RCdeltaProjLayer(x).squeeze(dim=1)
        Cdelta = PEQ(z, fCdelta, RCdelta, GCdelta)
        Cdelta = Cdelta.expand(self.M, -1, -1).permute(1, 2, 0)
        gamma = self.SAProjLayer(x)
        U = SAP(z, self.dAP, gamma)
        # channel-wise allpass filters
       
        # delay matrix
        D = torch.diag_embed(torch.unsqueeze(z, dim=-1)  ** (self.d)).expand(bs, -1 ,-1, -1)
        # unitary matrix - Householder

        Q0 = self.Q0
        # Gamma = torch.diag(0.9999**self.d)
        # TODO find why U creates resonances and balance energy of the ealry reflections
        # H = torch.einsum('ik,ijkk->ijk', c, torch.inverse(D -  torch.diag_embed(Cdelta)*torch.matmul(Q0,Gamma)))
        H = torch.einsum('ik,ijkk->ijk', c, torch.inverse(D - torch.diag_embed(U*Cdelta)*Q0 + 1e-16))
        
        H = C1*torch.einsum('ik,ijk->ij', b, H)
        # channel-wise allpass filters 
        ir_late =  torch.fft.irfft(H,  norm='ortho')
        h0 = F.pad(h0, (0, self.ir_length-h0.size(dim=1)))
        ir = (h0 + ir_late[:,:self.ir_length])
        return ir, ir_late, h0
    
    def get_filters(self, x, z):
        x = self.encoder(x) # out: [bs, 109, 256]
        # get filters parameters and freuqency response as dictionary 
        parameters = {}
        # TODO parameters are missing 
        parameters['fC1'] = self.fC1ProjLayer(x).squeeze(dim=1)
        parameters['RC1'] = self.RC1ProjLayer(x).squeeze(dim=1)
        parameters['mC1'] = self.mC1ProjLayer(x)

        parameters['fCdelta'] = self.fCdeltaProjLayer(x).squeeze(dim=1)
        parameters['GCdelta'] = self.GCdeltaProjLayer(x).squeeze(dim=1)
        parameters['RCdelta'] = self.RCdeltaProjLayer(x).squeeze(dim=1)

        parameters['gamma'] = self.SAProjLayer(x)

        filters_tf = {}
        filters_tf['C1'] = SVF(
            z, 
            parameters['fC1'], 
            parameters['RC1'], 
            parameters['mC1'][:, 0, :], 
            parameters['mC1'][:, 1, :], 
            parameters['mC1'][:, 2, :]).detach().numpy()
        filters_tf['Cdelta'] = PEQ(
            z, 
            parameters['fCdelta'], 
            parameters['GCdelta'], 
            parameters['RCdelta']).detach().numpy()
        filters_tf['U'] = SAP(
            z, 
            self.dAP, 
            parameters['gamma']).detach().numpy()
        
        for param_key in parameters:
            try:
                parameters[param_key] = parameters[param_key].detach().numpy()
            except:
                continue
        return parameters, filters_tf