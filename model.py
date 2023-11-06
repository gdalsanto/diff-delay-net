import torch
import torch.nn as nn 
import torch.nn.functional as F
from nnAudio import features
from diff_dsp import * 
from utils.processing import householder_matrix as householder
from utils.utility import get_device
from einops import rearrange

device = get_device()

# MODEL/TASK_AGNOSTIC ENCODER 
class Encoder(nn.Module):
    def __init__(self, sr=48000, n_fft=1024):
        super().__init__()
        ''' model/task agnostic decoder '''

        self.stft = features.stft.STFT( n_fft = n_fft,
                                        hop_length = n_fft//8,
                                        freq_scale = 'log',
                                        sr = sr,
                                        fmin=10,
                                        fmax=sr//2, 
                                        output_format = 'Magnitude',
                                        verbose=False)

        self.conv_list = nn.Sequential( nn.Conv2d(1, 64, (7, 5), (1, 2)), nn.ReLU(),
                                        nn.Conv2d(64, 128, 5, (2, 1)), nn.ReLU(),
                                        nn.Conv2d(128, 128, 5, 2), nn.ReLU(),
                                        nn.Conv2d(128, 128, 5, 2), nn.ReLU(),
                                        nn.Conv2d(128, 128, 5, 1), nn.ReLU())
        
        self.gru1 = nn.GRU(input_size=128, hidden_size=64, num_layers=2, 
            batch_first = True, bidirectional=True)
        self.gru2 = nn.GRU(input_size=7168, hidden_size=128, num_layers=1,
            batch_first = True, bidirectional=True)

        self.lin_list = nn.Sequential(  nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
                                        nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU())
    
    def forward(self, x):
        # STFT
        x = torch.log(self.stft(x) + 1e-32) # convert to log-freq log-mag stft 
        x = torch.unsqueeze(x, 1)    # add channel dimension 
        # ENCODER
        x = self.conv_list(x)   # conv layers
        x = torch.permute(x, (0, 3, 2, 1))
        T, F, C = x.size(dim=1), x.size(dim=2), x.size(dim=3)   # get dims
        x = self.gru1(torch.reshape(x, (-1, F, C)))[0]  # frequency GRU
        x = self.gru2(torch.reshape(x, (-1, T, F*C)))[0]    # time GRU
        x = self.lin_list(x)    # linear layers
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
            self.linear2.bias = nn.Parameter(bias)

            # self.linear1.bias.data.fill_(0)
        # in SL it seems that the bias is uneffected
        # else:
        #    self.linear1.bias.data.fill_(0)
        #    self.linear2.bias.data.fill_(0)
        self.activation = activation

    def forward(self, x):
        x = self.linear1(torch.transpose(x, 2, 1))
        y = self.linear2(torch.transpose(x, 2, 1))
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
        
        # initial cutoff freqs such that the freqs are equally spaced in the logarithmic scale 
        omegaK = 2*(omega_min*((omega_max/omega_min)**torch.linspace(0, 1, z2)))
        bias_f = lambda x: torch.log(x/(self.sr-x)) # inverse of the sigmoid function 
        activation_f = lambda x: torch.tan(torch.pi * self.sigmoid(x)/2)
        self.fC1ProjLayer = ProjectionLayer((76, 256), 1, 8, 
                                            bias = bias_f(omegaK),
                                            activation = activation_f)

        self.fCdeltaProjLayer = ProjectionLayer((76, 256), 1, 8, 
                                                bias = bias_f(omegaK),
                                                activation = activation_f)           

        activation_R = lambda x: F.softplus(x) / F.softplus(torch.zeros(1, device=device))
        self.RC1ProjLayer = ProjectionLayer((76, 256), 1, 8,
                                            activation = activation_R) 
        self.RCdeltaProjLayer = ProjectionLayer((76, 256), 1, 8,
                                                activation = activation_R) 

        bias_m = torch.ones((3, 8), device=device)
        bias_m[1, :] = 2*torch.ones((1, 8), device=device)
        self.mC1ProjLayer = ProjectionLayer((76, 256), 3, 8,
                                            bias = bias_m)

        self.GCdeltaProjLayer = ProjectionLayer((76, 256), 1, 8, 
                                                bias = 2*torch.ones((z1, z2), device=device), 
                                                activation = lambda x: 10**(-F.softplus(x-3)))

            

        self.SAProjLayer = ProjectionLayer( (76, 256), 6, 4, 
                                            activation = self.sigmoid)
        self.bcProjLayer = ProjectionLayer((76, 256), 2, 6)
        self.hProjLayer = ProjectionLayer((76, 256), 1, 100)  

        # delay lengths
        self.d = torch.tensor([233, 311, 421, 461, 587, 613],  device=device)
        self.M = self.d.size(0)

        # feedback matrix
        self.Q0 = torch.tensor(householder(self.M), device=device) 

        # all pass filter delay lengths
        self.dAP = torch.tensor([
            [131, 151, 337, 353], 
            [103, 173, 331,373], 
            [89, 181, 307, 401], 
            [79, 197, 281, 419], 
            [61, 211, 257, 431], 
            [47, 229, 251, 443]],  device=device)   
        # length of IR  
        self.ir_length = int(1.8*self.sr)
        
        # normalization term 
        self.bc_norm = nn.Parameter(torch.ones(2,6))
        self.h0_norm = torch.ones((1,1), device=get_device())   # TODO save them!! 
        self.ir_norm = torch.ones((1,1), device=get_device())

    def forward(self, x, z):

        bs = x.size(0)  # batch size

        # Model/Task-Agnostic ENCODER
        x = self.encoder(x) # out: [bs, 109, 256] or [bs, 78, 256]

        # ARP-Groupwise PROJECTION Laters
        bc = self.bcProjLayer(x)                            # input-output gains
        bc = torch.complex(bc, torch.zeros(bc.size(), device=device))
        b, c = bc[:, 0, :], bc[:, 1, :]

        h0 = self.hProjLayer(x).squeeze(dim=1)              # FIR early reflections
        
        fC1 = self.fC1ProjLayer(x).squeeze(dim=1)           # common post filter
        RC1 = self.RC1ProjLayer(x/np.sqrt(8)).squeeze(dim=1)
        mC1 = self.mC1ProjLayer(x/np.sqrt(8))
        C1 = SVF(z, fC1, RC1, mC1[:, 0, :], mC1[:, 1, :], mC1[:, 2, :])
        
        fCdelta = self.fCdeltaProjLayer(x).squeeze(dim=1)   # common parallel delta-coloration filters 
        GCdelta = self.GCdeltaProjLayer(x).squeeze(dim=1)
        RCdelta = self.RCdeltaProjLayer(x).squeeze(dim=1)
        R_bias = 1/torch.sqrt(torch.tensor(2, device=device))
        RCdelta[:,0] = RCdelta[:,0] + R_bias
        RCdelta[:,-1] = RCdelta[:,-1] + R_bias
        Cdelta = PEQ(z, fCdelta, RCdelta, GCdelta)
        Cdelta = Cdelta.expand(self.M, -1, -1).permute(1, 2, 0)

        gamma = self.SAProjLayer(x)                         # channel-wise allpass filters
        U = SAP(z, self.dAP, gamma)
        
        
        D = torch.diag_embed(                               # delay matrix
            torch.unsqueeze(z, dim=-1)  ** (self.d)).expand(bs, -1 ,-1, -1)
        
        Q0 = self.Q0                                        # Householder matrix 

        # FDN Transfer function
        H = torch.einsum('ik,ijkk->ijk', c, torch.inverse(D - torch.diag_embed(U*Cdelta)*Q0 + 1e-16))
        H = C1*torch.einsum('ik,ijk->ij', b, H)

        ir_late =  torch.fft.irfft(H,  norm='ortho')
        h0 = self.h0_norm.expand(bs, 1)*F.pad(h0, (0, self.ir_length-h0.size(dim=1)))
        ir = self.ir_norm.expand(bs, 1)*(h0 + ir_late[:,:self.ir_length])
        return ir, ir_late, h0

    def scatter(self, bs):
        "scattering matrix"
        adm = torch.Tensor(np.ones((bs, self.M)))
        const = 2 / torch.sum(adm ** 2, -1, keepdim = True).unsqueeze(-1)
        scat = const * torch.matmul(adm.unsqueeze(-1), adm.unsqueeze(-2))
        scat = scat - nn.Parameter(torch.eye(self.M))
        return scat
    
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params} \nTotal trainable parameters: {total_train_params}')
    
    def get_filters(self, x, z):
        x = self.encoder(x) # out: [bs, 109, 256]
        # get filters parameters and freuqency response as dictionary 
        parameters = {}
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

