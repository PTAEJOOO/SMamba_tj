import torch
import torch.nn as nn
from layers.Mamba_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted, LinearEmbedding
import torch.nn.functional as F

from mamba_ssm import Mamba
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        ### Intra-time series modeling ## 
        ## Time embedding
        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, configs.te_dim-1)
        
        ## TTCN
        input_dim = 1 + configs.te_dim
        ttcn_dim = configs.hid_dim - 1
        self.ttcn_dim = ttcn_dim
        self.Filter_Generators = nn.Sequential(
				nn.Linear(input_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, ttcn_dim, bias=True),
				nn.ReLU(inplace=True),
				nn.Linear(ttcn_dim, input_dim*ttcn_dim, bias=True))
        self.T_bias = nn.Parameter(torch.randn(1, ttcn_dim))

        # Embedding
        # self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
        #                                             configs.dropout) # c_in = configs.seq_len, d_model = configs.d_model
        self.enc_embedding = LinearEmbedding(configs.npatch*configs.hid_dim, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout) # c_in = configs.seq_len, d_model = configs.d_model
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                        Mamba(
                            d_model=configs.d_model,  # Model dimension d_model
                            d_state=configs.d_state,  # SSM state expansion factor
                            d_conv=2,  # Local convolution width
                            expand=1,  # Block expansion factor)
                        ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        ### Decoder ###
        self.decoder = nn.Sequential(
            nn.Linear(configs.d_model+configs.te_dim, configs.hid_dim),
            nn.ReLU(inplace=True),
			# nn.Linear(configs.hid_dim, configs.hid_dim),
			# nn.ReLU(inplace=True),
			nn.Linear(configs.hid_dim, 1)
			)
    
    def LearnableTE(self, tt):
        out1 = self.te_scale(tt) # (B*N*M, L, 1)
        out2 = torch.sin(self.te_periodic(tt)) # (B*N*M, L, 9)
        return torch.cat([out1, out2], -1)
    
    def TTCN(self, X_int, mask_X):
        # X_int: shape (B*N*M, L, F)
		# mask_X: shape (B*N*M, L, 1)

        N, Lx, _ = mask_X.shape
        Filter = self.Filter_Generators(X_int) # (B*N*M, L, F*ttcn_dim)
        Filter_mask = Filter * mask_X + (1 - mask_X) * (-1e8)
		# normalize along with sequence dimension
        Filter_seqnorm = F.softmax(Filter_mask, dim=-2)  # (B*N*M, L, F*ttcn_dim)
        Filter_seqnorm = Filter_seqnorm.view(N, Lx, self.ttcn_dim, -1) # (B*N*M, L, ttcn_dim, F)
        X_int_broad = X_int.unsqueeze(dim=-2).repeat(1, 1, self.ttcn_dim, 1) # (B*N*M, L, ttcn_dim, F)
        ttcn_out = torch.sum(torch.sum(X_int_broad * Filter_seqnorm, dim=-3), dim=-1) # (B*N*M, ttcn_dim)
        h_t = torch.relu(ttcn_out + self.T_bias) # (B*N*M, ttcn_dim)
        return h_t

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        B, M, L_in, N = x_enc.shape
        # B: batch_size;    E: d_model;     M: patch num;
        # L_in: seq_len;    S: pred_len;
        # N: number of variate (tokens), can also includes covariates
        """ Embedding """
        x_enc = x_enc.permute(0, 3, 1, 2).reshape(-1, L_in, 1) # (B*N*M, L, 1)
        x_mark_enc = x_mark_enc.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
        mask = mask.permute(0, 3, 1, 2).reshape(-1, L_in, 1)  # (B*N*M, L, 1)
        te_his = self.LearnableTE(x_mark_enc) # (B*N*M, L, F_te)
        x_enc = torch.cat([x_enc, te_his], dim=-1)  # (B*N*M, L, F)

        # mask for the patch
        mask_patch = (mask.sum(dim=1) > 0)

        ### TTCN for patch modeling ###
        x_patch = self.TTCN(x_enc, mask) # (B*N*M, hid_dim-1)
        x_patch = torch.cat([x_patch, mask_patch],dim=-1) # (B*N*M, hid_dim)
        # x_patch = x_patch.view(self.batch_size, self.N, self.M, -1) # (B, N, M, hid_dim)
        x_patch = x_patch.view(B, N, -1) # (B, N, M*hid_dim)
        
        # (B, N, M*hid_dim) -> B N E
        enc_out = self.enc_embedding(x_patch, x_mark=None) # covariates (e.g timestamp) can be also embedded as tokens

        """ SMamba encoding """
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        """ Decoder """
        L_pred = x_mark_dec.shape[-1] # Lp
        h = enc_out.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1) # (B, N, Lp, E)
        x_mark_dec = x_mark_dec.view(B, 1, L_pred, 1).repeat(1, N, 1, 1) # (B, N, Lp, 1)
        te_pred = self.LearnableTE(x_mark_dec) # (B, N, Lp, F_te)
        h = torch.cat([h, te_pred], dim=-1) # (B, N, Lp, E+F_te)

        # (B, N, Lp, E+F_te) -> (B, N, Lp, 1) -> (1, B, Lp, N)
        dec_out = self.decoder(h).squeeze(dim=-1).permute(0, 2, 1) #.unsqueeze(dim=0)
        # dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        # dec_out[:, -self.pred_len:, :]
        return dec_out  # [B, L, D]