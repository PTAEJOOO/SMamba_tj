import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted        # S-D-Mamba와 동일한 임베딩
from mamba_ssm import Mamba                            # ✨ 사용하지 않지만 import 경로 통일용

class Model(nn.Module):
    """
    Simple LSTM baseline that keeps exactly the same
    · input / output signature
    · embedding & (de)normalisation logic
    as the S-D-Mamba / Transformer implementations.

    Workflow
    --------
    1. DataEmbedding_inverted : (B, L, N) → (B, N, E)
       – 각 변수(token)마다 과거 seq_len 길이의 시계열을 압축해 d_model 차원 임베딩을 생성
    2. LSTM (batch_first=True) : (B, N, E) → (B, N, E)
       – token 차원(N) 을 ‘sequence length’ 로 간주해 간단한 시퀀스 모델링
    3. Linear projector       : (B, N, E) → (B, N, pred_len)
       – 시간축(pred_len) 생성
    4. Output permutation      : (B, N, S) → (B, S, N)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # 기본 하이퍼
        self.seq_len   = configs.seq_len
        self.pred_len  = configs.pred_len
        self.use_norm  = configs.use_norm
        self.output_attention = False          # LSTM baseline은 attention 반환 없음

        # ───────────────────── Embedding ─────────────────────
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model,
            configs.embed, configs.freq, configs.dropout
        )

        # ───────────────────── LSTM Encoder ───────────────────
        self.encoder = nn.LSTM(
            input_size  = configs.d_model,
            hidden_size = configs.d_model,
            num_layers  = configs.e_layers,
            batch_first = True,                # (B, N, E)
            dropout     = configs.dropout if configs.e_layers > 1 else 0.0,
            bidirectional=False
        )

        # ───────────────────── Output head ────────────────────
        self.projector = nn.Linear(configs.d_model, self.pred_len, bias=True)

    # ---------------------------------------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Parameters
        ----------
        x_enc : (B, L, N)   – 원 시계열
        x_mark_enc : (B, L, d_cov) – timestamp 등 covariate
        Returns
        -------
        dec_out : (B, pred_len, N)
        """
        # 1) Optional non-stationary normalisation
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()                    # (B,1,N)
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = (x_enc - means) / stdev

        B, L, N = x_enc.shape

        # 2) Embedding  (B,L,N) -> (B,N,E)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 3) LSTM       (B,N,E) -> (B,N,E)
        enc_out, _ = self.encoder(enc_out)          # ignore hidden states

        # 4) Projector  (B,N,E) -> (B,N,S) -> (B,S,N)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        # 5) De-normalise
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    # ---------------------------------------------------------
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward wrapper to keep the same signature as other models.
        Only the enc-part is used; dec inputs are ignored.
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]        # [B, L, N]