import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted          # 기존 임베딩 재사용
from mamba_ssm import Mamba                              # ⚠️ 경로 호환을 위해 import만

class Model(nn.Module):
    """
    GRU baseline (encoder-only)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len   = configs.seq_len
        self.pred_len  = configs.pred_len
        self.use_norm  = configs.use_norm
        self.output_attention = False      # 다른 모델과의 인터페이스 통일용

        # 1. Embedding  ──────────────────────────────────────
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # 2. GRU encoder  ────────────────────────────────────
        self.encoder = nn.GRU(
            input_size  = configs.d_model,
            hidden_size = configs.d_model,
            num_layers  = configs.e_layers,
            batch_first = True,             # (B, N, E)
            dropout     = configs.dropout if configs.e_layers > 1 else 0.0,
            bidirectional=False
        )

        # 3. Output head  ────────────────────────────────────
        self.projector = nn.Linear(configs.d_model, self.pred_len, bias=True)

    # ------------------------------------------------------------------
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Parameters
        ----------
        x_enc : (B, L, N)
        x_mark_enc : (B, L, d_cov)
        Returns
        -------
        dec_out : (B, pred_len, N)
        """
        # (opt) Non-stationary normalisation
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()                        # (B,1,N)
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = (x_enc - means) / stdev

        B, L, N = x_enc.shape

        # 1) Embedding  (B,L,N) -> (B,N,E)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 2) GRU        (B,N,E) -> (B,N,E)
        enc_out, _ = self.encoder(enc_out)

        # 3) Projector  (B,N,E) -> (B,N,S) -> (B,S,N)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        # 4) De-normalise
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    # ------------------------------------------------------------------
    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        Wrapper to stay compatible with existing Trainer.
        x_dec / x_mark_dec / mask are ignored in this simple baseline.
        """
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]      # (B, L, N)