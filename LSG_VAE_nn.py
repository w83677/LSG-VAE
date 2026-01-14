import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from probts.model.nn.prob.k2VAE.koopman import MLP
from probts.model.nn.prob.k2VAE.RevIN import RevIN



class LSG_VAE(nn.Module):
    """
    Standard VAE with Gaussian Prior N(0, I)
    + Location-Scale observation model:
        For each time step and variable:
            x_t ~ N(mu_t, sigma_t^2)

    Encoder:
        x -> patches -> encoder -> latent mu_z, logvar_z

    Decoder:
        z -> decoder -> [mu_out, logvar_out]
        - mu_out is denormalized via RevIN to obtain the output mean (original scale)
        - logvar_out is used directly as the log variance in original scale
    """

    def __init__(self, config):
        super().__init__()

        # === Basic Parameters ===
        self.config = config
        self.input_len = config.seq_len
        self.patch_len = config.patch_len
        self.multistep = config.multistep
        self.dynamic_dim = config.dynamic_dim  # latent dim H
        self.hidden_layers = config.hidden_layers
        self.hidden_dim = config.hidden_dim
        self.enc_in = config.n_vars             # input channels C

        # === Patching ===
        # number of input patches F
        self.freq = math.ceil(self.input_len / self.patch_len)
        self.pred_len = config.pred_len
        # number of output patches S
        self.step = math.ceil(self.pred_len / self.patch_len)
        self.padding_len = self.patch_len * self.freq - self.input_len

        # === Future Projection ===
        # Project flattened past latents [B, F*H] to flattened future latents [B, S*H]
        # self.future_proj = nn.Linear(
        #     self.freq * self.dynamic_dim,
        #     self.step * self.dynamic_dim
        # )
        self.future_proj = MLP(
            f_in=self.freq * self.dynamic_dim,
            f_out=self.step * self.dynamic_dim,
            activation="relu",
            hidden_dim=self.freq * self.dynamic_dim // 2,
            hidden_layers=1,
        )
        # === Encoder ===
        # Encoder outputs 2 * dynamic_dim (mu_z and logvar_z)
        self.encoder = MLP(
            f_in=self.patch_len * self.enc_in,
            f_out=self.dynamic_dim * 2,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === Decoder ===
        # Decoder outputs 2 * (patch_len * enc_in):
        #   - First half: mean head mu_out
        #   - Second half: logvar head logvar_out
        self.decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=2 * self.patch_len * self.enc_in,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )

        # === RevIN ===
        self.revin = RevIN(self.enc_in)

    # =============================================================
    #               ENCODING
    # =============================================================
    def encode(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            mu_z:      [B, F, H]
            logvar_z:  [B, F, H]
            z:         [B, F, H] (sampled latent)
        """
        B, L, C = x.shape

        # RevIN Normalization (instance-wise)
        x_norm = self.revin(x, "norm")  # [B, L, C]

        # === Patch Padding ===
        if self.padding_len > 0:
            # Pad with a segment from the end of the sequence
            padded = torch.cat(
                [x_norm[:, L - self.padding_len:, :], x_norm],
                dim=1
            )  # [B, F*P, C]
        else:
            padded = x_norm

        # Split into F patches, each of length P
        patches = padded.chunk(self.freq, dim=1)   # freq * [B, P, C]
        patches = torch.stack(patches, dim=1)      # [B, F, P, C]
        patches = patches.reshape(B, self.freq, -1)  # [B, F, P*C]

        # === Encoder Output ===
        encoded = self.encoder(patches)            # [B, F, 2*H]
        mu_z, logvar_z = torch.chunk(encoded, 2, dim=-1)  # [B, F, H], [B, F, H]

        # === Reparameterization ===
        std_z = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std_z)
        z = mu_z + eps * std_z if self.training else mu_z  # [B, F, H]

        return mu_z, logvar_z, z

    # =============================================================
    #           DECODING (Location-Scale)
    # =============================================================
    def decode(self, z: torch.Tensor):
        """
        z: [B, F, H] (Past latent)
        return:
            x_mu:       [B, L, C]  (past mean, denormed)
            y_mu:       [B, pred_len, C] (future mean, denormed)
            x_logvar:   [B, L, C]  (past log variance, original scale)
            y_logvar:   [B, pred_len, C] (future log variance, original scale)
        """
        B = z.shape[0]

        # ----- 1. Reconstruct Past (Location-Scale) -----
        # Decoder outputs 2 * P*C
        x_out = self.decoder(z)                    # [B, F, 2*P*C]
        x_mu_raw, x_logvar = torch.chunk(x_out, 2, dim=-1)  # [B, F, P*C] each

        # Reshape to temporal dimension
        x_mu_raw = x_mu_raw.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_mu_raw = x_mu_raw.reshape(B, -1, self.enc_in)[:, : self.input_len]  # [B, L, C]

        x_logvar = x_logvar.reshape(B, self.freq, self.patch_len, self.enc_in)
        x_logvar = x_logvar.reshape(B, -1, self.enc_in)[:, : self.input_len]  # [B, L, C]

        # Apply RevIN denormalization only to the mean head, aligning x_mu with x in original scale
        x_mu = self.revin(x_mu_raw, "denorm")      # [B, L, C]

        # ----- 2. Predict Future (Location-Scale) -----
        # Flatten past latent: [B, F, H] -> [B, F*H]
        z_flat = z.reshape(B, -1)                  # [B, F*H]
        # Project to future latent: [B, F*H] -> [B, S*H]
        z_future_flat = self.future_proj(z_flat)   # [B, S*H]
        # Reshape to patches: [B, S*H] -> [B, S, H]
        z_future = z_future_flat.reshape(B, self.step, self.dynamic_dim)  # [B, S, H]

        # Decode future latent
        y_out = self.decoder(z_future)             # [B, S, 2*P*C]
        y_mu_raw, y_logvar = torch.chunk(y_out, 2, dim=-1)  # [B, S, P*C]

        y_mu_raw = y_mu_raw.reshape(B, self.step, self.patch_len, self.enc_in)
        y_mu_raw = y_mu_raw.reshape(B, -1, self.enc_in)[:, : self.pred_len]  # [B, pred_len, C]

        y_logvar = y_logvar.reshape(B, self.step, self.patch_len, self.enc_in)
        y_logvar = y_logvar.reshape(B, -1, self.enc_in)[:, : self.pred_len]  # [B, pred_len, C]

        # Similarly, apply RevIN denormalization only to the mean
        y_mu = self.revin(y_mu_raw, "denorm")      # [B, pred_len, C]

        return x_mu, y_mu, x_logvar, y_logvar
       

    # =============================================================
    #                       FORWARD
    # =============================================================
    def forward(self, x: torch.Tensor):
        """
        x: [B, L, C]
        return:
            x_mu:      [B, L, C]        (past mean, original scale)
            y_mu:      [B, pred_len, C] (future mean, original scale)
            mu_z:      [B, F, H]        (latent mean)
            logvar_z:  [B, F, H]        (latent logvar)
            x_logvar:  [B, L, C]        (past log variance, original scale)
            y_logvar:  [B, pred_len, C] (future log variance, original scale)
        """
        mu_z, logvar_z, z = self.encode(x)
        x_mu, y_mu, x_logvar, y_logvar = self.decode(z)
        return x_mu, y_mu, mu_z, logvar_z, x_logvar, y_logvar

    # =============================================================
    #                  GENERATIVE SAMPLING
    # =============================================================
    def sample(self, x: torch.Tensor, num_samples: int = 100):
        """
        Generative Forecasting with Location-Scale observation.

        x: [B, L, C]
        return:
            samples: [B, num_samples, pred_len, C]
        """
        B, _, _ = x.shape

        # 1. Encode to obtain latent posterior parameters
        mu_z, logvar_z, _ = self.encode(x)
        std_z = torch.exp(0.5 * logvar_z)

        sample_list = []
        for _ in range(num_samples):
            # 2. Sample from latent posterior
            eps_z = torch.randn_like(std_z)
            z_sample = mu_z + eps_z * std_z  # [B, F, H]

            # 3. Decode to output mean & logvar
            _, y_mu, _, y_logvar = self.decode(z_sample)  # [B, pred_len, C]

            # 4. Sample from observation distribution N(y_mu, exp(y_logvar))
            std_y = torch.exp(0.5 * y_logvar)
            eps_y = torch.randn_like(std_y)
            y_sample = y_mu + eps_y * std_y               # [B, pred_len, C]

            sample_list.append(y_sample)

        samples = torch.stack(sample_list, dim=1)         # [B, num_samples, pred_len, C]
        return samples

    # =============================================================
    #                    KL DIVERGENCE
    # =============================================================
    def kl_divergence(self, mu, logvar):
        """
        KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

        mu:     [B, F, H]
        logvar: [B, F, H]
        return: scalar
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)  # [B, F]
        return kl.mean()

    # =============================================================
    #                Gaussian NLL for Location-Scale
    # =============================================================
    @staticmethod
    def gaussian_nll(x, mu, logvar, reduction: str = "mean"):
        """
        x, mu, logvar: [B, L, C] or [B, pred_len, C]

        For each element:
            NLL = 0.5 * (log(2π) + logvar + (x-mu)^2 / exp(logvar))

        reduction:
            "mean" -> average over batch
            "sum"  -> sum over batch
            "none" -> return [B] vector
        """
        # Clamp logvar to prevent numerical issues from extreme values
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        nll = 0.5 * (
            math.log(2 * math.pi) + logvar + (x - mu) ** 2 / logvar.exp()
        )  # [B, L, C]
        # Sum over time and variables
        nll = nll.sum(dim=[1, 2])  # [B]

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        else:
            return nll  # [B]