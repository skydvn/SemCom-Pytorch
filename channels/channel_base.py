import torch
import torch.nn as nn


class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=20):
        if channel_type not in ['AWGN', 'Rayleigh', 'Rician', 'Nakagami']:
            raise Exception('Unknown type of channel')
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr

    def forward(self, z_hat):
        if z_hat.dim() not in {3, 4}:
            raise ValueError('Input tensor must be 3D or 4D')

        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)

        k = z_hat[0].numel()
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2)

        if self.channel_type == 'Rayleigh':
            hc = torch.randn(2, device=z_hat.device)

            z_hat = z_hat.clone()
            z_hat[:, :z_hat.size(1) // 2] = hc[0] * z_hat[:, :z_hat.size(1) // 2]
            z_hat[:, z_hat.size(1) // 2:] = hc[1] * z_hat[:, z_hat.size(1) // 2:]
        elif self.channel_type == "Rician":
            # For Rician fading, add a deterministic LOS component plus a scattered (Gaussian) term.
            K = 5.0  # Rician K-factor (power ratio of LOS to scattered components)
            los_component = torch.sqrt(torch.tensor(K/(K+1), device=z_hat.device))
            scatter_std = torch.sqrt(torch.tensor(1/(2*(K+1)), device=z_hat.device))
            h_r = los_component + scatter_std * torch.randn(2, device=z_hat.device)
            z_hat = z_hat.clone()
            half = z_hat.size(1) // 2
            z_hat[:, :half] = h_r[0] * z_hat[:, :half]
            z_hat[:, half:] = h_r[1] * z_hat[:, half:]
        elif self.channel_type == "Nakagami":
            # For Nakagami fading, generate fading amplitudes from a Gamma distribution.
            m = 2.0    # Nakagami shape factor (m>=0.5; m=1 corresponds to Rayleigh fading)
            omega = 1.0  # Spread parameter (often normalized to 1)
            gamma_dist = torch.distributions.Gamma(m, m/omega)
            # Sample two independent coefficients and take the square root to obtain Nakagami-distributed amplitudes.
            h_n = torch.sqrt(gamma_dist.sample((2,))).to(z_hat.device)
            z_hat = z_hat.clone()
            half = z_hat.size(1) // 2
            z_hat[:, :half] = h_n[0] * z_hat[:, :half]
            z_hat[:, half:] = h_n[1] * z_hat[:, half:]
        else:
            pass
        return z_hat + noise

    def get_channel(self):
        return self.channel_type, self.snr


if __name__ == '__main__':
    # test
    channel = Channel(channel_type='AWGN', snr=10)
    z_hat = torch.randn(64, 10, 5, 5)
    z_hat = channel(z_hat)
    print(f"AWGN: {z_hat}")

    channel = Channel(channel_type='Rayleigh', snr=10)
    z_hat = torch.randn(10, 5, 5)
    z_hat = channel(z_hat)
    print(f"Rayleigh: {z_hat}")

    channel = Channel(channel_type='Rician', snr=10)
    z_hat = torch.randn(10, 5, 5)
    z_hat = channel(z_hat)
    print(f"Rician: {z_hat}")

    channel = Channel(channel_type='Nakagami', snr=10)
    z_hat = torch.randn(10, 5, 5)
    z_hat = channel(z_hat)
    print(f"Nakagami: {z_hat}")