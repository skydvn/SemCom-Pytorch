import torch
import torch.nn as nn
import math 

class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr = 20):
        if channel_type not in ['AWGN', 'Rayleigh', 'Rician', 'Nakagami']:
            raise Exception('Unknown type of channel') 
        super(Channel, self).__init__()
        self.channel_type = channel_type
        self.snr = snr

    def forward(self, z_hat):
        # print("sdf of channel ",self.snr)
        #print("dsfw of channel",self.channel_type)
        if z_hat.dim() not in {3, 4}:
            raise ValueError('Input tensor must be 3D or 4D')

        if z_hat.dim() == 3:
            z_hat = z_hat.unsqueeze(0)
        B, C, H, W = z_hat.shape
        k = z_hat[0].numel()
        sig_pwr = torch.sum(torch.abs(z_hat).square(), dim=(1, 2, 3), keepdim=True) / k
        if self.snr is None:
            noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        else:
            noi_pwr = sig_pwr / (10 ** (self.snr / 10))
        noise = torch.randn_like(z_hat) * torch.sqrt(noi_pwr / 2 )

        if self.channel_type == 'Rayleigh':
            hr = torch.sqrt(
                torch.randn(B, device=z_hat.device).pow(2) +
                torch.randn(B, device=z_hat.device).pow(2)
            ) / math.sqrt(2)   
            hi = torch.sqrt(
                torch.randn(B, device=z_hat.device).pow(2) +
                torch.randn(B, device=z_hat.device).pow(2)
            ) / math.sqrt(2)

            # Broadcast thành (B,1,1,1)
            hr = hr.view(B, 1, 1, 1)
            hi = hi.view(B, 1, 1, 1)

            z_hat = z_hat.clone()
            z_hat[:, :z_hat.size(1) // 2] = hr * z_hat[:, :z_hat.size(1) // 2]
            z_hat[:, z_hat.size(1) // 2:] = hi * z_hat[:, z_hat.size(1) // 2:]
        elif self.channel_type == "Rician":
            # For Rician fading, add a deterministic LOS component plus a scattered (Gaussian) term.
            K = 5.0  # Rician K-factor (power ratio of LOS to scattered components)
            los_component = torch.sqrt(torch.tensor(K/(K+1), device=z_hat.device))
            scatter_std = torch.sqrt(torch.tensor(1/(2*(K+1)), device=z_hat.device))
            hr = los_component + scatter_std * torch.randn(B, device=z_hat.device)
            hi = los_component + scatter_std * torch.randn(B, device=z_hat.device)
            hr = hr.view(B,1,1,1); hi = hi.view(B,1,1,1)
            # Áp fading giống hệt Rayleigh
            half = C // 2
            z_hat = z_hat.clone()
            z_hat[:, :half] *= hr
            z_hat[:, half:] *= hi
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
        #print('Noise is', noise)
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