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
            pass
        elif self.channel_type == "Nakagami":
            pass
        
        return z_hat + noise

    def get_channel(self):
        return self.channel_type, self.snr


if __name__ == '__main__':
    # test
    channel = Channel(channel_type='AWGN', snr=10)
    z_hat = torch.randn(64, 10, 5, 5)
    z_hat = channel(z_hat)
    print(z_hat)

    channel = Channel(channel_type='Rayleigh', snr=10)
    z_hat = torch.randn(10, 5, 5)
    z_hat = channel(z_hat)
    print(z_hat)