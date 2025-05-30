import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
from channels.channel_base import Channel
from models.model_base import BaseModel


class DJSCCF_CIFAR(BaseModel):
    def __init__(self, args, in_channel, class_num, latent_size=32, beta=1):
        super(DJSCCF_CIFAR, self).__init__(args, in_channel, class_num)
        self.channel = Channel(channel_type="AWGN", snr = args.base_snr)
        self.latent_size = latent_size
        self.beta = beta
        self.encoder = nn.Sequential(
            self._conv(3, 32), # B, 32, 16, 16
            self._conv(32, 32), # B, 32, 8, 8 
            self._conv(32, 64), # B, 64, 4, 4
            self._conv(64, 64), # B, 64, 2, 2
        )
        self.fc_mu = nn.Linear(256, latent_size) 
        self.fc_var = nn.Linear(256, latent_size)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(64, 64),
            self._deconv(64, 32),
            self._deconv(32, 32),
            self._deconv(32, 3),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_size, 256)

    def encode(self, x):
        x = self.encoder(x) # B, 64, 2, 2
        x = x.view(-1, 256) # Bx64x2x2/256, 256 
        return self.fc_mu(x), self.fc_var(x) # Bx64x2x2/256 , 32 

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    # def decode(self, z):
    #     z = self.fc_z(z)
    #     z = z.view(-1, 64, 2, 2)
    #     return self.decoder(z)

    def forward(self, x):
        #print("Input shape:", x.shape)
        
        x = self.encoder(x)
        B, C, H, W = x.shape
        x = x.view(-1, 256) # BxC(64)xHxW / 256, 256
        #print("Shape after flattening", x.shape)
        #print("Shape after fc", self.fc_mu(x).shape)
        mu = self.fc_mu(x) # 128, 32 
        logvar = self.fc_var(x)
        #print("Mu shape:", mu.shape)
        z = self.sample(mu, logvar)
        #print("Z shape:", z.shape)
        z = self.fc_z(z) # 128, 256
        z = z.reshape(B, 64, H, W) # 128, 64, 2, 2
        z = self.normalize_layer(z)
        z = self.channel(z)
        #print("Z shape after channel:", z.shape)
        rx = self.decoder(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, padding = 1,
                kernel_size=4, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    # out_padding is used to ensure output size matches EXACTLY of conv2d;
    # it does not actually add zero-padding to output :)
    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding = 1,output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='mean') # sum -> mean 
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #print(f"recon_loss: {recon_loss}, kl_diverge: {kl_diverge}")
        return (recon_loss + self.beta * kl_diverge) / x.shape[0]
    def normalize_layer(self, z):
        k = torch.tensor(1.0).to(self.device)  # torch.prod(torch.tensor(z.size()[1:], dtype=torch.float32))
        # Square root of k and P
        sqrt1 = torch.sqrt(k * self.P)
        sqrt2 = torch.sqrt(z*z + self.e)
        div = z / sqrt2
        z_out = div * sqrt1

        return z_out