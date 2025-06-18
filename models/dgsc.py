import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from channels.channel_base import Channel
from models.model_base import BaseModel
# from backpack import backpack, extend
# from backpack.extensions import BatchGrad

class DGSC_CIFAR(BaseModel):
    def __init__(self, args, in_channel, class_num):
        super(DGSC_CIFAR, self).__init__(args, in_channel, class_num)
        self.base_snr = args.base_snr 
        self.channel_type = args.channel_type 
        print(self.base_snr)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=self.var_cdim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        #self.encoder = extend(self.encoder)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.var_cdim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, in_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SELU(),
        )

    def forward(self, x,snr_chan):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        self.change_channel(channel_type=self.channel_type,snr = snr_chan)
        if self.channel is None:
            print("No Channel")
            #print("Channel is: ", self.channel.get_channel())
        enc = self.channel(enc)
        rec = self.decoder(enc)
        return rec

    def get_latent(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc


    def get_latent_size(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc.size() 

    def normalize_layer(self, z):
        k = torch.tensor(1.0).to(self.device)  # torch.prod(torch.tensor(z.size()[1:], dtype=torch.float32))
        # Square root of k and P
        sqrt1 = torch.sqrt(k * self.P)

        # Conjugate Transpose of z
        # if torch.is_complex(z):
        #     zT = torch.conj(z).permute(0, 1, 3, 2)
        # else:
        #     zT = z.permute(0, 1, 3, 2)
        # Multiply z and zT = sqrt2
        sqrt2 = torch.sqrt(z*z + self.e)
        div = z / sqrt2
        z_out = div * sqrt1
        return z_out

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None 

    def channel_perturb(self, x, chan_type, snr_chan):
        
        z = self.encoder(x)
        self.change_channel(channel_type = chan_type, snr = snr_chan)
        z_after_channel = self.channel(z)
        rec = self.decoder(z_after_channel)
        return rec 



# import torch
# import torch.nn as nn
# import numpy as np
# import torch.nn.functional as F
# from channels.channel_base import Channel
# from models.model_base import BaseModel




# class DGSC_CIFAR(BaseModel):
#     def __init__(self, args, in_channel, class_num,
#                  latent_size=100, beta=1.0):
#         super().__init__(args, in_channel, class_num)
#         self.latent_size = latent_size
#         self.beta = beta
#         self.base_snr = args.base_snr
#         self.batch    = args.bs

#         # Encoder → cuối ra B×256×2×2
#         self.e1 = self._conv(3,   32)   # →16×16
#         self.e2 = self._conv(32,  64)   # → 8×8
#         self.e3 = self._conv(64, 128)   # → 4×4
#         self.e4 = self._conv(128,256)   # → 2×2

#         # FC cấp mu/logvar từ 256·2·2 = 1024
#         self.fc_mu  = nn.Linear(256*2*2, latent_size)
#         self.fc_var = nn.Linear(256*2*2, latent_size)

#         # FC đưa latent về 256·2·2 = 1024
#         self.fc_z   = nn.Linear(latent_size, 256*2*2)

#         # Decoder: mỗi upconv chèn một lần Upsample
#         self.up1 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),  # 2→4
#             self._upconv(256, 128)                       # → B×128×4×4
#         )
#         self.up2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),  # 4→8
#             self._upconv(128, 64)                         # → B×64×8×8
#         )
#         self.up3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),  # 8→16
#             self._upconv(64, 32)                          # → B×32×16×16
#         )
#         self.up4 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='nearest'),  # 16→32
#             self._upconv(32, in_channel)                  # → B×3×32×32
#         )

#     def _conv(self, in_c, out_c):
#         return nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#     def _upconv(self, in_c, out_c):
#         return nn.Sequential(
#             nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_c),
#             nn.LeakyReLU(0.2, inplace=True)
#         )

#     def encode(self, x):
#         h = F.leaky_relu(self.e1(x))
#         h = F.leaky_relu(self.e2(h))
#         h = F.leaky_relu(self.e3(h))
#         h = F.leaky_relu(self.e4(h))          # → (B,256,2,2)
#         h_flat = h.view(self.batch, -1)       # → (B,1024)
#         return self.fc_mu(h_flat), self.fc_var(h_flat)

#     def sample(self, mu, logvar):
#         std = (0.5*logvar).exp()
#         eps = torch.randn_like(std)
#         return mu + eps * std                 # (B,latent_size)

#     def map_z(self, z_flat):
#     # latent → feature-map
#         z_map = self.fc_z(z_flat)            # (B, 256*2*2)
#         return z_map.view(-1, 256, 2, 2)     # (B,256,2,2)

#     def decode(self, z4d):
#     # chỉ decode 4-D → ảnh, không fc_z
#         z = self.up1(z4d)    # 2→4
#         z = self.up2(z)      # 4→8
#         z = self.up3(z)      # 8→16
#         z = self.up4(z)      #16→32
#         return torch.sigmoid(z)

#     def forward(self, x):
#         mu, logvar = self.encode(x)
#         z_flat = self.sample(mu, logvar)
#         z4d = self.map_z(z_flat)
#         if self.channel: z4d = self.channel(z4d)
#         rx = self.decode(z4d)
#         return rx, mu, logvar
#     def loss(self, recon_x, x, mu, logvar):
#         # reconstruction losses are summed over all elements and batch
#         recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

#         # see Appendix B from VAE paper:
#         # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#         # https://arxiv.org/abs/1312.6114
#         # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#         kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#         return (recon_loss + self.beta * kl_diverge) / x.shape[0]

#     def change_channel(self, channel_type='AWGN', snr=None):
#         if snr is None:
#             self.channel = None
#         else:
#             self.channel = Channel(channel_type, snr)

#     def channel_perturb(self, x, domain_str):
#         mu, logvar = self.encode(x)         # (B,latent)
#         z_flat = self.sample(mu, logvar)    # (B,latent)
#         z4d    = self.map_z(z_flat)         # (B,256,2,2)

#         self.change_channel(channel_type=domain_str, snr=self.base_snr)
#         z4d = self.channel(z4d)             # (B,256,2,2)

#         recon_x = self.decode(z4d)          # (B,3,32,32)
#         return recon_x, mu, logvar
