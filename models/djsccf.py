import torch
import torch.nn as nn


class DJSCCF_CIFAR(nn.Module):
    def __init__(self, args, in_channel, class_num):
        super(DJSCCF_CIFAR, self).__init__()

        self.in_channel = in_channel
        self.class_num = class_num
        self.inv_cdim = args.inv_cdim  # int(32)  # inv_cdim
        self.var_cdim = args.var_cdim  # int(32)  # var_cdim
        self.ib_cdim = self.inv_cdim + self.var_cdim
        self.P = 1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
        self.e = 1e-24

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

    def forward(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        rec = self.decoder(enc)
        return rec

    def get_latent(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc

    def get_semcom_recon(self, x, n_var, device):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        # generate Gaussian propagating noise
        noise = torch.normal(mean=torch.zeros(enc.size()),
                             std=torch.ones(enc.size()) * n_var).to(device)

        var = enc + noise  # simulate a noise by physical channels
        # print(f"lat: {F.mse_loss(var, enc)}")
        rec = self.decoder(var)
        # print(f"ori: {F.mse_loss(x, rec)}")
        return rec

    def get_latent_size(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc.size()

    def normalize_layer(self, z):
        k = torch.tensor(1.0).to(self.device)  # torch.prod(torch.tensor(z.size()[1:], dtype=torch.float32))
        # Square root of k and P
        sqrt1 = torch.sqrt(k * self.P)

        # Conjugate Transpose of z
        if torch.is_complex(z):
            zT = torch.conj(z).permute(0, 1, 3, 2)
        else:
            zT = z.permute(0, 1, 3, 2)
        # Multiply z and zT = sqrt2
        sqrt2 = torch.sqrt(zT*z + self.e)
        div = z / sqrt2
        z_out = div * sqrt1

        return z_out  # Adjusted return value as per PyTorch operations


class DJSCCNLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.rec_loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

    def forward(self, args, rec, img):
        rec_loss = self.rec_loss(rec, img)

        cls_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)
        inv_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)
        var_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)
        irep_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)
        kld_loss = torch.tensor(0.0, device='cuda:0', requires_grad=True)

        psnr_val = 20 * torch.log10(torch.max(img) / torch.sqrt(rec_loss))

        total_loss = args.rec_coeff * rec_loss

        return {
            "cls_loss": cls_loss,
            "rec_loss": rec_loss,
            "psnr_loss": psnr_val,
            "kld_loss": kld_loss,
            "inv_loss": inv_loss,
            "var_loss": var_loss,
            "irep_loss": irep_loss,
            "total_loss": total_loss
        }
