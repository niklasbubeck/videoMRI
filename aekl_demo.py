from generative.networks.nets import AutoencoderKL
from omegaconf import OmegaConf
import torch
import numpy as np
from einops import rearrange
from diffae.dataset import normalize_image_with_mean_lv_value
from diffae.trainer_aekl import PSNR
import imageio
import torch.nn.functional as F


class SSIMLoss(torch.nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, as_loss: bool = True):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)
        self.as_loss = as_loss

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        reduced: bool = True,
        as_loss: bool = True,
    ):
        assert isinstance(self.w, torch.Tensor)
        if X.is_complex(): X = torch.abs(X)
        if Y.is_complex(): Y = torch.abs(Y)
        data_range = torch.max(torch.max(Y,dim=1)[0], dim=1)[0]

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux**2 + uy**2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        if reduced: S = S.mean()
        if as_loss: S = 1 - S
        return S

config = OmegaConf.load('configs/aekl.yaml')
ckpt = torch.load('/home/peter/PycharmProjects/videoMRI/outputs/aekl_s3_l3_b8.pth')
data_sample = np.load('/ssd2t/ukbb/MedMAE/original/1000071/processed_seg_allax.npz')


model = AutoencoderKL(**config.stage1.params).cuda()
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()

sa = data_sample['sax']
sa = normalize_image_with_mean_lv_value(sa)
sa = torch.from_numpy(sa).unsqueeze(0).type(torch.float)     
sa = rearrange(sa, "c h w s t -> c s t h w")
sa5 = sa[:, 5:6, :32, ...].cuda()
sa6 = sa[:, 6:7, :32, ...].cuda()
sa7 = sa[:, 7:8, :32, ...].cuda()

# dummy_sa = torch.randn(1, 1, 32, 128, 128).cuda()
z5 = model.encode_stage_2_inputs(sa5)
recon,_,_ = model(sa5)
z7 = model.encode_stage_2_inputs(sa7)
z6_interpolated = (z5 +z7)/2
recon = model.decode_stage_2_outputs(z6_interpolated)

# print(f'psnr: {PSNR(max_value='on_fly')(recon, sa).item()}')
psnrs = []
ssims = []
for i in range(32):
   ssim = SSIMLoss()(recon[:,0, i].cpu(), sa5[:,0, i].cpu(), as_loss=False)
   psnr = PSNR(max_value='on_fly')(recon[:,:, i], sa5[:,:, i]).item()
   psnrs.append(psnr)
   ssims.append(ssim.item())
image_cat = torch.cat([sa6, recon], dim=3)

# save the gif image
image_cat = image_cat.squeeze().detach().cpu().numpy()

array_uint8 = (image_cat.clip(0, 1) * 255).astype('uint8')
imageio.mimsave('recon.gif', array_uint8, fps=8, duration=0.1)

a = 1
