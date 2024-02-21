import torch.nn as nn
import torch
# from monai.losses import DiceLoss

class SimpleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, noises):
        """
        Args:
            outputs (torch.tensor): A tensor of predicted noise.
                shape = (batch, channels, height, width)
                dtype = torch.float32
            noises (torch.tensor): A tensor of ground truth noise.
                shape = (batch, channels, height, width)
                dtype = torch.float32

        Returns:
            loss (torch.tensor): A tensor of simple loss.
                shape = ()
                dtype = torch.flaot32
        """
        loss = (noises - outputs).square().mean()
        return loss


# class SSIMLoss(torch.nn.Module):
#     """
#     SSIM loss module.
#     """

#     def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, as_loss: bool = True):
#         """
#         Args:
#             win_size: Window size for SSIM calculation.
#             k1: k1 parameter for SSIM calculation.
#             k2: k2 parameter for SSIM calculation.
#         """
#         super().__init__()
#         self.win_size = win_size
#         self.k1, self.k2 = k1, k2
#         self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
#         NP = win_size**2
#         self.cov_norm = NP / (NP - 1)
#         self.as_loss = as_loss

#     def forward(
#         self,
#         X: torch.Tensor,
#         Y: torch.Tensor,
#         reduced: bool = True,
#         as_loss: bool = True,
#     ):
#         assert isinstance(self.w, torch.Tensor)
#         if X.is_complex(): X = torch.abs(X)
#         if Y.is_complex(): Y = torch.abs(Y)
#         data_range = torch.max(torch.max(Y,dim=1)[0], dim=1)[0]

#         data_range = data_range[:, None, None, None]
#         C1 = (self.k1 * data_range) ** 2
#         C2 = (self.k2 * data_range) ** 2
#         ux = F.conv2d(X, self.w)  # typing: ignore
#         uy = F.conv2d(Y, self.w)  #
#         uxx = F.conv2d(X * X, self.w)
#         uyy = F.conv2d(Y * Y, self.w)
#         uxy = F.conv2d(X * Y, self.w)
#         vx = self.cov_norm * (uxx - ux * ux)
#         vy = self.cov_norm * (uyy - uy * uy)
#         vxy = self.cov_norm * (uxy - ux * uy)
#         A1, A2, B1, B2 = (
#             2 * ux * uy + C1,
#             2 * vxy + C2,
#             ux**2 + uy**2 + C1,
#             vx + vy + C2,
#         )
#         D = B1 * B2
#         S = (A1 * A2) / D

#         if reduced: S = S.mean()
#         if as_loss: S = 1 - S
#         return S


# class SegmentationCriterion(torch.nn.Module):
#     def __init__(self, type, loss_weights=None):
#         super().__init__()
        
#         if type == 'dice':
#             self.loss_fct = DiceLoss(reduction='none')
#         else:
#             raise NotImplementedError(f'Loss function {type} is not implemented for segmentation')
    
#     def forward(self, pred, target, mask=None):
#         mask = torch.ones_like(pred, dtype=torch.int32)
#         mask[:, :, 1, ...] = 0
#         mask_ = mask.view(*mask.shape[:2], -1)
#         pred_ = pred.view(*pred.shape[:2], -1)
#         target_ = target.view(*target.shape[:2], -1)
#         loss = 0.0
#         for i in range(pred_.shape[0]):
#             loss += self.loss_fct(pred_ * mask_, target_ * mask_).squeeze()
#         loss = loss / pred.shape[0]
#         dice = 1 - loss.detach()
#         return loss.mean(), dice