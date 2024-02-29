import torch.nn as nn
import torch
from monai.losses import DiceLoss
import torch.nn.functional as F
from monai.networks.utils import one_hot


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


class VideoSSIMLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ssim_loss = SSIMLoss()

    def forward(self, pred, target, dim=2, reduced=True, as_loss=True):
        assert pred.shape == target.shape, "given shapes of ssim are not equal"

        vid_loss = []
        for i in range(pred.shape[dim]):
            vid_loss.append(self.ssim_loss(pred[:,0,i,...], target[:,0,i,...], reduced=reduced, as_loss=as_loss)) # b, h ,w 

        return sum(vid_loss) / len(vid_loss)

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

class DiceCriterion(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dice_loss = DiceLoss(include_background=False)

    def forward(self, pred, target, as_loss=True):
        loss = self.dice_loss(pred, target)
        if as_loss:
            loss = 1 - loss
        return loss

class TemporalGradLoss(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, flow, mode=1, mask=None):
        dt = flow[:,:, 1:, ...] - flow[:,:, :-1, ...]
        if mode == 2:
            dt = dt[:,:, 1:, ...] - dt[:,:, :-1, ...]
        elif mode == 1:
            pass
        else:
            raise NotImplementedError
        print("dt shape: ", dt.shape)
        eps = 1e-6
        dt = torch.sqrt(dt**2 + eps)
        if mask is not None:
                dt = dt * mask
        return dt.mean()


class SegmentationCriterion(torch.nn.Module):
    def __init__(self, type, loss_weights=None, include_background=False):
        super().__init__()
        
        if type == 'dice':
            self.loss_fct = DiceLoss(reduction='none', include_background=include_background)
        else:
            raise NotImplementedError(f'Loss function {type} is not implemented for segmentation')
    
    def forward(self, pred, target, mask=None, one_hot_enc=True, num_classes=None):
        if one_hot_enc == True and num_classes != None: print("when using one hot, you have to specify the number of classes to encode")
        assert pred.shape == target.shape, f"shapes of pred and target do not match with {pred.shape} {target.shape}"
        if one_hot_enc:
            print(pred.unique(), num_classes, target.unique())
            target = one_hot(target, num_classes=num_classes)
            pred = one_hot(pred, num_classes=num_classes)
        mask = torch.ones_like(pred, dtype=torch.int32)
        mask[:, :, 1, ...] = 0
        mask_ = mask.view(*mask.shape[:2], -1)
        pred_ = pred.view(*pred.shape[:2], -1)
        target_ = target.view(*target.shape[:2], -1)
        loss = 0.0
        for i in range(pred_.shape[0]):
            loss += self.loss_fct(pred_ * mask_, target_ * mask_).squeeze()
        loss = loss / pred.shape[0]
        dice = 1 - loss.detach()
        return loss.mean(), dice