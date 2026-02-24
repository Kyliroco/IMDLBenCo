import torch
import torch.distributed as dist
from .abstract_class import AbstractEvaluator


class PixelFPR(AbstractEvaluator):
    """Pixel-level False Positive Rate normalised by total image area.

    Computes, per image:

        FPR_pix = FP_pix / (H × W)

    where FP_pix is the number of pixels incorrectly predicted as tampered
    (predicted positive but ground-truth negative) and H × W is the total
    number of pixels in the image.  The metric is returned as a per-image
    tensor from ``batch_update``; the caller (tester) averages it across the
    dataset.

    Reporting the mean over the reference set (rather than computing a single
    global ratio) makes the metric more sensitive to pathological cases where
    a model produces a large contiguous activation on an intact image
    (Guillaro et al., TruFor 2023).

    Parameters
    ----------
    threshold : float
        Binarisation threshold for the predicted mask (default 0.5).
    mode : str
        One of ``"origin"``, ``"reverse"``, or ``"double"``.

        * ``"origin"`` – tampered region is the positive class (standard).
        * ``"reverse"`` – intact region is the positive class; FPR is then
          measured w.r.t. the reversed labelling convention.
        * ``"double"`` – minimum FPR across both conventions.
    """

    def __init__(self, threshold: float = 0.5, mode: str = "origin") -> None:
        super().__init__()
        self.name = "pixel-level FPR"
        self.desc = "pixel-level FPR"
        self.threshold = threshold
        if mode not in ("origin", "reverse", "double"):
            raise ValueError(f"PixelFPR: unknown mode '{mode}'. "
                             "Choose from 'origin', 'reverse', 'double'.")
        self.mode = mode
        # Accumulators: only count samples whose mask IS all-zero (authentic images)
        self._sum = 0.0
        self._count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cal_fpr(self, predict_bin: torch.Tensor,
                 mask: torch.Tensor,
                 shape_mask) -> torch.Tensor:
        """Compute FPR_pix for each image in the batch.

        Parameters
        ----------
        predict_bin : Tensor  [B, 1, H, W]  – already binarised predictions.
        mask        : Tensor  [B, 1, H, W]  – ground-truth binary mask.
        shape_mask  : Tensor or None        – valid-pixel mask (same shape).

        Returns
        -------
        fpr : Tensor [B]  – per-image FPR_pix values.
        """
        if shape_mask is not None:
            FP = torch.sum(predict_bin * (1.0 - mask) * shape_mask,
                           dim=(1, 2, 3))
            total_pixels = torch.sum(shape_mask, dim=(1, 2, 3)).clamp(min=1)
        else:
            FP = torch.sum(predict_bin * (1.0 - mask), dim=(1, 2, 3))
            total_pixels = predict_bin.shape[2] * predict_bin.shape[3]

        return FP / total_pixels

    # ------------------------------------------------------------------
    # AbstractEvaluator interface
    # ------------------------------------------------------------------

    def batch_update(self, predict: torch.Tensor, mask: torch.Tensor,
                     shape_mask=None, *args, **kwargs):
        """Accumulate FPR_pix only for authentic images (all-zero mask).

        Returns None so the tester skips per-batch logging; the final mean
        is returned by ``epoch_update``.
        """
        self._check_pixel_level_params(predict, mask)

        predict_bin = (predict > self.threshold).float()

        if self.mode == "origin":
            fpr = self._cal_fpr(predict_bin, mask, shape_mask)
        elif self.mode == "reverse":
            fpr = self._cal_fpr(1.0 - predict_bin, mask, shape_mask)
        elif self.mode == "double":
            fpr_orig = self._cal_fpr(predict_bin, mask, shape_mask)
            fpr_rev = self._cal_fpr(1.0 - predict_bin, mask, shape_mask)
            fpr = torch.min(fpr_orig, fpr_rev)

        # Only accumulate for authentic images (mask is entirely zero)
        is_authentic = mask.sum(dim=(1, 2, 3)) == 0  # [B]
        valid_fpr = fpr[is_authentic]
        self._sum += valid_fpr.sum().item()
        self._count += valid_fpr.numel()
        return None

    def remain_update(self, predict: torch.Tensor, mask: torch.Tensor,
                      shape_mask=None, *args, **kwargs):
        return self.batch_update(predict, mask, shape_mask, *args, **kwargs)

    def epoch_update(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        t = torch.tensor([self._sum, self._count], dtype=torch.float64, device=device)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total_sum = t[0].item()
        total_count = t[1].item()
        if total_count == 0:
            return 0.0
        return total_sum / total_count

    def recovery(self):
        self._sum = 0.0
        self._count = 0
