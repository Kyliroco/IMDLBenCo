import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .abstract_class import AbstractEvaluator
import torch.distributed as dist
from sklearn.metrics import precision_recall_curve, auc


class PixelAUPRC(AbstractEvaluator):
    """
    Pixel-level Area Under Precision-Recall Curve metric.
    
    The AUPRC metric evaluates the pixel-level precision and recall trade-off,
    which is particularly useful for imbalanced datasets (like manipulation detection).
    """
    
    def __init__(self, threshold=0.5, mode="origin") -> None:
        self.name = "pixel-level AUPRC"
        self.desc = "pixel-level Area Under Precision-Recall Curve"
        self.threshold = threshold
        self.mode = mode
        # Accumulators: only count samples whose mask is NOT all-zero (tampered images)
        self._sum = 0.0
        self._count = 0

    def Cal_AUPRC(self, y_true, y_scores, shape_mask=None, image_name=None):
        """
        Calculate AUPRC for a single image or batch element.
        
        Args:
            y_true: Ground truth binary labels (1D or multi-dimensional tensor)
            y_scores: Predicted scores (same shape as y_true)
            shape_mask: Optional mask to restrict evaluation to certain regions
            image_name: Optional image identifier for logging
            
        Returns:
            auprc_val: AUPRC score (float)
        """
        name_str = image_name if image_name is not None else "<unknown>"

        if shape_mask is not None:
            y_true = y_true * shape_mask
            y_scores = y_scores * shape_mask
        
        y_true = y_true.flatten()
        y_scores = y_scores.flatten()

        # Handle case where y_true is all zeros (no tampered pixels)
        if torch.sum(y_true) == 0:
            # Return 0.0 for authentic images with no manipulated regions
            return 0.0

        # Exclude parts masked by shape_mask
        if shape_mask is not None:
            valid_mask = shape_mask.flatten() > 0
            y_true = y_true[valid_mask]
            y_scores = y_scores[valid_mask]

        # Convert to numpy for sklearn
        y_true_np = y_true.cpu().numpy().astype(np.int64)
        y_scores_np = y_scores.cpu().numpy()

        # Calculate precision-recall curve
        try:
            precision, recall, _ = precision_recall_curve(y_true_np, y_scores_np)
            # Calculate area under the PR curve
            auprc = auc(recall, precision)
            auprc_val = float(auprc)
        except Exception as e:
            print(
                f"[PixelAUPRC WARNING] Error calculating AUPRC for image '{name_str}': {e}\n"
                f"  y_true unique: {np.unique(y_true_np).tolist()}\n"
                f"  y_scores range: [{y_scores_np.min():.6f}, {y_scores_np.max():.6f}]"
            )
            return float('nan')

        return auprc_val
        
    def batch_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        """
        Update AUPRC metric with a batch of predictions and masks.
        
        Args:
            predict: Model predictions [B, C, H, W]
            mask: Ground truth masks [B, C, H, W]
            shape_mask: Optional shape mask [B, C, H, W]
            **kwargs: Additional arguments (e.g., 'name' for image identifiers)
        """
        self._check_pixel_level_params(predict, mask)
        names = kwargs.get('name', None)
        AUPRC_list = []
        
        if self.mode == "origin":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask is None else shape_mask[idx]
                image_name = names[idx] if names is not None else None
                AUPRC_list.append(self.Cal_AUPRC(mask[idx], predict[idx], single_shape_mask, image_name))
        elif self.mode == "reverse":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask is None else shape_mask[idx]
                image_name = names[idx] if names is not None else None
                AUPRC_list.append(self.Cal_AUPRC(mask[idx], 1 - predict[idx], single_shape_mask, image_name))
        elif self.mode == "double":
            for idx in range(predict.shape[0]):
                single_shape_mask = None if shape_mask is None else shape_mask[idx]
                image_name = names[idx] if names is not None else None
                AUPRC_list.append(max(self.Cal_AUPRC(mask[idx], predict[idx], single_shape_mask, image_name),
                                      self.Cal_AUPRC(mask[idx], 1 - predict[idx], single_shape_mask, image_name)))
        else:
            raise RuntimeError(f"Cal_AUPRC no mode name {self.mode}")

        AUPRC = torch.tensor(AUPRC_list)

        # Only accumulate for tampered images (non-zero mask)
        is_tampered = (mask.sum(dim=(1, 2, 3)) > 0).cpu()  # [B]
        valid_AUPRC = AUPRC[is_tampered]

        # Log any NaN values that slipped through
        nan_mask = torch.isnan(valid_AUPRC)
        if nan_mask.any():
            nan_indices = nan_mask.nonzero(as_tuple=True)[0]
            tampered_indices = is_tampered.nonzero(as_tuple=True)[0]
            for ni in nan_indices:
                orig_idx = tampered_indices[ni].item()
                img_name = names[orig_idx] if names is not None else f"batch_idx={orig_idx}"
                print(
                    f"[PixelAUPRC WARNING] NaN AUPRC detected for tampered image '{img_name}' "
                    f"(batch index {orig_idx}). This image will be excluded from the average."
                )
            valid_AUPRC = valid_AUPRC[~nan_mask]

        self._sum += valid_AUPRC.sum().item()
        self._count += valid_AUPRC.numel()
        return None

    def remain_update(self, predict, mask, shape_mask=None, *args, **kwargs):
        """
        Update with remaining samples (typically from the last batch).
        """
        return self.batch_update(predict, mask, shape_mask, *args, **kwargs)

    def epoch_update(self):
        """
        Calculate and return the final AUPRC value for the epoch across all GPUs.
        """
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
        """
        Reset accumulators for the next epoch.
        """
        self._sum = 0.0
        self._count = 0


# Example usage and testing
if __name__ == "__main__":
    # Generate some example data
    batch_size, channels, height, width = 2, 1, 10, 10
    predict = torch.rand(batch_size, channels, height, width)
    mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    # Generate a shape_mask
    shape_mask = torch.randint(0, 2, (batch_size, channels, height, width)).float()

    auprc = PixelAUPRC()
    reverse_auprc = PixelAUPRC(mode="reverse")
    double_auprc = PixelAUPRC(mode="double")

    auprc.batch_update(predict, mask, shape_mask)
    reverse_auprc.batch_update(predict, mask, shape_mask)
    double_auprc.batch_update(predict, mask, shape_mask)
    
    auprc_value = auprc.epoch_update()
    reverse_auprc_value = reverse_auprc.epoch_update()
    double_auprc_value = double_auprc.epoch_update()

    print(f"PyTorch AUPRC: {auprc_value}")
    print(f"PyTorch Reverse AUPRC: {reverse_auprc_value}")
    print(f"PyTorch Double AUPRC: {double_auprc_value}")
