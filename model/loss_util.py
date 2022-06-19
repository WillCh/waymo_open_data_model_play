import torch
from torch import nn
from torch.nn import functional as F


class CombinedRegressionClassificationLoss(nn.Module):
    def __init__(self, regression_weight: float = 1.0, 
                 classification_weight:float = 1.0) -> None:
        """Constructor for this class, in which we compute the regression
            and classification loss for the cloest prediction based on L2
            distance.
        
        Args:
            regression_weight: float weight for regression loss.
            classification_weight: float weight for classification loss.

        Returns: None
        """
        super(CombinedRegressionClassificationLoss, self).__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        self.crossentropy_loss = nn.CrossEntropyLoss()

    def forward(
        self, gt: torch.Tensor, pred: torch.Tensor, traj_logit: torch.Tensor,
        ) -> torch.Tensor:
        """Loss function which is linear combination of regression and
            classification. We only compute the regression loss for the
            cloest prediction based on L2 distance.
        
        Args:
            gt: groundtruth tensor, whose dimensions are [B, timestamps,
                attributions]. The attributions are x, y.
            pred: prediction tensor, whose dimensions are [B, num trajs,
                timestamps, attributions]. The attributions are x, y.
            traj_logit: predicted logit, whose dimensions are 
                [B, num trajs].

        Returns:
            1D tensor whose dimensions are [B], representing the final loss.
        """
        batch_size, timestamp, states_dim = gt.shape
        num_trajs = pred.shape[1]
        expected_gt = gt.view(batch_size, 1, timestamp, states_dim).expand(
            batch_size, num_trajs, timestamp, states_dim)
        diff_square = F.mse_loss(
            pred, expected_gt,
            reduction='none')
        sum_square = torch.sum(diff_square, dim=3)
        sum_each_timestamp_distance = torch.sqrt(sum_square)
        mean_traj_distance = torch.mean(sum_each_timestamp_distance, dim=2)
        min_idx = torch.argmin(mean_traj_distance, dim=1)
        regression_loss, _ = torch.min(mean_traj_distance, dim=1)
        regression_loss = torch.mean(regression_loss)
        classification_loss = self.crossentropy_loss(traj_logit, min_idx)
        final_loss = (self.regression_weight * regression_loss +
                      self.classification_weight * classification_loss)
        return final_loss 